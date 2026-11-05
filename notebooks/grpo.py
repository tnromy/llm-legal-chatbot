# %% [markdown]
# # GRPO: Mengajari Model "Berpikir" sebelum Menjawab (Kriteria 1 — Advanced)
#
# **Proyek Akhir — Pengembangan Generative AI berbasis LLM (Dicoding)** · Romi Fadli
#
# Model instruct buatan kita sendiri (`tnromy/Qwen2.5-1.5B-Legal-ID-SFT`, hasil notebook Fine-tuning)
# dilatih lanjut dengan **Group Relative Policy Optimization** (TRL `GRPOTrainer` + Unsloth) agar
# selalu menuliskan proses penalaran di dalam `<think> ... </think>` sebelum memberikan jawaban akhir
# berbahasa Indonesia. Dataset tetap `Ichsan2895/alpaca-gpt4-indonesian`.
#
# Empat reward function:
#
# | Reward | Aturan |
# |---|---|
# | `format_reward_func` | `<think>` +0.2, `</think>` +0.3, format sempurna = +1.0; tag ganda −0.5 |
# | `reasoning_length_reward` | isi `<think>`: kosong 0.0, <50 karakter 0.2, 50–199 0.5, ≥200 1.0 (toleran bila terpotong) |
# | `correctness_reward` | +1.0 bila jawaban akhir memuat ground truth atau ROUGE-L ≥ 0.3 terhadap kolom `output` |
# | `language_reward_func` | −0.5 bila jawaban berbahasa Inggris, +1.0 bila murni bahasa Indonesia |

# %% [markdown]
# ## 1. Instalasi & konfigurasi

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

# %%
import os, re, random
import torch
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
SFT_MODEL = "tnromy/Qwen2.5-1.5B-Legal-ID-SFT"
HF_REPO = "tnromy/Qwen2.5-1.5B-Legal-ID-GRPO"
MAX_SEQ_LEN = 1024
print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Memuat kembali model instruct buatan sendiri (QLoRA 4-bit + double quantization)

# %%
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_MODEL, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True,
    quantization_config=bnb_config, token=os.environ["HF_TOKEN"])
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)
model.print_trainable_parameters()

# %% [markdown]
# ## 3. Dataset prompt untuk GRPO
# Dipilih contoh dengan jawaban yang tidak terlalu panjang agar `correctness_reward` dapat dinilai dalam
# batas `max_completion_length`. Prompt yang terlalu panjang dibuang (pengganti `max_prompt_length`).

# %%
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Anda adalah asisten AI berbahasa Indonesia. Sebelum menjawab, tuliskan proses berpikir Anda "
    "di antara tag <think> dan </think>, kemudian tuliskan jawaban akhir dalam bahasa Indonesia.\n"
    "Format:\n<think>\n...penalaran langkah demi langkah...\n</think>\n...jawaban akhir..."
)

raw = load_dataset("Ichsan2895/alpaca-gpt4-indonesian", split="train").shuffle(seed=SEED)
raw = raw.filter(lambda r: r["input"] and r["output"] and len(r["input"]) < 400 and 30 < len(r["output"]) < 400)
raw = raw.select(range(800))

def to_prompt(r):
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": r["input"].strip()}],
            "answer": r["output"].strip()}

grpo_ds = raw.map(to_prompt, remove_columns=raw.column_names)
lengths = [len(tokenizer.apply_chat_template(p, add_generation_prompt=True)) for p in grpo_ds["prompt"]]
grpo_ds = grpo_ds.select([i for i, n in enumerate(lengths) if n <= 256])
print(grpo_ds)
print(grpo_ds[0])

# %% [markdown]
# ## 4. Reward functions

# %%
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

def _text(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion

def split_think(text):
    """Kembalikan (isi_think, jawaban_akhir). Toleran bila </think> terpotong batas token."""
    if THINK_OPEN not in text:
        return None, text.strip()
    after = text.split(THINK_OPEN, 1)[1]
    if THINK_CLOSE in after:
        thought, final = after.split(THINK_CLOSE, 1)
        return thought.strip(), final.strip()
    return after.strip(), ""

def format_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        t = _text(c)
        score = 0.0
        if THINK_OPEN in t:
            score += 0.2
        if THINK_CLOSE in t:
            score += 0.3
        perfect = re.match(r"^\s*<think>.+?</think>\s*\S", t, flags=re.S)
        if perfect and t.count(THINK_OPEN) == 1 and t.count(THINK_CLOSE) == 1:
            score = 1.0
        if t.count(THINK_OPEN) > 1 or t.count(THINK_CLOSE) > 1:
            score -= 0.5
        rewards.append(score)
    return rewards

def reasoning_length_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        thought, _ = split_think(_text(c))
        if not thought:
            rewards.append(0.0)
        elif len(thought) < 50:
            rewards.append(0.2)
        elif len(thought) < 200:
            rewards.append(0.5)
        else:
            rewards.append(1.0)
    return rewards

def _tokens(s):
    return re.findall(r"\w+", s.lower())

def rouge_l_f(candidate, reference):
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)

def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for c, gt in zip(completions, answer):
        _, final = split_think(_text(c))
        final = final or _text(c)
        exact = " ".join(_tokens(gt)) in " ".join(_tokens(final))
        rewards.append(1.0 if exact or rouge_l_f(final, gt) >= 0.3 else 0.0)
    return rewards

EN_WORDS = set("the and is are of to in that it for with as was this be on not by or from at which you your".split())
ID_WORDS = set("yang dan di ke dari ini itu dengan untuk adalah pada dalam tidak akan atau juga karena".split())

def language_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        _, final = split_think(_text(c))
        words = _tokens(final or _text(c))
        en = sum(w in EN_WORDS for w in words)
        idn = sum(w in ID_WORDS for w in words)
        if en > idn or en >= 3:
            rewards.append(-0.5)
        elif idn > 0 and en == 0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

REWARDS = [format_reward_func, reasoning_length_reward, correctness_reward, language_reward_func]

# %% [markdown]
# Uji cepat reward function pada beberapa contoh buatan.

# %%
samples = [
    "<think>Pengguna bertanya tentang ibu kota. Ibu kota Indonesia adalah Jakarta, sesuai fakta umum yang "
    "diketahui banyak orang dan tercatat dalam berbagai sumber resmi pemerintahan maupun buku pelajaran.</think>"
    " Ibu kota Indonesia adalah Jakarta.",
    "<think>singkat</think> The capital is Jakarta.",
    "Ibu kota Indonesia adalah Jakarta.",
    "<think>a</think><think>b</think> Jakarta",
    "<think>Penalaran ini terpotong karena batas token sehingga tag penutup tidak sempat",
]
comps = [[{"role": "assistant", "content": s}] for s in samples]
gts = ["Ibu kota Indonesia adalah Jakarta."] * len(samples)
import pandas as pd
pd.DataFrame({f.__name__: f(comps, answer=gts) for f in REWARDS}, index=[s[:45] for s in samples])

# %% [markdown]
# ## 5. Training GRPO
# `num_generations=6` dan `max_completion_length=320` dipilih agar muat di VRAM T4 (16 GB) tanpa OOM.
# Model SFT hampir tidak pernah menulis `<think>` sendiri, sehingga sinyal reward awal sangat jarang.
# Karena itu learning rate dibuat cukup besar untuk LoRA (3e-5), temperature 1.0 untuk eksplorasi,
# dan 6 generasi per prompt agar setiap grup lebih mungkin berisi minimal satu jawaban berformat benar.

# %%
from trl import GRPOConfig, GRPOTrainer

args = GRPOConfig(
    output_dir="outputs/grpo",
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_completion_length=320,
    max_steps=200,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=6,
    temperature=1.0,
    optim="adamw_8bit",
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=REWARDS,
                      train_dataset=grpo_ds, args=args)
train_result = trainer.train()
print(train_result)

# %%
import matplotlib.pyplot as plt

hist = pd.DataFrame(trainer.state.log_history).dropna(subset=["reward"])
cols = [c for c in hist.columns if c.startswith("rewards/") and c.endswith("/mean")]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(hist["step"], hist["reward"], marker="o"); axes[0].set_title("Total reward")
for c in cols:
    axes[1].plot(hist["step"], hist[c], label=c.split("/")[1])
axes[1].set_title("Reward per fungsi"); axes[1].legend()
for ax in axes:
    ax.set_xlabel("step"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
hist[["step", "reward"] + cols].tail(10)

# %% [markdown]
# ## 6. Uji model hasil GRPO
# Dua mode diuji:
# 1. **Bebas** — model menulis responsnya sendiri dari awal.
# 2. **Reasoning trigger** — giliran asisten diawali `<think>\n` (praktik yang direkomendasikan untuk
#    model reasoning seperti DeepSeek-R1). Generasi dihentikan bila model mulai membuka `<think>` kedua.
#    Mode ini yang dipakai di pipeline RAG.

# %%
from transformers import StoppingCriteria, StoppingCriteriaList

FastLanguageModel.for_inference(model)

class StopOnSecondThink(StoppingCriteria):
    def __init__(self, start): self.start = start
    def __call__(self, input_ids, scores, **kw):
        t = tokenizer.decode(input_ids[0][self.start:], skip_special_tokens=True)
        return THINK_CLOSE in t and THINK_OPEN in t.split(THINK_CLOSE, 1)[1]

def ask(question, trigger=False, max_new_tokens=400):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prefix = THINK_OPEN + "\n" if trigger else ""
    ids = tokenizer(text + prefix, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True,
                         repetition_penalty=1.1,
                         stopping_criteria=StoppingCriteriaList([StopOnSecondThink(ids.shape[1])]))
    resp = prefix + tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    if THINK_CLOSE in resp:  # buang blok <think> tambahan yang terlanjur dimulai
        head, tail = resp.split(THINK_CLOSE, 1)
        resp = head + THINK_CLOSE + tail.split(THINK_OPEN, 1)[0]
    return resp.strip()

for q in ["Jelaskan perbedaan antara pekerja tetap dan pekerja kontrak.",
          "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"]:
    for trigger in (False, True):
        resp = ask(q, trigger)
        comp = [[{"content": resp}]]
        scores = {f.__name__: f(comp)[0] for f in (format_reward_func, reasoning_length_reward, language_reward_func)}
        print(f"### {q}  [{'reasoning trigger' if trigger else 'bebas'}]\n{resp}\n{scores}\n")

# %% [markdown]
# ## 7. Push model GRPO ke Hugging Face Hub (`merged_16bit`)

# %%
model.push_to_hub_merged(HF_REPO, tokenizer, save_method="merged_16bit", token=os.environ["HF_TOKEN"])
from huggingface_hub import HfApi
info = HfApi(token=os.environ["HF_TOKEN"]).model_info(HF_REPO)
print(f"https://huggingface.co/{HF_REPO} | private: {info.private}")
