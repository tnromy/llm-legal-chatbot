# %% [markdown]
# # Fine-tuning SLM untuk Chatbot Tim Legal (Kriteria 1)
#
# **Proyek Akhir — Pengembangan Generative AI berbasis LLM (Dicoding)** · Romi Fadli
#
# Notebook ini menyiapkan "otak" chatbot: sebuah *Small Language Model* yang di-*fine-tune* dengan
# **QLoRA** (Unsloth + TRL `SFTTrainer`) menggunakan dataset instruksi berbahasa Indonesia
# [`Ichsan2895/alpaca-gpt4-indonesian`](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian).
#
# | Komponen | Pilihan |
# |---|---|
# | Base model | `Qwen/Qwen2.5-1.5B` (model *pre-trained* resmi Qwen untuk text generation, didukung Unsloth) |
# | Chat template | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
# | Kuantisasi | 4-bit NF4 + **double quantization** (QLoRA) |
# | Adapter LoRA | Multi-Head Attention (`q,k,v,o_proj`) **dan** Feed Forward Network (`gate,up,down_proj`) |
# | Training | `SFTTrainer` 800 steps × 2 eksperimen, dengan validation set |
# | Output | Model terbaik di-*merge* (16-bit) dan di-*push* ke Hugging Face Hub |

# %% [markdown]
# ## 1. Instalasi & konfigurasi
# Token Hugging Face **tidak ditulis di notebook**. Token dibaca dari *Kaggle Secrets* lalu disimpan
# sebagai *environment variable* `HF_TOKEN`.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

# %%
import os, gc, json, math, random
import torch
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
assert os.environ["HF_TOKEN"], "HF_TOKEN belum tersedia"

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
HF_REPO = "tnromy/Qwen2.5-1.5B-Legal-ID-SFT"
MAX_SEQ_LEN = 1024
print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Dataset: dari format Alpaca mentah ke Chat Template
# Dataset berisi kolom `input` (instruksi) dan `output` (jawaban). Kolom indeks `Unnamed: 0` dibuang.

# %%
from datasets import load_dataset

raw = load_dataset("Ichsan2895/alpaca-gpt4-indonesian", split="train")
raw = raw.remove_columns([c for c in raw.column_names if c.startswith("Unnamed")])
raw = raw.filter(lambda r: r["input"] and r["output"] and len(r["input"]) + len(r["output"]) < 3000)
print(raw)
print("\nContoh dataset yang BELUM di-mapping:")
print(json.dumps(raw[0], ensure_ascii=False, indent=2))

# %% [markdown]
# Pembagian **train** dan **validation**. 800 steps × batch efektif 8 = 6.400 contoh, sehingga
# 8.000 contoh train sudah cukup untuk satu epoch tanpa pengulangan; 400 contoh dipakai untuk evaluasi.

# %%
subset = raw.shuffle(seed=SEED).select(range(8400))
splits = subset.train_test_split(test_size=400, seed=SEED)
train_raw, val_raw = splits["train"], splits["test"]
print(len(train_raw), "train |", len(val_raw), "validation")

# %%
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

SYSTEM_PROMPT = "Anda adalah asisten AI yang membantu dan menjawab dalam bahasa Indonesia."

tokenizer = get_chat_template(AutoTokenizer.from_pretrained(BASE_MODEL), chat_template="chatml")

def to_chat(batch):
    texts = []
    for instruction, answer in zip(batch["input"], batch["output"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

train_ds = train_raw.map(to_chat, batched=True, remove_columns=train_raw.column_names)
val_ds = val_raw.map(to_chat, batched=True, remove_columns=val_raw.column_names)

print("Contoh dataset yang SUDAH di-mapping (lengkap dengan token spesial):\n")
print(train_ds[0]["text"])

# %% [markdown]
# ## 3. Memuat model dengan QLoRA (4-bit NF4 + double quantization)

# %%
from transformers import BitsAndBytesConfig

def load_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,       # double quantization
        bnb_4bit_compute_dtype=torch.float16,  # T4 belum mendukung bfloat16
    )
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        quantization_config=bnb_config,
    )
    return model, get_chat_template(tok, chat_template="chatml")

model, tokenizer = load_base()
print(model.config.quantization_config)
print(f"Memori GPU terpakai: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

# %% [markdown]
# ## 4. Eksperimen training
# Adapter LoRA dipasang pada **seluruh** komponen Multi-Head Attention dan Feed Forward Network.
# Dua kombinasi hyperparameter dibandingkan, masing-masing 800 steps dengan evaluasi setiap 100 steps:
#
# | Eksperimen | LoRA r | alpha | Learning rate | Scheduler |
# |---|---|---|---|---|
# | A | 16 | 32 | 2e-4 | linear |
# | B | 32 | 32 | 1e-4 | cosine |

# %%
from trl import SFTTrainer, SFTConfig

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",   # Multi-Head Attention
                  "gate_proj", "up_proj", "down_proj"]       # Feed Forward Network

EXPERIMENTS = {
    "A": dict(r=16, lora_alpha=32, learning_rate=2e-4, lr_scheduler_type="linear"),
    "B": dict(r=32, lora_alpha=32, learning_rate=1e-4, lr_scheduler_type="cosine"),
}

def run_experiment(name, base_model, cfg):
    peft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0, bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
    peft_model.print_trainable_parameters()
    args = SFTConfig(
        output_dir=f"outputs/exp_{name}",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        max_steps=800,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=20,
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="steps",   # nama baru dari evaluation_strategy
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    trainer = SFTTrainer(model=peft_model, processing_class=tokenizer,
                         train_dataset=train_ds, eval_dataset=val_ds, args=args)
    result = trainer.train()
    peft_model.save_pretrained(f"adapters/exp_{name}")
    tokenizer.save_pretrained(f"adapters/exp_{name}")
    return result, trainer.state.log_history

# %%
results, histories = {}, {}
name = "A"
results[name], histories[name] = run_experiment(name, model, EXPERIMENTS[name])
print(results[name])

# %% [markdown]
# Model dimuat ulang dari awal untuk eksperimen B supaya adapter A tidak ikut terlatih
# (cache GPU dibersihkan terlebih dahulu).

# %%
del model; gc.collect(); torch.cuda.empty_cache()
model, tokenizer = load_base()
name = "B"
results[name], histories[name] = run_experiment(name, model, EXPERIMENTS[name])
print(results[name])

# %% [markdown]
# ## 5. Perbandingan kurva loss

# %%
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
summary = []
for ax, (name, hist) in zip(axes, histories.items()):
    df = pd.DataFrame(hist)
    tr = df.dropna(subset=["loss"]) if "loss" in df else pd.DataFrame()
    ev = df.dropna(subset=["eval_loss"])
    ax.plot(tr["step"], tr["loss"], label="train loss")
    ax.plot(ev["step"], ev["eval_loss"], marker="o", label="validation loss")
    ax.set_title(f"Eksperimen {name}: r={EXPERIMENTS[name]['r']}, lr={EXPERIMENTS[name]['learning_rate']}")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    summary.append({
        "eksperimen": name, **EXPERIMENTS[name],
        "train_loss_akhir": round(tr["loss"].iloc[-1], 4),
        "val_loss_akhir": round(ev["eval_loss"].iloc[-1], 4),
        "val_loss_min": round(ev["eval_loss"].min(), 4),
        "gap_val_train": round(ev["eval_loss"].iloc[-1] - tr["loss"].iloc[-1], 4),
    })
plt.tight_layout(); plt.show()
summary = pd.DataFrame(summary).set_index("eksperimen")
summary

# %% [markdown]
# Model terbaik dipilih berdasarkan **validation loss akhir terendah**. Selisih validation dengan train
# loss (`gap_val_train`) dipakai untuk memastikan tidak ada *overfitting* (validation loss tidak naik
# sementara train loss terus turun).

# %%
best = summary["val_loss_akhir"].idxmin()
print("Eksperimen terbaik:", best)
print(summary.loc[best])

# %% [markdown]
# ## 6. Uji inferensi singkat

# %%
del model; gc.collect(); torch.cuda.empty_cache()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"adapters/exp_{best}", max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(model)

def chat(question, max_new_tokens=256):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

for q in ["Apa perbedaan antara undang-undang dan peraturan pemerintah?",
          "Berikan tiga tips agar tetap produktif saat bekerja dari rumah."]:
    print(f"### {q}\n{chat(q)}\n")

# %% [markdown]
# ## 7. Push model ke Hugging Face Hub (`merged_16bit`)

# %%
model.push_to_hub_merged(HF_REPO, tokenizer, save_method="merged_16bit", token=os.environ["HF_TOKEN"])
print(f"Model terunggah: https://huggingface.co/{HF_REPO}")

# %%
from huggingface_hub import HfApi
info = HfApi(token=os.environ["HF_TOKEN"]).model_info(HF_REPO)
print("Repo:", info.id, "| private:", info.private)
print([s.rfilename for s in info.siblings])
