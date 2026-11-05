# %% [markdown]
# # Chatbot Tim Legal berbasis RAG (Kriteria 2)
#
# **Proyek Akhir — Pengembangan Generative AI berbasis LLM (Dicoding)** · Romi Fadli
#
# Notebook ini menyuntikkan pengetahuan hukum ke model hasil fine-tuning kita melalui
# **Retrieval-Augmented Generation**. Knowledge base: empat dokumen resmi yang disediakan
# (UU 6/2023, PP 35/2021, PP 5/2021, PP 51/2023).
#
# | Tahap | Implementasi |
# |---|---|
# | Ingestion | `PyPDFLoader` → metadata enrichment → `RecursiveCharacterTextSplitter` (chunk 1000, overlap 150) |
# | Embedding & vector DB | `BAAI/bge-m3` (open-source) → **FAISS** lokal |
# | Retrieval | Ensemble **BM25 + FAISS** (bobot 0.4 / 0.6, k=6), **Parent-Child** retriever, **metadata filtering** |
# | Query transformation | **HyDE** dengan 2 jawaban hipotetis dari LLM |
# | Post-retrieval | **Reranker** `BAAI/bge-reranker-v2-m3` (Top-3) + fallback **DuckDuckGo** bila skor Top-1 < threshold |
# | Generation | Model GRPO kita `tnromy/Qwen2.5-1.5B-Legal-ID-GRPO` (menampilkan `<think>`) + **sitasi** |
# | Interface | Interactive loop `input()` + `IPython.display.Markdown` |

# %% [markdown]
# ## 1. Instalasi & konfigurasi

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "langchain-classic", "langchain-community", "langchain-text-splitters",
                "langchain-huggingface", "faiss-cpu", "rank_bm25", "pypdf", "ddgs",
                "sentence-transformers"], check=True)

# %%
import os, re, glob, textwrap
import torch
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")  # disimpan sebagai env var

# Lokasi mount dataset Kaggle bisa berbeda-beda, jadi folder PDF dicari otomatis.
_found = glob.glob("/kaggle/input/**/PP Nomor 35 Tahun 2021.pdf", recursive=True)
assert _found, "Dataset PDF belum terpasang di /kaggle/input"
PDF_DIR = os.path.dirname(_found[0])
LLM_REPO = "tnromy/Qwen2.5-1.5B-Legal-ID-GRPO"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True))

# %% [markdown]
# ## 2. Memuat dokumen PDF + metadata enrichment
# Setiap halaman diberi metadata **hardcode** (jenis, nomor, tahun, judul dokumen) dan metadata hasil
# **ekstraksi otomatis** (nomor pasal pertama yang muncul di halaman, BAB, dan topik).

# %%
from langchain_community.document_loaders import PyPDFLoader

DOC_INFO = {
    "UU Nomor 6 Tahun 2023.pdf": dict(dokumen="UU 6/2023", jenis="Undang-Undang", nomor=6, tahun=2023,
        judul="Penetapan Perpu 2/2022 tentang Cipta Kerja menjadi Undang-Undang"),
    "PP Nomor 35 Tahun 2021.pdf": dict(dokumen="PP 35/2021", jenis="Peraturan Pemerintah", nomor=35, tahun=2021,
        judul="PKWT, Alih Daya, Waktu Kerja dan Waktu Istirahat, serta Pemutusan Hubungan Kerja"),
    "PP Nomor 5 Tahun 2021.pdf": dict(dokumen="PP 5/2021", jenis="Peraturan Pemerintah", nomor=5, tahun=2021,
        judul="Penyelenggaraan Perizinan Berusaha Berbasis Risiko"),
    "PP Nomor 51 Tahun 2023.pdf": dict(dokumen="PP 51/2023", jenis="Peraturan Pemerintah", nomor=51, tahun=2023,
        judul="Perubahan atas PP 36/2021 tentang Pengupahan"),
}
TOPIK = {
    "upah|lembur|pengupahan": "pengupahan",
    "pemutusan hubungan kerja|pesangon|PHK": "PHK",
    "perjanjian kerja waktu tertentu|PKWT|alih daya": "hubungan kerja",
    "waktu kerja|waktu istirahat|cuti": "waktu kerja",
    "perizinan|izin usaha|KBLI|risiko": "perizinan berusaha",
}

def enrich(doc):
    text = doc.page_content
    pasal = re.search(r"\bPasal\s+(\d+[A-Z]?)\b", text)
    bab = re.search(r"\bBAB\s+([IVXLC]+)\b", text)
    topik = [t for pat, t in TOPIK.items() if re.search(pat, text, re.I)]
    doc.metadata.update(
        pasal=f"Pasal {pasal.group(1)}" if pasal else "-",
        bab=f"BAB {bab.group(1)}" if bab else "-",
        topik=", ".join(topik) or "umum",
        halaman=doc.metadata.get("page", 0) + 1,
    )
    return doc

pages = []
for path in sorted(glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)):
    info = DOC_INFO[os.path.basename(path)]
    for d in PyPDFLoader(path).load():
        d.page_content = re.sub(r"[ \t]+", " ", d.page_content).strip()
        if len(d.page_content) < 50:
            continue
        d.metadata.update(info)
        pages.append(enrich(d))
print(f"{len(pages)} halaman dimuat dari {len(DOC_INFO)} dokumen")
assert {p.metadata["dokumen"] for p in pages} == {v["dokumen"] for v in DOC_INFO.values()}, "Ada PDF yang tidak termuat"
print(pages[100].metadata)

# %% [markdown]
# ## 3. Chunking (ukuran & overlap eksplisit)

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                          separators=["\nPasal ", "\n\n", "\n", ". ", " "])
chunks = [enrich(c) for c in splitter.split_documents(pages)]
print(f"{len(chunks)} chunk (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
from collections import Counter
print(Counter(c.metadata["dokumen"] for c in chunks))

# %% [markdown]
# ## 4. Embedding open-source + FAISS

# %%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": DEVICE},
                                   encode_kwargs={"normalize_embeddings": True, "batch_size": 32})
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_legal")
print("Jumlah vektor:", vectorstore.index.ntotal)

# %% [markdown]
# ## 5. Uji retrieval

# %%
def show(docs, n=200):
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(f"[{i}] {m['dokumen']} hlm. {m['halaman']} {m['pasal']} | topik: {m['topik']}")
        print("    " + textwrap.shorten(d.page_content, n))

semantic = vectorstore.as_retriever(search_kwargs={"k": 4})
show(semantic.invoke("Berapa lama waktu kerja lembur maksimal dalam satu minggu?"))

# %% [markdown]
# ### Metadata filtering
# Pencarian dibatasi hanya pada dokumen tertentu menggunakan `filter` metadata.

# %%
pp35_only = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"dokumen": "PP 35/2021"}})
show(pp35_only.invoke("perhitungan upah lembur"))
pengupahan = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"dokumen": "PP 51/2023"}})
show(pengupahan.invoke("formula penghitungan upah minimum"))

# %% [markdown]
# ## 6. Ensemble Retriever (BM25 + FAISS)
# BM25 menangkap kata kunci eksak (nomor pasal, istilah teknis), FAISS menangkap kemiripan makna.
# Skor digabungkan dengan *Reciprocal Rank Fusion* dengan bobot 0.4 (BM25) dan 0.6 (semantik).

# %%
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
    from langchain_classic.storage import InMemoryStore
except ImportError:
    from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
    from langchain.storage import InMemoryStore

K = 6
bm25 = BM25Retriever.from_documents(chunks); bm25.k = K
dense = vectorstore.as_retriever(search_kwargs={"k": K})
ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
show(ensemble.invoke("Pasal 81 angka 23 jam kerja lembur"))

# %% [markdown]
# ## 7. Parent-Child Retriever
# *Child chunk* (400 karakter) dipakai untuk pencarian vektor yang presisi, sedangkan *parent chunk*
# (2000 karakter) dikirim ke LLM agar konteksnya utuh.

# %%
from langchain_core.documents import Document

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
child_store = FAISS.from_documents([Document(page_content="init", metadata={})], embeddings)
parent_retriever = ParentDocumentRetriever(vectorstore=child_store, docstore=InMemoryStore(),
                                           child_splitter=child_splitter, parent_splitter=parent_splitter,
                                           search_kwargs={"k": 8})
parent_retriever.add_documents(pages)
parents = parent_retriever.invoke("hak pekerja atas uang pesangon ketika terjadi PHK")
print(f"{len(parents)} parent chunk, panjang rata-rata {sum(len(p.page_content) for p in parents)//max(len(parents),1)} karakter")
show(parents[:3], 300)

# %% [markdown]
# ## 8. Memuat model hasil fine-tuning (SFT → GRPO)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(LLM_REPO, token=os.environ["HF_TOKEN"])
llm = AutoModelForCausalLM.from_pretrained(LLM_REPO, torch_dtype=torch.float16, device_map=DEVICE,
                                           token=os.environ["HF_TOKEN"])
llm.eval()

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSecondThink(StoppingCriteria):
    """Hentikan generasi bila model mulai membuka blok <think> kedua setelah </think>."""
    def __init__(self, start): self.start = start
    def __call__(self, input_ids, scores, **kw):
        t = tokenizer.decode(input_ids[0][self.start:], skip_special_tokens=True)
        return "</think>" in t and "<think>" in t.split("</think>", 1)[1]

def generate(messages, max_new_tokens=512, temperature=0.3, think=False):
    """think=True: giliran asisten diawali '<think>\\n' (reasoning trigger ala DeepSeek-R1)."""
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prefix = "<think>\n" if think else ""
    ids = tokenizer(text + prefix, return_tensors="pt").input_ids.to(llm.device)
    with torch.no_grad():
        out = llm.generate(ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                           temperature=temperature, top_p=0.9, repetition_penalty=1.1,
                           pad_token_id=tokenizer.eos_token_id,
                           stopping_criteria=StoppingCriteriaList([StopOnSecondThink(ids.shape[1])]))
    resp = prefix + tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    if "</think>" in resp:
        head, tail = resp.split("</think>", 1)
        resp = head + "</think>" + tail.split("<think>", 1)[0]
    return resp.strip()

# %% [markdown]
# ## 9. HyDE (Hypothetical Document Embeddings)
# LLM membuat **dua** jawaban hipotetis (boleh berhalusinasi). Jawaban tersebut dipakai sebagai query
# pencarian, sehingga yang dicari adalah "dokumen yang mirip jawaban", bukan "mirip pertanyaan".

# %%
def hyde(question, n=2):
    messages = [{"role": "system", "content": "Tuliskan satu paragraf singkat bergaya pasal peraturan "
                 "perundang-undangan Indonesia yang menjawab pertanyaan berikut. Jangan gunakan tag <think>."},
                {"role": "user", "content": question}]
    hypos = []
    for i in range(n):
        text = generate(messages, max_new_tokens=160, temperature=0.8)
        hypos.append(re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip())
    return hypos

q_demo = "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"
hypothetical = hyde(q_demo)
for i, h in enumerate(hypothetical, 1):
    print(f"Hipotesis {i}: {textwrap.shorten(h, 300)}\n")

# %% [markdown]
# ## 10. Reranker (Cross-Encoder) + fallback DuckDuckGo

# %%
from sentence_transformers import CrossEncoder
from ddgs import DDGS

reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)
TOP_K, THRESHOLD = 5, 0.10

def retrieve_candidates(question):
    queries = [question] + hyde(question)
    pool = {}
    for q in queries:
        for d in ensemble.invoke(q) + parent_retriever.invoke(q)[:3]:
            pool.setdefault(d.page_content[:200], d)
    return list(pool.values()), queries

def rerank(question, docs, queries=None):
    """Skor tiap dokumen = skor tertinggi terhadap pertanyaan asli maupun hipotesis HyDE.
    Pertanyaan bergaya percakapan ("beresin laporan") sering kalah skor dengan bahasa pasal,
    sehingga hipotesis HyDE membantu menjembatani gaya bahasa tersebut."""
    queries = queries or [question]
    pairs = [(q, d.page_content) for q in queries for d in docs]
    raw = reranker.predict(pairs).reshape(len(queries), len(docs))
    scores = raw.max(axis=0)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [(d, float(s)) for d, s in ranked[:TOP_K]]

def web_search(question, n=4):
    results = DDGS().text(f"{question} peraturan ketenagakerjaan Indonesia", region="id-id", max_results=n)
    return [Document(page_content=f"{r['title']}. {r['body']}",
                     metadata={"dokumen": "Web", "halaman": "-", "pasal": "-", "topik": "-", "url": r["href"]})
            for r in results]

cands, qs = retrieve_candidates(q_demo)
top = rerank(q_demo, cands, qs)
for d, s in top:
    print(f"score={s:.3f} | {d.metadata['dokumen']} hlm. {d.metadata['halaman']} {d.metadata['pasal']}")
print("\nUji fallback internet (pertanyaan di luar dokumen):")
q_off = "Siapa pemenang Piala Dunia FIFA 2022?"
off = rerank(q_off, *retrieve_candidates(q_off))
print(f"Top-1 score={off[0][1]:.3f} -> {'DuckDuckGo' if off[0][1] < THRESHOLD else 'lokal'}")

# %% [markdown]
# ## 11. Pipeline RAG lengkap
# Prompt berisi `{context}` dan `{question}`. Model diminta berpikir di dalam `<think>...</think>`
# sebelum memberikan jawaban final, lalu sistem menambahkan **sitasi** sumber. Giliran asisten
# diawali `<think>\n` (reasoning trigger) dan generasi berhenti bila model membuka `<think>` kedua.

# %%
SYSTEM_PROMPT = ("Anda adalah asisten hukum internal Tim Legal. Jawab HANYA berdasarkan konteks. "
                 "Tuliskan proses berpikir di dalam <think> ... </think>, lalu jawaban akhir yang "
                 "ringkas dalam bahasa Indonesia dan sebutkan dasar hukum atau sumbernya.")
PROMPT_TEMPLATE = "Konteks:\n{context}\n\nPertanyaan: {question}\n\n{guide}"
GUIDE_LEGAL = ("Di dalam <think>, uraikan: (1) siapa penanya dan statusnya sebagai pekerja, "
               "(2) pasal mana dari konteks yang relevan dan apa isinya, (3) hak dan kewajiban yang "
               "timbul dari pasal tersebut bagi penanya, (4) kesimpulannya. Setelah </think>, berikan "
               "jawaban akhir beserta dasar hukumnya.")
GUIDE_WEB = ("Konteks berasal dari internet, bukan dari dokumen hukum internal. Di dalam <think>, uraikan "
             "informasi dari konteks yang menjawab pertanyaan, lalu simpulkan. Setelah </think>, berikan "
             "jawaban akhir yang ringkas beserta sumbernya.")

def cite(d):
    m = d.metadata
    return f"{m['dokumen']}, hlm. {m['halaman']}" + (f", {m['pasal']}" if m.get("pasal", "-") != "-" else "") \
        + (f" ({m['url']})" if m.get("url") else "")

def quality(text):
    """Skor kandidat: format benar + panjang penalaran (selaras reward function GRPO)."""
    m = re.match(r"\s*<think>(.*?)</think>\s*(\S.*)", text, re.S)
    if not m:
        return 0.0
    thought = m.group(1).strip()
    length = 1.0 if len(thought) >= 200 else 0.5 if len(thought) >= 50 else 0.2 if thought else 0.0
    return 1.0 + length + min(len(m.group(2)), 600) / 600

def answer(question):
    cands, queries = retrieve_candidates(question)
    top = rerank(question, cands, queries)
    source = "dokumen internal"
    if top[0][1] < THRESHOLD:
        source = "internet (DuckDuckGo)"
        docs = web_search(question)
    else:
        docs = [d for d, _ in top]
    context = "\n\n".join(f"[{i}] ({cite(d)})\n{d.page_content[:1200]}" for i, d in enumerate(docs, 1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    context=context, question=question,
                    guide=GUIDE_WEB if source.startswith("internet") else GUIDE_LEGAL)}]
    # Best-of-3: pilih kandidat dengan penalaran paling lengkap (memakai aturan reward GRPO)
    outputs = [generate(messages, think=True, temperature=0.5) for _ in range(3)]
    output = max(outputs, key=quality)
    return dict(output=output, docs=docs, source=source, top1=top[0][1])

from IPython.display import Markdown, display

def render(question, res):
    think = re.search(r"<think>(.*?)(</think>|$)", res["output"], re.S)
    final = re.sub(r"<think>.*?(</think>|$)", "", res["output"], flags=re.S).strip()
    md = f"### ❓ {question}\n\n"
    if think:
        quoted = "\n".join("> " + line for line in think.group(1).strip().splitlines())
        md += f"**🧠 Proses berpikir (`<think>`):**\n\n{quoted}\n\n"
    final = re.sub(r"^\s*Jawaban Akhir\s*:\s*", "", final)
    md += f"**✅ Jawaban:**\n\n{final or res['output']}\n\n"
    md += f"**📚 Sumber ({res['source']}, skor reranker Top-1 = {res['top1']:.3f}):**\n\n"
    md += "\n".join(f"- {cite(d)}" for d in res["docs"])
    display(Markdown(md))

res = answer(q_demo)
print("Output mentah model:\n", res["output"], "\n")
render(q_demo, res)

# %% [markdown]
# ## 12. Interface: Interactive Python Loop
# Ketik pertanyaan lalu tekan Enter; ketik `exit` untuk berhenti. Saat notebook dijalankan tanpa
# input interaktif (misalnya *Save & Run All* di Kaggle), loop memakai daftar pertanyaan contoh.

# %%
DEMO_QUESTIONS = [
    "Berapa jam maksimal waktu kerja lembur dalam satu hari dan satu minggu?",
    "Apa saja hak pekerja yang terkena PHK karena perusahaan tutup?",
    "Siapa pemenang Piala Dunia FIFA 2022?",
    "exit",
]

def ask(prompt):
    try:
        return input(prompt)
    except Exception:  # stdin tidak tersedia pada mode batch
        q = DEMO_QUESTIONS.pop(0)
        print(prompt + q)
        return q

while True:
    question = ask("Pertanyaan Tim Legal (ketik 'exit' untuk keluar): ").strip()
    if question.lower() in {"exit", "quit", "keluar"}:
        print("Sesi selesai.")
        break
    if not question:
        continue
    render(question, answer(question))
