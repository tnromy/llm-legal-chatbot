# LLM Legal Chatbot — Fine-tuning, GRPO, and RAG

An end-to-end pipeline for building a legal-domain chatbot in Indonesian, combining LLM fine-tuning (SFT + GRPO reinforcement learning) with a Retrieval-Augmented Generation (RAG) system. The chatbot answers questions about Indonesian labor law by retrieving relevant regulation passages and generating reasoned responses.

> ⚠️ **Not legal advice.** This is an educational project. The model can hallucinate, misquote articles, or reach wrong conclusions. Always verify against official regulations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                          │
│                                                          │
│  User Query                                              │
│      │                                                   │
│      ├─── BM25 Retriever ──┐                            │
│      │                      ├── Ensemble → Reranker ──┐ │
│      └─── FAISS (bge-m3) ──┘     (bge-reranker-v2)   │ │
│                                                        │ │
│                    ┌──────────────────────────────────┐ │ │
│                    │  HyDE (hypothetical doc embed)   │ │ │
│                    └──────────────────────────────────┘ │ │
│                                                          │
│      Context + Query → LLM Generator ──→ Answer         │
│                        (Qwen2.5-1.5B-Legal-ID-GRPO)     │
│                        <think>reasoning</think>          │
└─────────────────────────────────────────────────────────┘
```

## Models

| Model | Description | Link |
|-------|-------------|------|
| **SFT** | QLoRA fine-tuned Qwen2.5-1.5B for Indonesian instruction-following | [tnromy/Qwen2.5-1.5B-Legal-ID-SFT](https://huggingface.co/tnromy/Qwen2.5-1.5B-Legal-ID-SFT) |
| **GRPO** | Reinforcement learning on SFT model for structured reasoning (`<think>`) | [tnromy/Qwen2.5-1.5B-Legal-ID-GRPO](https://huggingface.co/tnromy/Qwen2.5-1.5B-Legal-ID-GRPO) |

## Pipeline Components

### 1. Supervised Fine-Tuning (SFT)

QLoRA (4-bit NF4) fine-tuning of Qwen2.5-1.5B on `Ichsan2895/alpaca-gpt4-indonesian`:

| Experiment | LoRA r | LR | Scheduler | Val Loss |
|-----------|--------|-----|-----------|----------|
| **A (selected)** | 16 | 2e-4 | linear | **1.2456** |
| B | 32 | 1e-4 | cosine | 1.2650 |

### 2. Group Relative Policy Optimization (GRPO)

Trains the SFT model to produce structured reasoning with 4 reward functions:

| Reward | Purpose |
|--------|---------|
| `format_reward` | Proper `<think>...</think>` structure |
| `reasoning_length` | Encourages substantive reasoning |
| `correctness` | ROUGE-L / contains ground truth |
| `language` | Indonesian language compliance |

### 3. RAG System

- **Document processing**: PDF parsing with LangChain, parent-child document chunking
- **Retrieval**: Ensemble of BM25 + FAISS (bge-m3 embeddings), weight 0.4/0.6
- **HyDE**: Hypothetical Document Embeddings for better retrieval
- **Reranking**: bge-reranker-v2-m3 with multi-query scoring
- **Generation**: Best-of-3 quality selection with structured prompts
- **Fallback**: DuckDuckGo web search when retrieval score < threshold

## Legal Documents

The RAG system processes these Indonesian regulations:
- UU 6/2023 (Cipta Kerja)
- PP 35/2021 (Perjanjian Kerja Waktu Tertentu)
- PP 5/2021 (Perizinan Berusaha)
- PP 51/2023 (Pengupahan)

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Notebooks

The notebooks are designed to run on Kaggle (GPU T4 x2):

1. **`Fine-tuning_submission_PGABL_Romi-Fadli.ipynb`** — SFT training
2. **`GRPO_submission_PGABL_Romi-Fadli.ipynb`** — GRPO training
3. **`RAG_submission_PGABL_Romi-Fadli.ipynb`** — RAG pipeline with interactive demo

### Use the Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "tnromy/Qwen2.5-1.5B-Legal-ID-GRPO"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")

messages = [
    {"role": "system", "content": "Anda adalah asisten AI berbahasa Indonesia."},
    {"role": "user", "content": "Apa hak pekerja kontrak menurut UU Cipta Kerja?"}
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + "<think>\n"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=400, temperature=0.3)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Project Structure

```
├── notebooks/
│   ├── Fine-tuning_submission_PGABL_Romi-Fadli.ipynb  # SFT notebook (executed)
│   ├── GRPO_submission_PGABL_Romi-Fadli.ipynb         # GRPO notebook (executed)
│   ├── RAG_submission_PGABL_Romi-Fadli.ipynb          # RAG notebook (executed)
│   ├── finetune.py                                     # SFT source code
│   ├── grpo.py                                         # GRPO source code
│   └── rag.py                                          # RAG source code
├── requirements.txt
├── link_huggingface.txt                                # HuggingFace model links
└── README.md
```

## Tech Stack

- **Unsloth** + **TRL** — efficient QLoRA training and GRPO
- **Transformers** — model loading and inference
- **LangChain** — document processing and retrieval
- **FAISS** — vector similarity search
- **bge-m3 / bge-reranker-v2-m3** — embeddings and reranking
- **Qwen2.5-1.5B** — base language model
- **Kaggle** — GPU compute (T4 x2)

## License

- Model weights: Apache-2.0 (following Qwen2.5-1.5B)
- Training data: [Ichsan2895/alpaca-gpt4-indonesian](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian) — CC BY-SA 4.0
- Code: MIT
