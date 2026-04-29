# 🧠 GPT from Scratch

A character-level GPT language model built entirely from scratch in PyTorch, trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt) dataset. This project follows **Andrej Karpathy's** legendary YouTube lecture — [**"Let's build GPT: from scratch, in code, spelled out"**](https://www.youtube.com/watch?v=kCc8FmEb1nY) — and implements every component of a decoder-only Transformer, step by step.

---

## 📚 Acknowledgements

This project is heavily inspired by and built while following along with:

- 🎬 **[Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — the primary reference for the entire implementation.
- 📄 **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) — the original Transformer paper.
- 🔗 **[nanoGPT](https://github.com/karpathy/nanoGPT)** — Karpathy's minimal GPT training repository.

All credit for the teaching methodology and the incremental build-up approach goes to Andrej Karpathy. This repo is my personal learning implementation.

---

## 🏗️ Project Structure

```
GPT from Scratch/
├── bigram.py          # V1 — Simple bigram language model (baseline)
├── v2.py              # V2 — Full GPT with multi-head self-attention & transformer blocks
├── gpt_dev.ipynb      # Development notebook (step-by-step exploration & experimentation)
├── demo.py            # Quick script to verify CUDA / GPU availability
├── input.txt          # Training data — Tiny Shakespeare (~1.1 MB)
├── requirements.txt   # Python dependencies (PyTorch + CUDA 12.1)
└── .venv/             # Virtual environment (managed with uv)
```

---

## 🔬 What's Implemented

### V1 — Bigram Language Model (`bigram.py`)

The simplest possible language model used as a starting baseline:

- Character-level tokenization (encode / decode)
- Single embedding lookup table
- Cross-entropy loss
- Autoregressive generation via multinomial sampling

| Hyperparameter | Value |
|---|---|
| Batch Size | 32 |
| Block Size (Context Length) | 8 |
| Max Iterations | 3,000 |
| Learning Rate | 1e-2 |

### V2 — Full GPT Transformer (`v2.py`)

The complete decoder-only Transformer architecture:

- **Token + Positional Embeddings**
- **Multi-Head Self-Attention** — scaled dot-product attention with causal masking
- **Feed-Forward Network** — two-layer MLP with ReLU activation (4× expansion)
- **Transformer Blocks** — with residual connections and pre-norm LayerNorm
- **Dropout** — applied throughout for regularization
- **Autoregressive Generation** — context-windowed sampling

| Hyperparameter | Value |
|---|---|
| Batch Size | 64 |
| Block Size (Context Length) | 256 |
| Embedding Dimension (`n_embd`) | 384 |
| Number of Heads (`n_head`) | 6 |
| Number of Layers (`n_layer`) | 6 |
| Dropout | 0.2 |
| Max Iterations | 5,000 |
| Learning Rate | 3e-4 |
| **Total Parameters** | **~10.8M** |

---

## 🛠️ Setup & Installation

This project uses [**uv**](https://docs.astral.sh/uv/) as the Python package manager for fast, reliable dependency management.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed (`pip install uv` or see [installation docs](https://docs.astral.sh/uv/getting-started/installation/))
- NVIDIA GPU with CUDA support (recommended; CPU fallback is available)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/gpt-from-scratch.git
cd gpt-from-scratch
```

### 2. Create the virtual environment with uv

```bash
uv venv
```

### 3. Activate the virtual environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### 4. Install dependencies with uv

```bash
uv pip install -r requirements.txt
```

This installs PyTorch with **CUDA 12.1** support along with torchvision, torchaudio, and altair.

### 5. Verify GPU availability

```bash
python demo.py
```

You should see output like:

```
True
12.1
NVIDIA GeForce RTX XXXX
```

---

## 🚀 Usage

### Download the training data

The Tiny Shakespeare dataset should already be included as `input.txt`. If not:

```bash
# Using wget
wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt

# Or using curl
curl -O https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt
```

### Train the bigram model (V1 — baseline)

```bash
python bigram.py
```

### Train the full GPT model (V2)

```bash
python v2.py
```

> ⚠️ Training the V2 model requires a CUDA-capable GPU for reasonable training times. On CPU, expect it to be significantly slower.

### Explore in the notebook

```bash
jupyter notebook gpt_dev.ipynb
```

The development notebook walks through the implementation incrementally — from raw text processing to the full Transformer — making it great for learning and experimentation.

---

## 📊 Training Data

The model is trained on the **Tiny Shakespeare** dataset (~1.1 MB, ~1M characters), which is a concatenation of all of Shakespeare's works. The character-level vocabulary consists of 65 unique characters.

- **Train split**: 90% of the data
- **Validation split**: 10% of the data

---

## 🧩 Key Concepts Covered

Following along with Karpathy's lecture, this project covers:

1. **Character-level tokenization** — mapping characters to integers and back
2. **Bigram language models** — the simplest autoregressive model
3. **Self-attention mechanism** — the core of Transformers
4. **Scaled dot-product attention** — with `C^(-0.5)` scaling
5. **Causal masking** — lower-triangular mask to prevent attending to future tokens
6. **Multi-head attention** — running multiple attention heads in parallel
7. **Feed-forward networks** — position-wise MLPs within Transformer blocks
8. **Residual connections** — enabling deep network training
9. **Layer normalization** — pre-norm formulation for training stability
10. **Positional embeddings** — encoding sequence position information
11. **Dropout regularization** — preventing overfitting
12. **AdamW optimizer** — weight-decay-decoupled Adam

---

## 📝 Requirements

```
torch
torchvision
torchaudio
altair
```

All PyTorch packages are installed from the **CUDA 12.1** wheel index (`https://download.pytorch.org/whl/cu121`).

---

## 📜 License

This project is for educational purposes. Feel free to use, modify, and learn from it.

---

## 🙏 Special Thanks

A huge thank you to **[Andrej Karpathy](https://karpathy.ai/)** for creating one of the best educational resources on Transformers and GPT. His ability to break down complex concepts into intuitive, code-first explanations is truly unmatched. If you haven't watched the video yet, I highly recommend it — it's a masterclass in deep learning education.

> *"The spelled out intro to language modeling: building makeGPT"* — Andrej Karpathy
