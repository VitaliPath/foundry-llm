# Foundry-LLM

*A principled, iterative, and object-oriented journey to build a large language model from scratch.*

[![CI](https://github.com/VitaliPath/foundry-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/VitaliPath/foundry-llm/actions)

---

## ✨ Project Overview

**Foundry-LLM** is an open, architecture-first implementation of a modern Transformer-based language model.
It emphasizes:

- Clean object-oriented structure
- Progressive architectural upgrades (tokenization, attention, normalization, activations)
- Budget-aware development (Kaggle, Colab Pro, spot GPUs)
- Clear, well-documented code and design reasoning

Core design is documented in [`docs/architecture.md`](docs/architecture.md).

---

## 📁 Project Layout

```
docs/        # High-level design docs
research/    # Deep dives worth sharing
notes/       # Working drafts and quick scratchpads
```

Model code will live in `foundry_llm/` and training scripts in `scripts/` as development progresses.

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/foundry-llm.git
cd foundry-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

---

## 🧭 Roadmap

1. ✅ Project structure, CI, and linting
2. ⏳ Character-level GPT MVP
3. ⏳ Subword (BPE) tokenizer
4. ⏳ Transformer upgrades (RMSNorm, SwiGLU, GQA)
5. ⏳ Gutenberg pretraining (target perplexity < 50)
6. ⏳ Controlled sampling and Streamlit demo
7. ⏳ Research extensions (GRFT, agent wrapper, LoRA fine-tuning)

---

## 🤝 Contributing

Contributions are welcome! Please:

- Run `pre-commit` locally before committing
- Follow [Conventional Commits](https://www.conventionalcommits.org/)
- Include or update tests and docs for any public-facing changes

---

## 📜 License

MIT License — see [`LICENSE`](LICENSE) for full terms.
