# 🤖 Transformer Chatbot (Cornell Movie Dialogues)

A simple conversational AI (chatbot) trained using a Transformer model on the Cornell Movie-Dialogs Corpus. This project includes:

- A basic Transformer encoder-decoder chatbot implemented in PyTorch.
- A REST API using FastAPI for external communication.
- A Gradio-based UI for interactive web deployment.
- Ready-to-deploy structure for Railway and Hugging Face Spaces.

---

## 📦 Features

- Sequence-to-sequence chatbot with transformer architecture.
- Trained on real conversational data (movie dialogues).
- Simple and lightweight (for learning/demo purposes).
- Two interface modes: REST API and Web UI.
- Deployable to free hosting platforms: [Railway](https://railway.app), [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🧠 Dataset

Cornell Movie Dialogues (via 🤗 Datasets):

```python
from datasets import load_dataset
dataset = load_dataset("cornell_movie_dialog")
