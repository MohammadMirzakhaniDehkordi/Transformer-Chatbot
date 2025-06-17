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
```

## 📁 Project Structure

```
chatbot_project/
├── chatbot_transformer.py      # Core model: Transformer + training + reply()
├── chatbot_api.py              # FastAPI-based REST endpoint (for /chat)
├── app_gradio.py               # Gradio web interface for Hugging Face
├── requirements_api.txt        # Dependencies for API version
├── requirements_gradio.txt     # Dependencies for UI version
├── Procfile                    # Required for Railway deployment
├── .gitignore
└── README.md
```

## 🚀 How to Run Locally

1. Clone the repo

```bash
git clone https://github.com/your_username/chatbot_project.git
cd chatbot_project
```

2. Install dependencies

Run the Project in a Virtual Environment (macOS):
```bash
python3 -m venv venv
```
Activate the environment
```
source venv/bin/activate
```

For API:

```bash
pip install -r requirements_api.txt
```
For Gradio UI:

```bash
pip install -r requirements_gradio.txt
```

✅ For M1/M2 Mac (Apple Silicon):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you're using an Intel Mac:

```bash
pip install torch torchvision torchaudio
```

🧪 Test Installation

```bash
python -c "import torch; print(torch.__version__)"
```


3. 🚀 Run the chatbot:

- 🧠  Train the model (optional - already trains when imported):

```bash
python chatbot_transformer.py
```

- 🚀  Run FastAPI server:

```bash 
uvicorn chatbot_api:app --reload
```
 
 - - Then POST to: http://127.0.0.1:8000/chat
 

- 🌐 Run Gradio UI:
This will open a chat interface in your browser.
``` bash 
python app_gradio.py
```

## 🐳 Docker Support
### ✅ Build and Run with Docker (API Mode)
Build Docker image:
```bash
docker build -t transformer-chatbot .
```
Run the container
```bash
docker run -p 8000:8000 transformer-chatbot
```
Then open: http://localhost:8000/docs

## ☁️  Deployment

### 1. 🌐  Deploy API to Railway


- Create a new Railway project and connect this repo.
- In Railway:
- - Set the build command: ``` pip install -r requirements_api.txt ```
- - Set the start command: ``` uvicorn chatbot_api:app --host=0.0.0.0 --port=8000 ```

### 2. 🌐 2. Deploy UI to Hugging Face Spaces
- Create a new space → Choose “Gradio” + connect this GitHub repo.
- Set app_gradio.py as main file.
- Ensure requirements_gradio.txt is present.
- The bot will be live at: https://huggingface.co/spaces/your_username/chatbot_project

## 📝 Example API Usage
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"hello"}'
```

Return:
```bash
{"response": "hi . how are you ?"}
```

## 📌 Limitations

- Only trained on 5,000 dialogue pairs.
- Very basic tokenizer (no subwords, no attention masks).
- Not stateful — replies are single-turn.
- No persistence or database.

## ✨ TODO (Contributions welcome!)

 - Add better tokenizer (e.g. BPE, spaCy).
 - Multi-turn chat memory.
 - Support for Persian (فارسی) data.
 - Pretrained Transformer (e.g. BART, T5).
 - Beam search decoding.


 ## 📜 License

MIT License — for learning and educational use.