#!/bin/bash

echo "🔧 Setting up your AI Transformer Chat Bot environment on macOS..."

# Step 1: Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Step 4: Detect CPU architecture
ARCH=$(uname -m)

echo ""
echo "🧠 Detected architecture: $ARCH"

if [[ "$ARCH" == "arm64" ]]; then
    echo "🍏 Apple Silicon (M1/M2/M3) detected. Installing PyTorch with CPU-only wheel..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "💻 Intel Mac detected. Installing standard PyTorch..."
    pip install torch torchvision torchaudio
fi

# Step 5: Ask user which mode to install
echo ""
echo "🛠️  Choose a mode:"
echo "1 - FastAPI (REST API)"
echo "2 - Gradio (Web UI)"
read -p "Enter 1 or 2: " mode

if [ "$mode" = "1" ]; then
    echo "📡 Installing FastAPI requirements..."
    pip install -r requirements_api.txt --no-deps
elif [ "$mode" = "2" ]; then
    echo "🌐 Installing Gradio requirements..."
    pip install -r requirements_gradio.txt --no-deps
else
    echo "❌ Invalid input. Exiting setup."
    exit 1
fi

# Step 6: Ask if user wants to run the chatbot
echo ""
read -p "🚀 Do you want to run the chatbot now? (y/n): " run_now
if [ "$run_now" = "y" ]; then
    if [ "$mode" = "1" ]; then
        echo "🚀 Launching FastAPI server..."
        uvicorn chatbot_api:app --reload
    elif [ "$mode" = "2" ]; then
        echo "🚀 Launching Gradio UI..."
        python app_gradio.py
    fi
else
    echo "✅ Setup complete."
    echo "To activate your environment later:"
    echo "source venv/bin/activate"
fi
