#!/usr/bin/env bash
set -e

# ======================
# Config
# ======================
ENV_NAME="sft-qwen3"
PYTHON_BIN=python3

echo "🐍 Using python: $($PYTHON_BIN --version)"

# ======================
# Create venv
# ======================
if [ ! -d "$ENV_NAME" ]; then
  echo "📦 Creating virtual environment: $ENV_NAME"
  $PYTHON_BIN -m venv $ENV_NAME
else
  echo "📦 Virtual environment already exists: $ENV_NAME"
fi

# ======================
# Activate venv
# ======================
echo "🔌 Activating environment..."
source $ENV_NAME/bin/activate

echo "🐍 Python in use:"
which python
python --version

# ======================
# Upgrade pip
# ======================
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ======================
# Install dependencies
# ======================
echo "📚 Installing required packages..."

pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  safetensors \
  huggingface_hub \
  tensorboard \
  tqdm

# TRL (fixed version)
pip install trl==0.23.1

# ⚠️ DeepSpeed（如集群未预装才打开）
# pip install deepspeed

# ======================
# Verify installation
# ======================
echo ""
echo "✅ Verifying installation..."
python - << 'EOF'
import torch
import transformers
import datasets
import trl
import huggingface_hub

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("trl:", trl.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

echo ""
echo "🎯 Environment ready!"
echo "👉 To activate later:"
echo "   source $ENV_NAME/bin/activate"
