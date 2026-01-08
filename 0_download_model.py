# -*- coding: utf-8 -*-
"""
Download Qwen/Qwen3-8B model locally (weights + tokenizer + config).

Recommended for:
- SFT / CPT
- Offline training
- Multi-node / shared FS (e.g. /shared_workspace_mfs)
"""

import os
from huggingface_hub import snapshot_download

# ======================
# Config
# ======================
MODEL_ID = "Qwen/Qwen3-8B"
LOCAL_DIR = "/shared_workspace_mfs/ruochen/models/Qwen3-8B"

# 如果你在国内 / 私有镜像，可自行改 HF_ENDPOINT
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"📥 Downloading model: {MODEL_ID}")
    print(f"📂 Saving to: {LOCAL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,  # 强烈建议 False（共享文件系统更稳）
        resume_download=True,
    )

    print("\n🎯 Download complete!")
    print(f"Model saved at: {LOCAL_DIR}")

    print("\n📁 Directory preview:")
    for name in sorted(os.listdir(LOCAL_DIR)):
        print("  -", name)


if __name__ == "__main__":
    main()
