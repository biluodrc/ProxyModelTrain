# -*- coding: utf-8 -*-
"""
Download and save allenai/Dolci-Instruct-SFT locally.

- Uses Hugging Face datasets
- Saves each split as JSONL
- Suitable for SFT training (messages format)
"""

import os
import json
from datasets import load_dataset

# ======================
# Config
# ======================
DATASET_NAME = "allenai/Dolci-Instruct-SFT"
OUTPUT_DIR = "/shared_workspace_mfs/ruochen/datasets"

SAVE_FORMAT = "jsonl"   # jsonl recommended
MAX_SAMPLES_PER_SPLIT = None  # e.g. 10000 for debug, None = all


def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📥 Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    for split, dset in ds.items():
        print(f"🔹 Split: {split}, samples: {len(dset):,}")

        if MAX_SAMPLES_PER_SPLIT is not None:
            dset = dset.select(range(min(MAX_SAMPLES_PER_SPLIT, len(dset))))
            print(f"   ↳ Truncated to {len(dset):,} samples")

        out_path = os.path.join(OUTPUT_DIR, f"{split}.{SAVE_FORMAT}")

        if SAVE_FORMAT == "jsonl":
            save_jsonl(dset, out_path)
        else:
            raise ValueError("Only jsonl supported in this script.")

        print(f"✅ Saved to {out_path}")

    print("\n🎯 Done! Dataset saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
