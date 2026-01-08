# -*- coding: utf-8 -*-
"""
Train Qwen3-8B on Dolci-Instruct-SFT (messages) using TRL SFTTrainer (v0.23.1)
- Uses locally downloaded model + locally saved dataset (jsonl)
- DeepSpeed ZeRO-2
"""

import os
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ======================
# Global config (edit here)
# ======================
# Local model path (downloaded via snapshot_download)
MODEL_PATH = "/shared_workspace_mfs/ruochen/models/Qwen3-8B"

# Local dataset path (downloaded/saved via datasets -> jsonl)
TRAIN_JSONL = "/shared_workspace_mfs/ruochen/sft_dataset/Dolci-Instruct-SFT/train.jsonl"

OUTPUT_DIR = "./sft_output"
DS_CONFIG = "ds_config_zero2.json"

SEED = 42
SMALL_TEST = False     # True: only run <=32 samples
MAX_LENGTH = 2048


def setup_env():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("❌ No CUDA devices detected.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    print(f"🧠 Auto-detected {num_gpus} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(random.randint(29500, 29999))
        print(f"🔄 Assigned MASTER_PORT={os.environ['MASTER_PORT']}")

    # Print DS ZeRO stage (so you KNOW it's Zero-2)
    if os.path.exists(DS_CONFIG):
        with open(DS_CONFIG, "r", encoding="utf-8") as f:
            ds = json.load(f)
        stage = ds.get("zero_optimization", {}).get("stage", None)
        print(f"🧾 DeepSpeed ZeRO stage = {stage}")


def main():
    setup_env()
    random.seed(SEED)

    # ======================
    # Load dataset (jsonl)
    # ======================
    if not os.path.exists(TRAIN_JSONL):
        raise FileNotFoundError(f"❌ TRAIN_JSONL not found: {TRAIN_JSONL}")

    print(f"📂 Loading dataset from: {TRAIN_JSONL}")
    dataset = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    print(f"Loaded raw samples: {len(dataset):,}")
    print("Columns:", dataset.column_names)

    if SMALL_TEST:
        dataset = dataset.select(range(min(32, len(dataset))))
        print("⚠️ Using small test subset (<=32).")

    # ======================
    # Tokenizer (local)
    # ======================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ MODEL_PATH not found: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ======================
    # Filter & convert messages -> text
    # ======================
    def ends_with_assistant(example):
        msgs = example.get("messages", [])
        return (
            isinstance(msgs, list)
            and len(msgs) > 0
            and msgs[-1].get("role") == "assistant"
            and isinstance(msgs[-1].get("content", ""), str)
            and len(msgs[-1]["content"].strip()) > 0
        )

    dataset = dataset.filter(ends_with_assistant)

    def to_text(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    dataset = dataset.map(to_text, num_proc=8)

    # keep only text
    remove_cols = [c for c in dataset.column_names if c != "text"]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    print(f"📊 Final training samples: {len(dataset):,}")
    print("🔎 Example text preview:\n", dataset[0]["text"][:300])

    # ======================
    # SFT config
    # ======================
    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        deepspeed=DS_CONFIG,

        # --- SFT ---
        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,
        dataset_text_field="text",
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,          # ✅ use local model directory
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\n🎯 Training complete! Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
