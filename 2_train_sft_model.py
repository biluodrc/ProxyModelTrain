# -*- coding: utf-8 -*-
"""
2_train_sft_model.py
Train Qwen3-8B with tool-calling trajectories using TRL SFTTrainer.

Input jsonl format (per line):
  {"messages_json":"[...]", "tools_json":"[...]"}

Key features:
- Avoid pyarrow nested schema issues (we parse json strings in map).
- Rank0 does preprocessing (render chat_template -> text) and save_to_disk once.
  Other ranks wait and load from disk. (No duplicated preprocessing)
- Control training data size ONLY by max sample count (deterministic & reproducible)
- Save checkpoints by STEPS, keep the last 4 checkpoints
"""

import os
import json
import random
import argparse
import torch
import torch.distributed as dist

from datasets import load_dataset, load_from_disk, Features, Value
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig


# ======================
# EDIT THESE CONSTANTS
# ======================
MODEL_PATH = "/shared_workspace_mfs/ruochen/models/Qwen3-8B"
TRAIN_JSONL = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.qwen3_tool_sft.jsonl"
OUTPUT_DIR = "/shared_workspace_mfs/ruochen/sft_proxy_model_2M"
DS_CONFIG = "ds_config_zero2.json"

# how much data to really train
MAX_TRAIN_SAMPLES = 100000      # set None for full dataset

SEED = 42
MAX_LENGTH = 2048
NUM_PROC = 16

PER_DEVICE_BS = 4
GRAD_ACC = 4
LR = 2e-5
EPOCHS = 1

COMPLETION_ONLY_LOSS = False  # tool-calling data: usually keep False

# ---- checkpoint saving (Plan B) ----
SAVE_STEPS = 10            # save every N optimizer steps
SAVE_TOTAL_LIMIT = 4        # keep last 4 checkpoints

# where to cache rendered text dataset (shared filesystem!)
PREP_DIR = "/shared_workspace_mfs/ruochen/datasets/_cache_qwen3_tool_sft_text"
# ======================


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_init_if_needed():
    if is_dist() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_local_rank())


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="override MAX_TRAIN_SAMPLES (deterministic cap)",
    )
    args = parser.parse_args()

    ddp_init_if_needed()
    rank = get_rank()

    max_train_samples = (
        MAX_TRAIN_SAMPLES if args.max_samples is None else args.max_samples
    )

    if rank == 0:
        print(f"🧠 rank={rank} world_size={os.environ.get('WORLD_SIZE','1')}")
        print(f"📌 MODEL_PATH={MODEL_PATH}")
        print(f"📌 TRAIN_JSONL={TRAIN_JSONL}")
        print(f"📌 OUTPUT_DIR={OUTPUT_DIR}")
        print(f"📌 max_samples={max_train_samples}")
        print(f"💾 save_strategy=steps, save_steps={SAVE_STEPS}, save_total_limit={SAVE_TOTAL_LIMIT}")

    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    features = Features({
        "messages_json": Value("string"),
        "tools_json": Value("string"),
    })

    # ---- Preprocess once on rank0 ----
    if rank == 0:
        os.makedirs(PREP_DIR, exist_ok=True)

        print(f"📂 Loading converted dataset: {TRAIN_JSONL}")
        ds = load_dataset(
            "json",
            data_files=TRAIN_JSONL,
            split="train",
            features=features,
        )
        print(f"✅ Loaded: {len(ds):,}")

        if max_train_samples is not None:
            ds = ds.shuffle(seed=SEED).select(
                range(min(max_train_samples, len(ds)))
            )
            print(f"🎯 Using {len(ds):,} samples (max_samples={max_train_samples})")

        def to_text(ex):
            messages = json.loads(ex["messages_json"])
            tools = []
            tj = ex.get("tools_json", "[]")
            if tj:
                try:
                    tools = json.loads(tj)
                    if not isinstance(tools, list):
                        tools = []
                except Exception:
                    tools = []
            ex["text"] = tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=False,
            )
            return ex

        print("🧩 Rendering chat_template -> text ...")
        ds = ds.map(to_text, num_proc=NUM_PROC)

        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
        print("🔎 Preview text:\n", ds[0]["text"][:400])

        print(f"💾 Saving preprocessed dataset to: {PREP_DIR}")
        ds.save_to_disk(PREP_DIR)

    ddp_barrier()

    # ---- All ranks load rendered dataset ----
    ds = load_from_disk(PREP_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = SFTConfig(
        output_dir=OUTPUT_DIR,

        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,

        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,

        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,

        # ✅ Plan B: save by steps, keep last 4 checkpoints
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        report_to="tensorboard",

        deepspeed=DS_CONFIG,

        completion_only_loss=COMPLETION_ONLY_LOSS,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,
        args=config,
        processing_class=tokenizer,
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    if rank == 0:
        print(f"\n🎯 Training complete! Saved to: {OUTPUT_DIR}")
        print(f"📌 Checkpoints: {OUTPUT_DIR}/checkpoint-* (keeping last {SAVE_TOTAL_LIMIT})")


if __name__ == "__main__":
    main()
