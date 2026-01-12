# -*- coding: utf-8 -*-
"""
2_train_sft_model.py
Train Qwen3-8B with tool-calling trajectories using TRL SFTTrainer.

Input jsonl format (per line):
  {"messages_json":"[...]", "tools_json":"[...]"}

This version fixes NCCL ALLREDUCE timeout caused by slow/uneven preprocessing across ranks:
- Rank0: load jsonl -> render chat_template -> tokenize -> save_to_disk (input_ids/attention_mask)
- Other ranks: wait for READY flag on shared FS -> load_from_disk only
- No dist.barrier() for preprocessing sync (avoid NCCL barrier timeout); use filesystem flag instead.
"""

import os
import json
import time
import shutil
import random
import argparse
import datetime as dt

import torch
import torch.distributed as dist

from datasets import load_dataset, load_from_disk, Features, Value
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig


# ======================
# EDIT THESE CONSTANTS
# ======================
MODEL_PATH = "/shared_workspace_mfs/ruochen/models/Qwen3-8B-Base"
TRAIN_JSONL = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.qwen3_tool_sft.jsonl"
OUTPUT_DIR = "/shared_workspace_mfs/ruochen/sft_proxy_model_2M-Base"
DS_CONFIG = "ds_config_zero2.json"

# how much data to really train
MAX_TRAIN_SAMPLES = None      # set None for full dataset

SEED = 42
MAX_LENGTH = 32768
NUM_PROC = 16

PER_DEVICE_BS = 4
GRAD_ACC = 4
LR = 5e-5
EPOCHS = 1

COMPLETION_ONLY_LOSS = False  # tool-calling data: usually keep False

# ---- checkpoint saving ----
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 10

# where to cache tokenized dataset (shared filesystem!)
PREP_DIR = "/shared_workspace_mfs/ruochen/datasets/_cache_qwen3_tool_sft_tok"
# ======================

READY_FLAG = os.path.join(PREP_DIR, "_READY")
TMP_DIR = PREP_DIR + ".__tmp__"


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_init_if_needed():
    """
    Initialize NCCL process group if launched with torchrun.
    Increase timeout to avoid false kills if something is slow (still recommended to fix root cause).
    """
    if is_dist() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=dt.timedelta(hours=2),
        )
        torch.cuda.set_device(get_local_rank())


def wait_for_ready(flag_path: str, poll_sec: float = 2.0, timeout_sec: float = 24 * 3600):
    """
    Poll shared filesystem for READY flag. Avoid NCCL barrier (which can itself timeout).
    """
    t0 = time.time()
    while not os.path.exists(flag_path):
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Timed out waiting for READY flag: {flag_path}")
        time.sleep(poll_sec)


def safe_rmtree(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def build_tokenized_dataset_rank0(tokenizer, max_train_samples: int | None):
    """
    Rank0 only:
    - load json dataset
    - apply chat template -> text
    - tokenize -> input_ids/attention_mask
    - save_to_disk atomically to PREP_DIR
    """
    os.makedirs(PREP_DIR, exist_ok=True)

    # If already prepared and READY exists, reuse (fast path)
    if os.path.exists(READY_FLAG):
        print(f"✅ Found READY flag, reuse cached tokenized dataset: {PREP_DIR}")
        return

    # Clean any previous tmp artifacts
    safe_rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Remove READY if exists but dataset not valid (paranoia)
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)

    features = Features({
        "messages_json": Value("string"),
        "tools_json": Value("string"),
    })

    print(f"📂 Loading dataset: {TRAIN_JSONL}")
    ds = load_dataset(
        "json",
        data_files=TRAIN_JSONL,
        split="train",
        features=features,
    )
    print(f"✅ Loaded: {len(ds):,}")

    if max_train_samples is not None:
        ds = ds.shuffle(seed=SEED).select(range(min(max_train_samples, len(ds))))
        print(f"🎯 Using {len(ds):,} samples (max_samples={max_train_samples})")

    def to_text(ex):
        messages = json.loads(ex["messages_json"])
        tools = []
        tj = ex.get("tools_json", None)
        if tj is None or tj == "":
            tj = "[]"
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

    # keep only text for tokenization
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    print("🔎 Preview text:\n", ds[0]["text"])

    def tok_batch(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        return out

    print("🔪 Tokenizing text -> input_ids/attention_mask ...")
    ds = ds.map(
        tok_batch,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # keep only what trainer needs
    keep_cols = {"input_ids", "attention_mask"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    print(f"💾 Saving tokenized dataset to tmp: {TMP_DIR}")
    ds.save_to_disk(TMP_DIR)

    # Atomic replace: move TMP_DIR -> PREP_DIR
    # We first remove PREP_DIR content except TMP, then rename.
    print("🧱 Atomic commit of cached dataset ...")
    safe_rmtree(PREP_DIR)
    os.makedirs(os.path.dirname(PREP_DIR), exist_ok=True)
    os.rename(TMP_DIR, PREP_DIR)

    # Create READY flag last
    with open(READY_FLAG, "w") as f:
        f.write("ok\n")

    print(f"✅ Cached tokenized dataset ready: {PREP_DIR}")


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

    max_train_samples = MAX_TRAIN_SAMPLES if args.max_samples is None else args.max_samples

    if rank == 0:
        print(f"🧠 rank={rank} world_size={os.environ.get('WORLD_SIZE','1')}")
        print(f"📌 MODEL_PATH={MODEL_PATH}")
        print(f"📌 TRAIN_JSONL={TRAIN_JSONL}")
        print(f"📌 OUTPUT_DIR={OUTPUT_DIR}")
        print(f"📌 max_samples={max_train_samples}")
        print(f"💾 save_strategy=steps, save_steps={SAVE_STEPS}, save_total_limit={SAVE_TOTAL_LIMIT}")
        print(f"🧊 PREP_DIR={PREP_DIR}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Preprocess/tokenize once on rank0; others wait on READY flag --------
    if rank == 0:
        build_tokenized_dataset_rank0(tokenizer, max_train_samples)

    # Everyone waits for READY (filesystem)
    wait_for_ready(READY_FLAG, poll_sec=2.0)

    # All ranks load tokenized dataset
    ds = load_from_disk(PREP_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Note: dataset already has input_ids/attention_mask, so we DO NOT set dataset_text_field.
    config = SFTConfig(
        output_dir=OUTPUT_DIR,

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
