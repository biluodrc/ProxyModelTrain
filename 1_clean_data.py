# -*- coding: utf-8 -*-
"""
Clean Dolci JSONL to a schema-stable JSONL for HF datasets.

Input:  train.jsonl (may have function_calls/functions with mixed types)
Output: train.clean.jsonl with only:
  {"messages": [{"role": "...", "content": "..."}, ...]}
"""

import os
import json

IN_PATH  = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.jsonl"
OUT_PATH = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.clean.jsonl"

# 可选：只保留以 assistant 结尾的样本（推荐）
FILTER_ENDS_WITH_ASSISTANT = True

def main():
    assert os.path.exists(IN_PATH), f"Input not found: {IN_PATH}"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n_in = 0
    n_out = 0
    n_bad = 0

    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            msgs = ex.get("messages", None)
            if not isinstance(msgs, list) or len(msgs) == 0:
                n_bad += 1
                continue

            clean_msgs = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                content = m.get("content", "")

                # 强制转成 string，避免 None / 非字符串
                role = "" if role is None else str(role)
                content = "" if content is None else str(content)

                role = role.strip()
                # content 不要 strip 到空（有些内容前后空格无所谓），这里轻微处理
                content = content.rstrip()

                if role and content:
                    clean_msgs.append({"role": role, "content": content})

            if not clean_msgs:
                n_bad += 1
                continue

            if FILTER_ENDS_WITH_ASSISTANT and clean_msgs[-1]["role"] != "assistant":
                continue

            fout.write(json.dumps({"messages": clean_msgs}, ensure_ascii=False) + "\n")
            n_out += 1

            if n_in % 200000 == 0:
                print(f"Processed {n_in:,} lines -> {n_out:,} saved, bad={n_bad:,}")

    print(f"✅ Done. Read {n_in:,} lines, saved {n_out:,}, bad={n_bad:,}")
    print("Output:", OUT_PATH)

if __name__ == "__main__":
    main()
