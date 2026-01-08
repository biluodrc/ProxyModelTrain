# -*- coding: utf-8 -*-
# prep_qwen3_tool_sft_jsonl.py

import json
import re
from typing import Any, Dict, List, Optional
from tqdm import tqdm

IN_PATH = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.jsonl"
OUT_PATH = "/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.qwen3_tool_sft.jsonl"

CALL_RE = re.compile(r'^\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*\((.*)\)\s*$')


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def parse_py_style_calls(s: str) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(s, str) or not s.strip():
        return None

    tool_calls = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = CALL_RE.match(line)
        if not m:
            return None

        name, args_str = m.group(1), m.group(2).strip()
        args: Dict[str, Any] = {}

        if args_str:
            parts, buf, depth, in_quote = [], "", 0, None
            for ch in args_str:
                if in_quote:
                    buf += ch
                    if ch == in_quote:
                        in_quote = None
                    continue
                if ch in ("'", '"'):
                    in_quote = ch
                    buf += ch
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(buf.strip())
                    buf = ""
                else:
                    buf += ch
            if buf.strip():
                parts.append(buf.strip())

            for p in parts:
                k, v = p.split("=", 1)
                k, v = k.strip(), v.strip()

                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    val = v[1:-1]
                elif re.fullmatch(r"-?\d+", v):
                    val = int(v)
                elif re.fullmatch(r"-?\d+\.\d+", v):
                    val = float(v)
                else:
                    val = v
                args[k] = val

        tool_calls.append({
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            }
        })

    return tool_calls if tool_calls else None


def normalize_one(ex: Dict[str, Any]):
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        return None, False, None, None

    # extract tools
    tools = []
    for m in msgs:
        if m.get("role") == "system":
            f = m.get("functions")
            if isinstance(f, str) and f.strip():
                try:
                    tools = json.loads(f)
                except Exception:
                    pass
            break

    out_msgs = []
    has_tool = False

    for m in msgs:
        role = m.get("role")
        content = safe_str(m.get("content"))

        if role == "system":
            out_msgs.append({"role": "system", "content": content})

        elif role == "user":
            out_msgs.append({"role": "user", "content": content})

        elif role == "assistant":
            tc_raw = m.get("function_calls")
            tool_calls = parse_py_style_calls(tc_raw) if isinstance(tc_raw, str) else None

            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                has_tool = True

            out_msgs.append(msg)

        elif role == "environment":
            out_msgs.append({"role": "tool", "content": content})

    if not out_msgs or out_msgs[-1]["role"] != "assistant":
        return None, False, None, None

    return {
        "messages_json": json.dumps(out_msgs, ensure_ascii=False),
        "tools_json": json.dumps(tools, ensure_ascii=False),
    }, has_tool, msgs, out_msgs


def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total):
            ex = json.loads(line)
            result, has_tool, raw_msgs, out_msgs = normalize_one(ex)

            if result is None:
                continue

            if has_tool:
                print("\n" + "=" * 100)
                print("🧪 FOUND TOOL-CALLING SAMPLE")
                print("=" * 100)

                print("\n🔹 RAW MESSAGES:")
                print(json.dumps(raw_msgs, ensure_ascii=False, indent=2))

                print("\n🔹 CONVERTED MESSAGES:")
                print(json.dumps(out_msgs, ensure_ascii=False, indent=2))

                print("\n🔹 tools_json:")
                print(result["tools_json"])

                print("\n⛔ breakpoint() hit. Inspect freely.")
                breakpoint()

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
