
# ProxyModelTrain

本项目用于对 **Qwen3-8B** 进行 **Tool-Calling SFT（Supervised Fine-Tuning）**，
数据来源为 **Dolci-Instruct-SFT**，支持 **单卡 / 多卡（DeepSpeed ZeRO-2）** 训练。

---

## 1. 创建并初始化环境

```bash
bash setup_sft_env.sh
source sft-qwen3/bin/activate
```

> 建议：所有训练均在虚拟环境中进行，避免污染系统 Python。

---

## 2. 下载模型（Qwen3-8B）

```bash
python 0_download_model.py
```

模型将下载到：

```
/shared_workspace_mfs/ruochen/models/Qwen3-8B
```

目录中包含 tokenizer、config 以及模型 shard（safetensors）。

---

## 3. 下载数据（Dolci-Instruct-SFT）

```bash
python 0_download_data.py
```

数据保存到：

```
/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.jsonl
```

原始数据包含多轮对话、`function_calls` 与 `environment`，**不能直接用于 Qwen3 SFT**。

---

## 4. 转换为 Qwen3 标准 Tool-Calling SFT 格式

```bash
python 1_prep_qwen3_tool_sft_jsonl.py
```

该脚本会：

* 解析并转换 `function_calls → tool_calls`
* 将嵌套结构 **序列化为字符串**，规避 pyarrow schema 问题
* 丢弃不合法样本（无 user / 无最终 assistant）

输出文件：

```
/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.qwen3_tool_sft.jsonl
```

单条样本格式：

```json
{
  "messages_json": "[{...}]",
  "tools_json": "[{...}]"
}
```

---

## 5. 训练（单卡验证，强烈推荐先跑）

```bash
python 2_train_sft_model.py
```

适用于：

* 验证数据是否能正常 tokenize
* 检查 chat_template 渲染是否正确
* 快速 sanity check（建议配合 `MAX_TRAIN_SAMPLES=1000`）

---

## 6. 训练（8 卡 + DeepSpeed ZeRO-2）

### ⚠️ 强烈建议使用 tmux

#### 6.1 创建 tmux 会话

```bash
tmux new -s qwen3_sft
```

#### 6.2 启动训练

```bash
torchrun --nproc_per_node=8 2_train_sft_model.py
```

#### 6.3 常用 tmux 命令

```text
Ctrl+b d        # detach（后台运行）
tmux attach -t qwen3_sft
tmux ls
tmux kill-session -t qwen3_sft
```

> ⚠️ 不要在普通 SSH 会话中裸跑 torchrun，断线会直接终止训练。

---

## 7. 数据预处理缓存机制（多卡关键设计）

* **仅 rank 0：**

  * 加载 jsonl
  * `apply_chat_template`
  * 生成 `text`
  * `save_to_disk`

* **其他 rank：**

  * 等待 barrier
  * `load_from_disk`

缓存目录：

```
/shared_workspace_mfs/ruochen/datasets/_cache_qwen3_tool_sft_text
```

避免 8 卡重复 tokenize / IO 爆炸。

---

## 8. Checkpoint 保存策略（方案 B：steps = 10）

### 8.1 保存位置

所有模型与 checkpoint 保存到：

```
/shared_workspace_mfs/ruochen/sft_proxy_model/
```

示例结构：

```
sft_proxy_model/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
├── checkpoint-400/
├── trainer_state.json
├── config.json
└── ...
```

---

### 8.2 保存逻辑说明

在 `2_train_sft_model.py` 中：

```python
save_strategy = "steps"
save_steps = 100
save_total_limit = 4
```

含义：

* ✅ **每 100 个 optimizer step 保存一次 checkpoint**
* ✅ **最多保留最近 4 个 checkpoint**
* ❌ 更早的 checkpoint 会被自动删除（滑动窗口）

示例：

当训练到 step = 500 时，仅保留：

```
checkpoint-200
checkpoint-300
checkpoint-400
checkpoint-500
```

适合：

* 小样本调试
* 新数据 / 新模板验证
* 频繁中断风险环境

---

### 8.3 最终模型

训练完成后，会额外保存一份 **最终模型** 到：

```
/shared_workspace_mfs/ruochen/sft_proxy_model/
```

该目录等价于 **最后一个 checkpoint 的完整拷贝**，用于推理或继续训练。

---

## 9. DeepSpeed 配置（ZeRO-2）

配置文件：

```
ds_config_zero2.json
```

说明：

* 使用 ZeRO Stage-2
* optimizer / gradient state 分片
* 显著降低显存占用
* 建议 batch / grad_acc 使用 `auto` 与 HF 对齐

---
