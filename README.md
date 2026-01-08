# ProxyModelTrain

## 1. 创建并初始化环境

```bash
bash setup_sft_env.sh
source sft-qwen3/bin/activate
````

---

## 2. 下载模型（Qwen3-8B）

```bash
python 0_download_model.py
```

模型将下载到：

```
/shared_workspace_mfs/ruochen/models/Qwen3-8B
```

---

## 3. 下载数据（Dolci-Instruct-SFT）

```bash
python 0_download_data.py
```

数据将保存到：

```
/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.jsonl
```

---

## 4. 清洗数据（生成稳定 schema）

```bash
python 1_clean_data.py
```

生成文件：

```
/shared_workspace_mfs/ruochen/datasets/Dolci-Instruct-SFT/train.clean.jsonl
```

---

## 5. 训练（单卡验证）

```bash
python 2_train_sft_model.py
```

---

## 6. 训练（8 卡 + DeepSpeed ZeRO-2）

```bash
torchrun --nproc_per_node=8 2_train_sft_model.py
```

---

## 7. 输出

模型与 checkpoint 保存到：

```
/shared_workspace_mfs/ruochen/sft_proxy_model
```

---

## 8. DeepSpeed 配置

```text
ds_config_zero2.json
```

用于 ZeRO-2 并行训练。

```
