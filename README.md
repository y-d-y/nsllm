# nsllm — 从零开始的大语言模型训练教程

**不用任何训练框架，只用 PyTorch 原生 API**，从 Tokenizer 开始，亲手完成一个大语言模型的 **预训练（Pretrain）→ 指令微调（SFT）** 全流程，并在 2×A100-40GB 上跑通单卡与 DDP 分布式两种训练方案。

> 这个项目的目的不是训练一个很强的模型，而是**把每一个环节都亲手写一遍**，真正搞懂一个 LLM 从无到有的每一步发生了什么。

---

## 🎯 为什么做这个项目

现成的框架（transformers / trainer / accelerate）一条命令就能跑通训练，但也把细节全部屏蔽了。这个仓库选择**全部用 PyTorch 原生 API 手写**，为了弄清楚这些问题：

- RoPE 旋转位置编码到底在旋转什么？`cos/sin` 表是怎么构造、怎么按位置偏移的？
- GQA（分组查询注意力）中 K/V 是如何按 `kv_group_size` 复用的？KV Cache 的 prefill / decode / 批量续写三种路径分别怎么处理？
- 批量训练时 padding 的 mask 和 causal mask 是怎么相加融合进 `scaled_dot_product_attention` 的？
- SFT 时为什么 label 要用 `-100` 填充、只对 assistant 部分计算 loss？
- 预训练词表要加 2 个对话 special token 时，Embedding 和 lm_head 怎么扩容、新增行用什么方式初始化？
- 流式数据集怎么 shuffle、怎么切验证集才能不泄漏？
- DDP 的 `DistributedSampler`、梯度同步、loss 的 all_reduce 平均、rank 0 独占的 eval/保存分别怎么做？

代码里保留了大量的中文注释和手推示意图（例如 `model.py` 中增量 decode 时 future_mask 的构造过程），希望它既能跑起来，也能**读得懂**。

## 🏗️ 模型架构

122M 参数的 Llama 风格 decoder-only 模型，所有组件从零手写：

| 配置 | 值 | 配置 | 值 |
| --- | --- | --- | --- |
| 参数量 | ~122M | 词表 | 32,000（SFT 扩到 32,002） |
| layers | 12 | hidden / emb_dim | 2048 / 768 |
| 注意力头 | 12（head_dim 64） | KV 头（GQA） | 2 |
| 上下文长度 | 2048 | 激活 | SwiGLU |
| 位置编码 | RoPE (theta=10000) | 归一化 | RMSNorm（Pre-Norm） |

核心实现见 [model.py](model.py)：

- **RoPE**：预计算 `cos/sin` 表，支持 `position_offset`，增量 decode 时新 token 能接上历史的旋转位置；
- **GQA + KV Cache**：`past_key_values` 逐层缓存 K/V，`repeat_interleave` 展开 KV 头；注意力分三条路径——首步 prefill 用 `is_causal=True`、单 token decode、以及带 past 的批量续写（手推 future_mask + pad_mask 相加）；
- **RMSNorm / SwiGLU FFN**：均按论文公式直接实现；
- 训练直接使用 PyTorch 原生 `F.scaled_dot_product_attention`（自动走 FlashAttention/高效 kernel）。

## 📦 Tokenizer：从零训练 BPE

[tokenizer_trainer.py](tokenizer_trainer.py) 用 HuggingFace `tokenizers` 在预训练语料上**从零训练了一个 32,000 词表的 ByteLevel BPE**，特殊 token：`<unk> <pad> <bos> <eos>`。

[tokenizer_fixed.py](tokenizer_fixed.py) 修复了训练后 decoder 缺失 ByteLevel 的问题（否则 decode 出来的是乱码字节串）。

SFT 阶段新增 `<|im_start|>` `<|im_end|>` 两个 token（ChatML 风格对话模板），并把模型的 Embedding / lm_head 各扩容 2 行：旧行拷贝原权重，新增行随机初始化（`std=0.02`）——代码里对比了「均值初始化」与「随机初始化」两种方案，结构性 token 推荐随机初始化。

## 🚂 训练流程

### 1. Pretrain（单卡）

[model_pre_trainer.py](model_pre_trainer.py)

- **数据**：FineWeb（~6B tokens），`IterableDataset` 流式读取，document 之间插入 `<eos>` 防止无边界拼接；用「同 seed shuffle + `skip_sequences`」的技巧从同一份数据流中切出 10,000 条验证集，保证不泄漏；
- **优化**：AdamW（weight_decay 0.01），cosine 衰减 3e-4 → 3e-5，10,000 步 warmup；
- **规模**：batch_size 12 × 2048 tokens，237,710 步，约 58 亿 token（基本 1 epoch）；
- **耗时**：单张 A100 约 **117 小时**（≈ 1.4 万 tokens/s）；
- 每 5,000 步在验证集上评估，每 100 步把训练曲线实时刷写到 PNG。

**Pretrain Loss（train ≈ 3.3，eval ≈ 3.28）：**

![pretrain loss](pretrain_loss_curve.png)

### 2. SFT（单卡 & DDP 双卡两种实现）

数据：Magpie-Pro-300K-Filtered（270,000 训练 / 30,000 验证），ChatML 模板拼接 system prompt，**loss 只统计 assistant 回复部分**（user 部分与 padding 的 label 全部置 `-100`，配合 `cross_entropy` 的 `ignore_index`），动态 padding + attention_mask。

| | [单卡版](model_sft_trainer.py) | [DDP 双卡版](model_sft_ddp_trainer.py) |
| --- | --- | --- |
| 启动方式 | `python model_sft_trainer.py` | `torchrun --nproc_per_node=2 model_sft_ddp_trainer.py` |
| 数据切分 | 全量数据 | `DistributedSampler` 按 rank 切半 |
| 每步吞吐 | batch 12 | batch 12 × 2 卡 = 全局 24 |
| 总步数 | 22,400 | 11,230（步数减半，每步双倍数据） |
| 超参 | lr 3e-5→3e-6 cosine，warmup 5%，grad clip 1.0 | 同左 |

DDP 版额外处理了这些细节：loss 先 `all_reduce` 求均值再记录（否则每个 rank 只看到一半数据）；eval / 画曲线 / 存权重只在 rank 0 执行，其他 rank 用 `barrier` 同步等待；`sampler.set_epoch(epoch)` 保证每轮 shuffle 不同。

**两个实现的 loss 曲线——最终都收敛到 eval loss ≈ 1.63，DDP 以一半的步数达到相同的收敛点：**

| 单卡 SFT（22,400 步） | DDP 双卡 SFT（11,230 步） |
| :---: | :---: |
| ![sft loss](sft_loss_curve.png) | ![sft ddp loss](sft_ddp_loss_curve.png) |

### 3. 推理验证

- [pretrain_inference_test.ipynb](pretrain_inference_test.ipynb)：预训练模型的续写测试，验证 KV Cache 与无 cache 结果的一致性；
- [sft_inference_test.ipynb](sft_inference_test.ipynb)：SFT 模型的对话测试，对比了「带 / 不带 KV Cache」「带 / 不带 attention mask」的生成。

模型确实学会了对话格式——能按 ChatML 模板在 `assistant` 角色下生成、并在 `<|im_end|>` 处停止：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
tell me a joke<|im_end|>
<|im_start|>assistant
...
```

坦率地说：122M 的模型 + 6B token 的预训练量，它学会了流利的英语结构和对话格式，但内容的连贯性和知识性还很有限——这恰好也是这个项目想直观感受的：**能力是「模型规模 × 数据量」堆出来的，而格式和语感是这么小的投入就能学会的**。

## 🚀 复现步骤

```bash
# 1. 环境
pip install torch datasets tokenizers matplotlib   # 本项目: torch 2.5.1+cu121, 2×A100-40GB

# 2. 下载数据（按需修改脚本内的路径）
bash data/hf_download.sh

# 3. 从零训练 BPE tokenizer + 修复 decoder
python tokenizer_trainer.py
python tokenizer_fixed.py

# 4. 预训练（单卡，流式读取）
python model_pre_trainer.py

# 5. SFT —— 二选一或都跑
python model_sft_trainer.py                 # 单卡
torchrun --nproc_per_node=2 model_sft_ddp_trainer.py   # DDP 双卡

# 6. 推理测试
jupyter notebook sft_inference_test.ipynb
```

## 📂 仓库结构

```
nsllm/
├── model.py                        # 模型结构：RoPE / GQA / KV Cache / RMSNorm / SwiGLU
├── tokenizer_trainer.py            # 从零训练 32k 词表 BPE tokenizer
├── tokenizer_fixed.py              # 修复 ByteLevel decoder
├── model_pre_trainer.py            # 预训练（单卡，流式 FineWeb，cosine + warmup）
├── model_sft_trainer.py            # SFT 单卡实现
├── model_sft_ddp_trainer.py        # SFT DDP 双卡实现（torchrun 启动）
├── sft_data_explore.ipynb          # SFT 数据探索
├── pretrain_inference_test.ipynb   # 预训练模型续写 & KV Cache 一致性验证
├── sft_inference_test.ipynb        # SFT 模型对话推理测试
├── pretrain_loss_curve.png         # 预训练 loss 曲线
├── sft_loss_curve.png              # SFT 单卡 loss 曲线
├── sft_ddp_loss_curve.png          # SFT DDP loss 曲线
├── data/                           # 数据下载脚本（数据本体不入库）
└── model/                          # tokenizer json / config / 训练产物
```

## 🗺️ Roadmap

- [ ] DPO 对齐训练（数据已备好）
- [ ] 在 GSM8K / MMLU / ARC / HellaSwag / HumanEval 上做标准化评测（已下载）
- [ ] 混入中文与安全领域语料的继续预训练
- [ ] bf16 混合精度 + `torch.compile` 加速
- [ ] 更大规模：更多数据与更深的模型，验证 scaling 对 loss 的影响

## 🙏 致谢

- 预训练数据：[FineWeb](https://huggingface.co/datasets/weights-and-wires/fineweb-6b)
- SFT 数据：[Magpie-Align/Magpie-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered)
- 架构参考：Llama 2 / Llama 3（RoPE、GQA、RMSNorm、SwiGLU）
