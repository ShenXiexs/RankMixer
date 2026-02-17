# RankMixer复现说明

本项目是对论文《RankMixer: Scaling Up Ranking Models in Industrial Recommenders》的工程复现版，
在现有服务器训练框架中落地了 RankMixer 的核心结构，并加入了可选的 Sparse-MoE 扩展与工程化训练细节。项目文件直接对应服务器上的代码框架。

## 1. 项目定位与目标

- 复现 RankMixer 的三大核心设计：**语义 Tokenization**、**参数无关 Token Mixing**、**Per-token FFN**。
- 保持高并行、低显存开销的结构，适配大规模工业训练。
- 支持 Dense 版本与 Sparse-MoE 版本的扩展。

## 2. 代码结构

```
config/
  RankMixer_Shen0118/
    train_config.py            # 训练配置入口
    select_feature.conf        # dense 特征列表
    slot.conf / schema.conf    # 训练数据 schema
    boundaries_map.json / fg.json
models/
  rankmixer_shen0118.py         # 模型主入口（Estimator model_fn）
  rankmixer_layers/
    tokenization.py             # Tokenization v1（默认规则）
    token_mixing.py             # 参数无关 Token Mixing
    per_token_ffn.py            # Per-token FFN
    sparse_moe.py               # Sparse-MoE 版本的 Per-token FFN
paper_ref/
  2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf
```

## 3. 整体流程（从输入到输出）

在 `models/rankmixer_shen0118.py` 的 `model_fn` 中，数据流大致如下：

1. **Embedding Lookup（动态表）**
   - 使用 `tensorflow_recommenders_addons` 的 dynamic embedding。
   - dense 特征来自 `select_feature.conf`，序列特征来自 `train_config.py` 的 `seq_features_config`。
2. **序列特征池化**
   - 对每个序列特征做 pooling（默认 `mean`，支持 `max/target`）。
3. **Semantic Tokenization**
   - 将 heterogeneous 特征（dense + seq pooling）映射为固定长度 `T` 个 token。
4. **RankMixer Encoder**
   - 多层 `RankMixerBlock`：`TokenMixing + Per-token FFN (+ Residual + LN)`。
5. **Pooling + Head**
   - 默认对 token 维度做 mean pooling；
   - 接 CTR/CVR 两个 task head；
6. **Loss / Metric**
   - CTR 用 BCE；
   - CTCVR 使用 `ctr_prob * cvr_prob` 计算，加入 total loss；
   - 可选 conditional CVR loss。

## 4. Tokenization 详细说明

### 4.1 Tokenization 版本选择

通过 `train_config.py` 中的 `tokenization_version` 控制：

- `v1`：使用 `tokenization.py` 的默认分组规则
- `v2`：使用 `tokenization_v2.py`，默认与 v1 一致（用于兼容/扩展）
- `v3`：使用 `tokenization_v3.py`，默认与 v1 一致（用于兼容/扩展）

配置示例（`config/RankMixer_Shen0118/train_config.py`）：

```
"tokenization_strategy": "semantic",
"tokenization_version": "v2",
```

### 4.2 分组逻辑

Tokenization 有两种模式：

1. **显式语义分组（semantic_groups）**
   - 当 `semantic_groups` 非空时，每个 group 对应一个 token；
   - 如果 group 数量少于 `T`，会 pad；多于 `T` 会截断；
   - group 中未匹配到特征时，会用零向量占位。

2. **规则分组（DEFAULT_SEMANTIC_GROUP_RULES）**
   - 当 `semantic_groups` 为空时使用；
   - v2/v3 默认规则与 v1 保持一致（不内置额外特征分组）；
   - 实际流程是：先按规则排序特征，然后均匀 chunk 成 `T` 个 token；
   - 未命中规则的特征会放在排序末尾，保证不会丢特征。

### 4.3 关键实现文件

- `models/rankmixer_layers/tokenization.py`
- `models/rankmixer_layers/tokenization_v2.py`
- `models/rankmixer_layers/tokenization_v3.py`

## 5. Token Mixing 实现

文件：`models/rankmixer_layers/token_mixing.py`

核心特性：

- **参数无关**：只做 reshape + transpose；
- **严格限制**：`num_heads == num_tokens`；
- **d_model 可分**：`d_model % num_heads == 0`；

实现逻辑等价于论文中的 multi-head token mixing：拆 head → 交叉拼接 → 还原 token。

## 6. Per-token FFN 与 Sparse-MoE

### 6.1 Per-token FFN

文件：`models/rankmixer_layers/per_token_ffn.py`

每个 token 独立一套 `W1/W2` 参数：

- `W1: [T, D, kD]`
- `W2: [T, kD, D]`

避免不同语义子空间互相“压制”，符合论文设计。

### 6.2 Sparse-MoE

文件：`models/rankmixer_layers/sparse_moe.py`

特点：

- 每个 token 有多个 experts；
- 使用 ReLU routing + DTSI（训练/推理两套路由）；
- L1 正则约束激活稀疏度；

通过 `train_config.py` 的 `use_moe` 开关控制。

## 7. RankMixerBlock / Encoder

文件：`models/rankmixer_shen0118.py`

- `RankMixerBlock`：TokenMixing + Per-token FFN (+ Residual + LN)
  - 支持 `ln_style` = pre/post
- `RankMixerEncoder`：多层堆叠 + 可选 final LN

## 8. Head 与 Loss 设计

在 `model_fn` 中：

- 对编码后 token 做 pooling（默认 mean）
- 进入共享 MLP head，再分别输出 CTR / CVR logits
- CTCVR = CTR * CVR
- Loss = CTR loss + CTCVR loss (+ optional CVR conditional loss)

这部分是工程任务头，与 RankMixer 主干解耦。

## 9. 训练配置要点

入口：`config/RankMixer_Shen0118/train_config.py`

重点参数：

- `d_model / num_layers / num_tokens / num_heads / ffn_mult`
  - 控制模型规模与计算量
- `tokenization_version`
  - 选择 v1/v2 规则
- `token_mixing_type`
  - 默认 `paper_strict`（要求 H = T）
- `enable_timing`
  - 可选的图内耗时统计（输出到训练日志）
- `use_moe / moe_num_experts / moe_sparsity_ratio`
  - 1B 版扩展相关
- `seq_pool`
  - 序列 pooling 策略
- `use_hkv`
  - 动态 embedding 是否使用 GPU/HKV；关闭时使用 CPU Cuckoo 表

## 10. 运行方式（与原工程一致）

```
nohup bash test.sh RankMixer_Shen0118 20251101 > /data/share/opt/model/RankMixer_Shen0118/logs/RankMixer_Shen0119.log 2>&1 &
```

训练脚本由原工程的 pipeline 调用 `model_fn` 执行。

## 11. 常见注意事项

- `num_heads == token_count` 必须满足，否则会抛错；
- 如果 `include_seq_in_tokenization=True` 且 `semantic_groups` 为空，序列特征会参与排序/分组；
- `embedding_size` 与 `d_model` 不一致时，会对 seq token 做线性投影对齐；
- `semantic_groups` 为空时不是严格“一个 group = 一个 token”，而是规则排序后均匀切分；
- 要严格语义 token，请显式填写 `semantic_groups`。
