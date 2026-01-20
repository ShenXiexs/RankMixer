# RankMixer 论文与复现

## 1) 论文（RankMixer: Scaling Up Ranking Models in Industrial Recommenders）

Zhu, Jie, et al. "Rankmixer: Scaling up ranking models in industrial recommenders." *Proceedings of the 34th ACM International Conference on Information and Knowledge Management*. 2025.

### 1.1 论文动机

工业排序模型面对的输入是大量异构特征，传统 attention 类模型在 token 数量很大时：

- ###### 计算和显存开销高；
- ###### 训练吞吐低，难以 scaling。

RankMixer 的目标是用更轻量的 token 交互形式替代 attention，同时保留强表达能力，从而提升 MFU 与可扩展性。

### 1.2 论文提出的三块核心设计

**(A) Semantic Tokenization（语义 token)**

对应的是图片左下角部分，**先按语义分组（Semantic Grouping）**，论文里面指出用户侧一组、视频侧一组、序列行为一组、交叉特征一组…… 这一部分是 *人工 + domain knowledge*，不是模型学出来的。这个很关键，分组后**组内 concat，再切块（Proj + split）**，把同一语义组里的 embedding 拼成一个长向量，再等分切成固定维度的 token（每个 D 维）。总结就是降维加对齐，后面的所有操作（mix / FFN）**都默认 token 是语义一致的**

- 将原始特征按**`<u>`语义分组`</u>`**，得到固定数量的 token（T 个），每个 token 维度为 D。
- 语义分组让每个 token 有“可解释语义”，避免简单展平导致的语义混杂。
- 这样做的好处：

  - 将高维稀疏特征压缩为固定 T 个 token，模型复杂度与 T 解耦，更易扩展。
  - 每个 token 对应清晰语义簇，减少无意义交互，提升表示稳定性。
- 与常见做法的差异：

  - 传统 MLP/Embedding 直接拼接特征，语义混杂且特征主导问题明显。
  - Attention 需要 token 之间两两交互，T 大时成本高。

**(B) Parameter-free Token Mixing**

每个 token 的一部分子空间被送去和所有其他 token 的对应子空间拼接

- 论文给出无参数的 token mixing（Split -> Shuffle -> Merge）。
- 关键约束：**H = T**，保证 reshape 能回到 `[B, T, D]`。
- 与 attention 相比，完全无参数、计算稳定、吞吐高。
- 这样做的好处：
  - 无参数意味着显存更小、训练更稳，且无额外优化难度。
  - 复杂度线性于 token 数量，不存在 attention 的 O(T^2) 交互开销。
- 与常见做法的差异：
  - Attention 用可学习的 Q/K/V 做交互，表达强但成本高、吞吐低。
  - RankMixer 用固定混合实现 token 信息流动，牺牲少量灵活性换取可扩展性。

**(C) Per-token FFN**

每一个 PFFNₜ **就是一个普通的两层 MLP**：ℝ^D  →  ℝ^{kD}  →  ℝ^D

T token 数；D 宽度；k FFN 扩展倍数； L 层数

一个 FFNₜ：第一层：`D × kD` ；第二层：`kD × D`

```
总参数量（忽略 bias）：≈ 2 · k · D²，则 T 个就是  ≈ 2 · k · T · D²
```

- 每个 token 单独建模，参数不共享。
- FFN 隐层为 `k * D`，k 为扩展系数。
- 强化“每个语义 token 的独立表达能力”。
- 这样做的好处：

  - 每个 token 有独立容量，避免共享 FFN 导致的“强特征主导”。
  - 扩展模型规模时可通过 `T/D/L/k` 直接放大参数量，所有参数增长，几乎全来自 Per-token FFN。
- 与常见做法的差异：

  - Transformer/MLP 通常共享 FFN，token 之间竞争同一表达空间。
  - RankMixer 让 token 之间“混合后再独立建模”，更符合语义分组假设。

  | 架构                     | 输入                 | 参数               |
  | ------------------------ | -------------------- | ------------------ |
  | MMoE                     | 同一个输入           | 多专家             |
  | Transformer FFN          | 多 token             | 同一 FFN           |
  | **RankMixer PFFN** | **不同 token** | **不同参数** |

### 1.3 RankMixer Block 结构

论文中每一层 Block 的概念可概括为：

- Pre-LN + Residual
- Token Mixing 子层 + Per-token FFN 子层

对应流程：

1) `X -> LN -> TokenMix -> Residual`
2) `-> LN -> Per-token FFN -> Residual`

这样做的原因与收益：

- Pre-LN 结构训练更稳定，深层堆叠时梯度更平滑。
- TokenMix 先做全局信息交换，再用 per-token FFN 做局部增强，符合“先融合、再专门化”的直觉。

### 1.4 RankMixer 编码后

- token 经 L 层 RankMixer 编码后，通过 mean pooling 得到一个样本级向量 z， 再由任务专属的线性（或轻量 MLP）head 输出 CTR / CVR。

### 1.5 论文总结

```
Semantic Tokenization
   |
   |-- if semantic_groups: group -> concat -> proj
   |-- else: rule order -> chunk -> proj
   v
Tokens X: [B, T, D]
   |
   |-- (optional) input LN
   v
RankMixer Encoder (L blocks)
   |
   |-- Block:
   |   LN -> TokenMix (param-free) -> Residual
   |   LN -> Per-token FFN (or MoE) -> Residual
   v
Encoded Tokens: [B, T, D]
   |
   |-- pooling (mean / cls)
   v
Head MLP
   |
   |-- ctr_logits -> ctr_prob
   |-- cvr_logits -> cvr_prob
   v
Final Scores
   |
   |-- ctcvr_prob = ctr_prob * cvr_prob

```

- RankMixer 在工业排序数据上优于传统 MLP/Attention 基线；
- Scaling law 更陡，模型规模增大收益显著；
- 更容易达到高 MFU，训练吞吐更高。

总结：

- **可扩展性**：语义 token + 无参混合将交互成本从 O(T^2) 降到近似线性，可更大规模训练。
- **表达能力**：per-token FFN 保证每个语义 token 的独立容量，提升细粒度建模能力。
- **工程效率**：更高 MFU 来自于更规整的矩阵操作与更少的注意力开销。

## 2) 复现方式与代码实现

### 2.1 代码结构与模块映射

模型入口：

- `models/rankmixer_shen0118.py`

核心组件（独立文件，便于管理）：

- `models/rankmixer_shen0118_layers/token_mixing.py`
- `models/rankmixer_shen0118_layers/per_token_ffn.py`
- `models/rankmixer_shen0118_layers/sparse_moe.py`
- `models/rankmixer_shen0118_layers/tokenization.py`
- `models/rankmixer_shen0118_layers/__init__.py`

配置入口：

- `config/RankMixer_Shen0118/train_config.py`

### 2.2 Tokenization 在代码中的实现

**入口类**：`SemanticTokenizer`（`tokenization.py`）

**输入**：

- dense embeddings：来自 `select_feature` 的稠密特征
- seq embeddings：来自 `seq_features_config` 的序列特征（先 pool）

**流程**：

1) 将每个特征名映射到一个 embedding 向量。
2) 如果配置了 `semantic_groups`：
   - 每个 group 聚合若干 feature embedding（concat）。
   - 使用线性投影到 `d_model` 维度。
   - 每个 group 对应一个 token。
3) 若 `semantic_groups` 为空：
   - 按 `DEFAULT_SEMANTIC_GROUP_RULES` 进行规则排序；
   - 将排序后的特征均分切成 `T` 组；
   - 每组 concat + linear projection -> token。

**要点**：

- `semantic_groups` 为空时属于“快速跑通模式”，非严格语义 token。
- 论文严格语义 token 需要提供显式 `semantic_groups`。

### 2.3 Token Mixing 在代码中的实现

**类**：`ParameterFreeTokenMixer`（`token_mixing.py`）

**核心约束**：

- 无任何可学习参数（仅 reshape/transpose）。
- 强制 `num_heads == num_tokens`，否则直接报错。

**行为**：

1) `X: [B, T, D]` reshape 到 `[B, T, H, D/H]`
2) transpose -> `[B, H, T, D/H]`
3) reshape -> `[B, H, D]`
4) reshape -> `[B, T, D]`

这严格对应论文的 parameter-free token mixing。

### 2.4 Per-token FFN 在代码中的实现

**类**：`PerTokenFFN`（`per_token_ffn.py`）

**特点**：

- 参数按 token 维度展开：`W1/W2` 都有 token 维度。
- 使用 `tf.einsum` 并行计算每个 token 的 FFN。
- 激活函数为 GELU。

### 2.5 Sparse MoE（可选）实现方式

**类**：`PerTokenSparseMoE`（`sparse_moe.py`）

**机制**：

- 每个 token 有多个 experts。
- 训练时：
  - 若 `routing_type=relu_dtsi`，训练路由采用 softmax（dense），推理路由采用 ReLU（sparse）。
- 推理时：
  - 用 ReLU gate 进行稀疏激活。
- L1 正则控制稀疏度（`moe_l1_lambda`）。

### 2.6 RankMixer Block / Encoder

在 `models/rankmixer_shen0118.py` 中：

- `RankMixerBlock` 组合 Token Mixing + Per-token FFN（或 MoE）
- `RankMixerEncoder` 堆叠 `L` 层
- 默认采用 Pre-LN（论文结构）

### 2.7 Head 与 Loss

**Head**：

- pooling 默认 `mean`（可选 `cls`）
- 两层 MLP 输出 CTR 和 CVR logits

**Loss**：

- CTR + CTCVR（默认开启）
- 可选点击条件 CVR loss（默认关闭）
- MoE L1 loss 在 `use_moe=True` 时加到总 loss

### 2.8 优化器与学习率

优化器：Adam + warmup + grad clip学习率规则：

- 1101/1102 使用 `5e-4`
- 1103 及以后使用 `2e-4`
  由 `--time_str` 或 `--end_time_str` 控制。

## 3) 当前配置与 Scaling Up

当前 Shen0118 配置：

- `D = 768`
- `T = 64`
- `H = 64`
- `L = 8`
- `k = 8`
- `train_batch_size = 2048`

参数量主要由 Per-token FFN 决定，近似：

```
Param ~= 2 * k * T * D^2 * L
```

因此通过增加 `D/T/L/k` 即可做 scaling up。

与 Base_RankMixer 相比：

- Shen0118 结构对齐论文要求；
- 参数规模已经不小于 Base；
- token mixing 为严格无参版本。

## 4) Token 是如何筛选与分组的？

### 4.1 特征来源

- dense 特征来自 `select_feature.conf`（`common/utils.py` 读取）。
- 序列特征来自 `seq_features_config`（`train_config.py` 定义），先做 pooling。

### 4.2 严格语义分组

在 `config/RankMixer_Shen0118/train_config.py` 明确写入：

```
"semantic_groups": {
    "group_0": ["feature_a", "feature_b"],
    ...
    "group_63": [...]
}
```

- 每个 group 直接对应一个 token；
- 组数必须等于 `T`（否则语义不严格）；
- 这是最符合论文“语义 token”定义的方式。

### 4.3 默认规则分组

若 `semantic_groups` 为空：

- 使用 `DEFAULT_SEMANTIC_GROUP_RULES` 对特征排序；
- 再均分切成 `T` 份组成 token。

规则分组示例：

- `core_id`: `combination_un_id`
- `seq`: `seq::` / `seq_`
- `item_meta`: `brand_name`, `first_category`, `second_category`, `annual_vol`, `shop_id`...
- `price`: `reserve_price`, `final_promotion_price`, `commission`, `commission_rate`
- `adslot`: `adslot_id`, `ad_unit_id`, `template_id`, `promotion_type`...
- `app`: `app_pkg`, `app_first_type`...
- `device`: `device_*`, `network`, `ip_city`...
- `user_stat` / `item_stat`: `user__*`, `item__*`

> 若需要严格语义分组，请明确填充 `semantic_groups`，并保证组数与 T 一致。

## 5) l20 运行方式

```
export TRAIN_CONFIG=config/RankMixer_Shen0118/train_config.py
nohup bash test.sh RankMixer_Shen0118 20251101 > /data/share/opt/model/RankMixer_Shen0118/logs/RankMixer_Shen0119.log 2>&1 &
```

### 推理与评估输出

模型在 PREDICT / EVAL 分支输出符合 Estimator 规范：

- PREDICT: `export_outputs` 中包含 `serving_default`
- EVAL: 返回 `loss` 与 `eval_metric_ops`
