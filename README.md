# RankMixer Reproduction Guide

For a quick start version, you may refer to encap version: https://github.com/ShenXiexs/rankMixer-encap

This project is an engineering reproduction of the paper *"RankMixer: Scaling Up Ranking Models in Industrial Recommenders"*.
It implements the core RankMixer architecture in an existing server-side training framework, and adds an optional Sparse-MoE extension plus practical training details. The project files directly map to the production server code structure.

## 1. Project Positioning and Goals

- Reproduce the three core RankMixer designs: **semantic tokenization**, **parameter-free token mixing**, and **per-token FFN**.
- Keep the architecture highly parallel with low memory overhead for large-scale industrial training.
- Support both Dense and Sparse-MoE variants.

## 2. Code Structure

```
config/
  RankMixer_Shen0118/
    train_config.py            # training config entry
    select_feature.conf        # dense feature list
    slot.conf / schema.conf    # training data schema
    boundaries_map.json / fg.json
models/
  rankmixer_shen0118.py         # model entry (Estimator model_fn)
  rankmixer_layers/
    tokenization.py             # Tokenization v1 (default rules)
    token_mixing.py             # parameter-free token mixing
    per_token_ffn.py            # per-token FFN
    sparse_moe.py               # Sparse-MoE version of per-token FFN
paper_ref/
  2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf
```

## 3. End-to-End Pipeline (Input to Output)

In `model_fn` of `models/rankmixer_shen0118.py`, the data flow is roughly:

1. **Embedding Lookup (dynamic table)**
   - Uses dynamic embedding from `tensorflow_recommenders_addons`.
   - Dense features come from `select_feature.conf`; sequence features come from `seq_features_config` in `train_config.py`.
2. **Sequence Feature Pooling**
   - Pooling is applied to each sequence feature (`mean` by default, with `max/target` supported).
3. **Semantic Tokenization**
   - Maps heterogeneous features (dense + pooled sequence features) into `T` fixed-length tokens.
4. **RankMixer Encoder**
   - Multi-layer `RankMixerBlock`: `TokenMixing + Per-token FFN (+ Residual + LN)`.
5. **Pooling + Heads**
   - Mean pooling over the token dimension by default.
   - Then two task heads for CTR/CVR.
6. **Loss / Metrics**
   - CTR uses BCE.
   - CTCVR is computed as `ctr_prob * cvr_prob` and added to total loss.
   - Optional conditional CVR loss.

## 4. Tokenization Details

### 4.1 Tokenization Version Selection

Controlled by `tokenization_version` in `train_config.py`:

- `v1`: uses default grouping rules from `tokenization.py`
- `v2`: uses `tokenization_v2.py`, same behavior as v1 by default (for compatibility/extension)
- `v3`: uses `tokenization_v3.py`, same behavior as v1 by default (for compatibility/extension)

Config example (`config/RankMixer_Shen0118/train_config.py`):

```
"tokenization_strategy": "semantic",
"tokenization_version": "v2",
```

### 4.2 Grouping Logic

Tokenization supports two modes:

1. **Explicit semantic groups (`semantic_groups`)**
   - When `semantic_groups` is non-empty, each group maps to one token.
   - If number of groups is less than `T`, tokens are padded; if more than `T`, extra groups are truncated.
   - If no feature matches within a group, a zero vector is used.

2. **Rule-based grouping (`DEFAULT_SEMANTIC_GROUP_RULES`)**
   - Used when `semantic_groups` is empty.
   - Default rules in v2/v3 are aligned with v1 (no additional built-in feature groups).
   - Features are first ordered by rules, then uniformly chunked into `T` tokens.
   - Features not matched by rules are placed at the end, so no feature is dropped.

### 4.3 Key Implementation Files

- `models/rankmixer_layers/tokenization.py`
- `models/rankmixer_layers/tokenization_v2.py`
- `models/rankmixer_layers/tokenization_v3.py`

## 5. Token Mixing Implementation

File: `models/rankmixer_layers/token_mixing.py`

Core properties:

- **Parameter-free**: only reshape + transpose operations.
- **Strict constraint**: `num_heads == num_tokens`.
- **Divisibility required**: `d_model % num_heads == 0`.

The implementation is equivalent to the paper's multi-head token mixing: split heads -> cross-token rearrangement -> restore token layout.

## 6. Per-token FFN and Sparse-MoE

### 6.1 Per-token FFN

File: `models/rankmixer_layers/per_token_ffn.py`

Each token has an independent set of `W1/W2` parameters:

- `W1: [T, D, kD]`
- `W2: [T, kD, D]`

This avoids mutual suppression across different semantic subspaces and follows the paper design.

### 6.2 Sparse-MoE

File: `models/rankmixer_layers/sparse_moe.py`

Characteristics:

- Multiple experts per token.
- ReLU routing + DTSI (separate routing for training and inference).
- L1 regularization to enforce activation sparsity.

Enabled by the `use_moe` switch in `train_config.py`.

## 7. RankMixerBlock / Encoder

File: `models/rankmixer_shen0118.py`

- `RankMixerBlock`: TokenMixing + Per-token FFN (+ Residual + LN)
  - Supports `ln_style` = pre/post.
- `RankMixerEncoder`: multi-layer stacking + optional final LN.

## 8. Head and Loss Design

In `model_fn`:

- Encoded tokens are pooled (mean by default).
- Passed through a shared MLP head, then split into CTR/CVR logits.
- CTCVR = CTR * CVR.
- Loss = CTR loss + CTCVR loss (+ optional conditional CVR loss).

This part is the engineering task head, decoupled from the RankMixer backbone.

## 9. Training Configuration Highlights

Entry: `config/RankMixer_Shen0118/train_config.py`

Key parameters:

- `d_model / num_layers / num_tokens / num_heads / ffn_mult`
  - control model scale and compute cost.
- `tokenization_version`
  - selects v1/v2 rules.
- `token_mixing_type`
  - default `paper_strict` (requires H = T).
- `enable_timing`
  - optional in-graph timing stats (written to training logs).
- `use_moe / moe_num_experts / moe_sparsity_ratio`
  - related to 1B-scale extension.
- `seq_pool`
  - sequence pooling strategy.
- `use_hkv`
  - whether dynamic embedding uses GPU/HKV; when disabled, CPU Cuckoo table is used.

## 10. How to Run (same as original pipeline)

```
nohup bash test.sh RankMixer_Shen0118 20251101 > /data/share/opt/model/RankMixer_Shen0118/logs/RankMixer_Shen0119.log 2>&1 &
```

The training script is executed by the original project pipeline and invokes `model_fn`.

## 11. Common Notes

- `num_heads == token_count` must hold, otherwise an error is raised.
- If `include_seq_in_tokenization=True` and `semantic_groups` is empty, sequence features participate in sorting/grouping.
- If `embedding_size` differs from `d_model`, a linear projection aligns sequence token dimensions.
- When `semantic_groups` is empty, behavior is not strictly "one group = one token"; it uses rule-based ordering plus uniform splitting.
- For strictly semantic tokens, explicitly define `semantic_groups`.
