# 模型入口：models/rankmixer_shen0118.py
# RankMixer_Shen0118 Estimator：严格论文版 token mixing + per-token FFN（可选 MoE）。
from collections import OrderedDict

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de

from models.ctr_hstu_seq import LayerNorm
from models.rankmixer_shen0118_layers import (
    ParameterFreeTokenMixer,
    PerTokenFFN,
    PerTokenSparseMoE,
    SemanticTokenizer,
    SemanticTokenizerV2,
    gelu,
)
from common.utils import select_feature, train_config as TrainConfig, seq_features_config
from common.metrics import evaluate

logger = tf.compat.v1.logging


# 若未配置 seq_length，则根据 seq_features_config 构建。
if not hasattr(TrainConfig, "seq_length"):
    TrainConfig.seq_length = OrderedDict(
        (cfg["name"], cfg["length"])
        for cfg in seq_features_config
        if cfg.get("is_download", 1) == 1
    )

class RankMixerBlock(tf.layers.Layer):
    """
    RankMixer block: TokenMixing + Per-token FFN with residuals.
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        name=None,
    ):
        super(RankMixerBlock, self).__init__(name=name)
        self.ln1 = LayerNorm(name="ln1")
        self.ln2 = LayerNorm(name="ln2")
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)

        self.token_mixer = ParameterFreeTokenMixer(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            dropout=token_dp,
            name="token_mixer",
        )

        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                num_experts=moe_experts,
                dropout=ffn_dp,
                l1_coef=moe_l1_coef,
                sparsity_ratio=moe_sparsity_ratio,
                use_dtsi=moe_use_dtsi,
                routing_type=moe_routing_type,
                name="per_token_moe",
            )
        else:
            self.per_token_ffn = PerTokenFFN(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                dropout=ffn_dp,
                name="per_token_ffn",
            )
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        moe_loss = tf.constant(0.0)
        if self.ln_style == "post":
            # Post-LN：先 mix/ffn，再残差相加并归一化。
            y = self.token_mixer(x, training=training)
            x = self.ln1(x + y)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x, training=training)
            else:
                z = self.per_token_ffn(x, training=training)
            out = self.ln2(x + z)
        else:
            # Pre-LN：先归一化，再 mix/ffn，最后残差相加。
            y = self.ln1(x)
            y = self.token_mixer(y, training=training)
            x = x + y
            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z, training=training)
            else:
                z = self.per_token_ffn(z, training=training)
            out = x + z
        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(tf.layers.Layer):
    """Stack RankMixer blocks."""

    def __init__(
        self,
        num_layers,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        use_final_ln=True,
        name=None,
    ):
        super(RankMixerEncoder, self).__init__(name=name)
        self.use_final_ln = bool(use_final_ln)
        self.blocks = [
            RankMixerBlock(
                num_tokens=num_tokens,
                d_model=d_model,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                token_dp=token_dp,
                ffn_dp=ffn_dp,
                ln_style=ln_style,
                use_moe=use_moe,
                moe_experts=moe_experts,
                moe_l1_coef=moe_l1_coef,
                moe_sparsity_ratio=moe_sparsity_ratio,
                moe_use_dtsi=moe_use_dtsi,
                moe_routing_type=moe_routing_type,
                name="block_%d" % i,
            )
            for i in range(num_layers)
        ]
        self.final_ln = LayerNorm(name="encoder_ln")
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        out = x
        moe_losses = []
        for blk in self.blocks:
            out = blk(out, training=training)
            moe_losses.append(blk.moe_loss)
        self.moe_loss = tf.add_n(moe_losses) if moe_losses else tf.constant(0.0)
        return self.final_ln(out) if self.use_final_ln else out


def _dense_if_sparse(x, default_value=""):
    return tf.sparse.to_dense(x, default_value=default_value) if isinstance(x, tf.SparseTensor) else x


def _pad_trunc_to_length(tokens_dense, length):
    cur = tf.shape(tokens_dense)[1]
    tokens_cut = tokens_dense[:, :tf.minimum(cur, length)]
    pad_len = tf.maximum(0, length - tf.shape(tokens_cut)[1])
    # 用空字符串补齐到固定长度。
    tokens_fix = tf.pad(tokens_cut, paddings=[[0, 0], [0, pad_len]])
    tokens_fix.set_shape([None, length])
    return tokens_fix


def _get_seq_embedding(tokens_2d, embeddings_table, policy, name="seq_lookup"):
    batch_size = tf.shape(tokens_2d)[0]
    length = tf.shape(tokens_2d)[1]
    flat = tf.reshape(tokens_2d, [-1])
    uniq, idx = tf.unique(flat)
    # 将字符串 token 哈希为 int64 id 以做动态 embedding 查表。
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])
    update_tstp_op = policy.apply_update(ids)
    restrict_op = policy.apply_restriction(int(1e8))
    emb_u, _ = de.embedding_lookup(embeddings_table, ids, return_trainable=True, name=name)
    gathered = tf.gather(emb_u, idx)
    seq_emb = tf.reshape(gathered, [batch_size, length, embeddings_table.dim])
    return seq_emb, update_tstp_op, restrict_op


def _get_dense_emb_from_features(features, embeddings_table, policy):
    x = features["features"]
    x = _dense_if_sparse(x, default_value="")
    batch_size = tf.shape(x)[0]
    fea_size = len(select_feature)
    flat = tf.reshape(x, [-1])
    uniq, idx = tf.unique(flat)
    # 将字符串 token 哈希为 int64 id 以做动态 embedding 查表。
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])
    update_tstp_op = policy.apply_update(ids)
    restrict_op = policy.apply_restriction(int(1e8))
    emb_u, _ = de.embedding_lookup(embeddings_table, ids, return_trainable=True, name="features_lookup")
    gathered = tf.gather(emb_u, idx)
    dense_emb = tf.reshape(gathered, [batch_size, fea_size * embeddings_table.dim])
    return dense_emb, update_tstp_op, restrict_op


def _sequence_pool(seq_emb, tokens, mode="mean"):
    mode = str(mode).lower()
    # 将 "" 和 "0" 视作序列 padding。
    pad_mask = tf.logical_or(tf.equal(tokens, ""), tf.equal(tokens, "0"))
    valid = tf.cast(tf.logical_not(pad_mask), tf.float32)
    denom = tf.reduce_sum(valid, axis=1, keepdims=True) + 1e-6
    if mode == "mean":
        summed = tf.reduce_sum(seq_emb * tf.expand_dims(valid, -1), axis=1)
        return summed / denom
    if mode == "max":
        neg_inf = tf.cast(-1e9, seq_emb.dtype)
        masked = tf.where(tf.expand_dims(pad_mask, -1), tf.fill(tf.shape(seq_emb), neg_inf), seq_emb)
        max_val = tf.reduce_max(masked, axis=1)
        has_valid = tf.reduce_any(tf.logical_not(pad_mask), axis=1, keepdims=True)
        return tf.where(has_valid, max_val, tf.zeros_like(max_val))
    if mode == "target":
        counts = tf.reduce_sum(valid, axis=1)
        last_idx = tf.maximum(tf.cast(counts, tf.int32) - 1, 0)
        batch_idx = tf.range(tf.shape(tokens)[0])
        gather_idx = tf.stack([batch_idx, last_idx], axis=1)
        gathered = tf.gather_nd(seq_emb, gather_idx)
        gathered = tf.where(tf.expand_dims(counts > 0, -1), gathered, tf.zeros_like(gathered))
        return gathered
    raise ValueError("Unsupported seq_pool mode: %s" % mode)


def _parse_pool_modes(value, default="mean"):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip().lower() for v in value.split(",") if v.strip()]
    else:
        items = []
    return items or [default]


def _prepare_seq_tokens(features, embeddings_table, policy, seq_cfg, pool_modes, restrict, update_ops,
                        return_names=False):
    if not seq_cfg:
        return (None, 0, []) if return_names else (None, 0)
    if "seq_features" not in features:
        logger.warning("features missing seq_features, RankMixer uses dense-only tokens.")
        return (None, 0, []) if return_names else (None, 0)
    seq_features_flat = _dense_if_sparse(features["seq_features"], default_value="0")
    start = 0
    seq_tokens = []
    seq_names = []
    for seq_col, length in seq_cfg.items():
        # 按配置长度切分每个序列列的原始 tokens。
        tokens_slice = seq_features_flat[:, start:start + length]
        start += length
        mask = tf.equal(tokens_slice, tf.constant("0", dtype=tf.string))
        empty = tf.fill(tf.shape(tokens_slice), "")
        tokens_slice = tf.where(mask, empty, tokens_slice)
        tokens = _pad_trunc_to_length(tokens_slice, length)
        seq_emb, up_s, rs_s = _get_seq_embedding(tokens, embeddings_table, policy, name="%s_lookup" % seq_col)
        update_ops.append(up_s)
        if restrict:
            update_ops.append(rs_s)
        for pool_mode in pool_modes:
            pooled = _sequence_pool(seq_emb, tokens, mode=pool_mode)
            seq_tokens.append(pooled)
            if return_names:
                seq_names.append("seq::%s::%s" % (seq_col, pool_mode))
    if not seq_tokens:
        return (None, 0, []) if return_names else (None, 0)
    seq_stack = tf.stack(seq_tokens, axis=1)
    return (seq_stack, len(seq_tokens), seq_names) if return_names else (seq_stack, len(seq_tokens))


def _binary_cross_entropy_from_probs(labels, probs, eps=1e-7):
    labels = tf.cast(labels, tf.float32)
    probs = tf.clip_by_value(probs, eps, 1.0 - eps)
    loss = -labels * tf.math.log(probs) - (1.0 - labels) * tf.math.log(1.0 - probs)
    return tf.reduce_mean(loss)


def _compute_bce_from_logits(logits, labels):
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(tf.cast(logits, tf.float32), [-1, 1])
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(ce)
    prob = tf.nn.sigmoid(logits)
    return loss, prob


def _pick_label(source_features, source_labels, candidates):
    for key in candidates:
        if key in source_features:
            return source_features[key]
    if source_labels is not None:
        if isinstance(source_labels, dict):
            for key in candidates:
                if key in source_labels:
                    return source_labels[key]
        else:
            return source_labels
    return None


def _get_train_date_from_flags():
    try:
        flags = tf.app.flags.FLAGS
        time_str = getattr(flags, "end_time_str", None) or getattr(flags, "time_str", None)
        if time_str:
            return str(time_str)[:8]
    except Exception:
        return None
    return None


def _resolve_learning_rate(opt_cfg):
    lr = float(opt_cfg.get("learning_rate", 1e-3))
    schedule = opt_cfg.get("lr_schedule") or opt_cfg.get("lr_by_date")
    if not schedule:
        return lr
    cutoff = str(schedule.get("cutoff_date", "")).strip()
    if not cutoff:
        return lr
    before = float(schedule.get("before", lr))
    after = float(schedule.get("after", lr))
    train_date = _get_train_date_from_flags()
    if not train_date:
        return lr
    return before if train_date <= cutoff else after


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    rank_cfg = params.get("rankmixer", {}) if params else {}

    # 模型规模与正则相关参数。
    d_model = int(rank_cfg.get("d_model", 128))
    num_layers = int(rank_cfg.get("num_layers", 2))
    num_tokens = int(rank_cfg.get("num_tokens", rank_cfg.get("semantic_target_tokens", 16)))
    num_heads = int(rank_cfg.get("num_heads", num_tokens))
    ffn_mult = int(rank_cfg.get("ffn_mult", 4))
    token_dp = float(rank_cfg.get("token_mixing_dropout", 0.0))
    ffn_dp = float(rank_cfg.get("ffn_dropout", 0.0))
    use_other = bool(rank_cfg.get("use_other_features", True))
    seq_pool_modes = _parse_pool_modes(rank_cfg.get("seq_pool", ["mean"]))
    embedding_size = int(rank_cfg.get("embedding_size", 9))
    add_cls_token = bool(rank_cfg.get("add_cls_token", False))
    use_input_ln = bool(rank_cfg.get("use_input_ln", rank_cfg.get("input_ln", True)))
    input_dropout = float(rank_cfg.get("input_dropout", 0.0))
    head_dropout = float(rank_cfg.get("head_dropout", 0.0))

    # Tokenization 策略与版本选择。
    tokenization = str(rank_cfg.get("tokenization_strategy", rank_cfg.get("tokenization", "semantic"))).lower()
    tokenization_version = str(rank_cfg.get("tokenization_version",
                                            rank_cfg.get("tokenizer_version", "v1"))).lower()
    token_mixing_type = str(rank_cfg.get("token_mixing_type", "paper_strict")).lower()
    ln_style = str(rank_cfg.get("ln_style", "pre")).lower()
    use_final_ln = bool(rank_cfg.get("use_final_ln", True))
    per_token_ffn = bool(rank_cfg.get("per_token_ffn", True))

    semantic_groups = rank_cfg.get("semantic_groups")
    semantic_group_rules = rank_cfg.get("semantic_group_rules")
    token_projection = rank_cfg.get("token_projection", "linear")
    include_seq_in_tokenization = bool(rank_cfg.get("include_seq_in_tokenization", True))

    output_pooling = str(rank_cfg.get("output_pooling", rank_cfg.get("pooling", "mean"))).lower()

    use_moe = bool(rank_cfg.get("use_moe", False))
    moe_experts = int(rank_cfg.get("moe_num_experts", 4))
    moe_l1_coef = float(rank_cfg.get("moe_l1_lambda", rank_cfg.get("moe_l1_coef", 0.0)))
    moe_sparsity_ratio = float(rank_cfg.get("moe_sparsity_ratio", 1.0))
    moe_routing_type = str(rank_cfg.get("moe_routing_type", "relu_dtsi")).lower()
    moe_use_dtsi = bool(rank_cfg.get("moe_use_dtsi", True))

    use_ctcvr_loss = bool(rank_cfg.get("use_ctcvr_loss", True))
    use_conditional_cvr = bool(rank_cfg.get("use_conditional_cvr", False))

    ps_num = int(params.get("ps_num", 0)) if params else 0
    restrict = bool(params.get("restrict", False)) if params else False

    # 配置 TFRA 动态 embedding 的设备分布。
    device = params.get("device", "CPU") if params else "CPU"
    if is_training:
        devices_info = ["/job:localhost/replica:0/task:{}/{}:{}".format(i, device, i) for i in range(ps_num)]
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    elif mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        devices_info = ["/job:localhost/replica:0/task:{}/{}:0".format(0, device) for _ in range(ps_num)]
        initializer = tf.compat.v1.zeros_initializer()
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for _ in range(ps_num)]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0:
        devices_info = "/job:localhost/replica:0/task:0/CPU:0"

    embeddings_table = tfra.dynamic_embedding.get_variable(
        name="embeddings", dim=embedding_size, devices=devices_info,
        trainable=is_training, initializer=initializer)

    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings_table)
    update_tstp_op = policy.apply_update(tf.constant([0], dtype=tf.int64))
    restrict_op = policy.apply_restriction(int(1e8))
    groups = [update_tstp_op]
    if restrict:
        groups.append(restrict_op)

    seq_cfg = TrainConfig.seq_length
    update_ops = []

    dense_embeddings = None
    dense_feature_names = []
    if use_other:
        # dense 特征后续会作为单独 token 参与。
        other_emb, up_t, rs_t = _get_dense_emb_from_features(features, embeddings_table, policy)
        update_ops.append(up_t)
        if restrict:
            update_ops.append(rs_t)
        fea_size = len(select_feature)
        dense_embeddings = tf.reshape(other_emb, [-1, fea_size, embedding_size])
        dense_embeddings.set_shape([None, fea_size, embedding_size])
        dense_feature_names = list(select_feature)

    seq_embeddings = None
    seq_names = []
    seq_tokens, seq_token_count, seq_names = _prepare_seq_tokens(
        features, embeddings_table, policy, seq_cfg, seq_pool_modes, restrict, update_ops, return_names=True
    )
    if seq_tokens is not None:
        seq_embeddings = seq_tokens
        seq_embeddings.set_shape([None, seq_token_count, embedding_size])

    # RankMixer 仅支持 semantic tokenization 与无参 token mixing。
    if tokenization != "semantic":
        raise ValueError("RankMixer_Shen0118 expects semantic tokenization.")

    if token_mixing_type not in ("paper_strict", "paper", "param_free"):
        raise ValueError("RankMixer_Shen0118 only supports parameter-free token mixing.")

    if tokenization_version in ("v2", "tokenization_v2", "semantic_v2"):
        # v2 使用更新后的默认语义分组规则。
        tokenizer_cls = SemanticTokenizerV2
    else:
        tokenizer_cls = SemanticTokenizer

    tokenizer = tokenizer_cls(
        target_tokens=num_tokens,
        d_model=d_model,
        embedding_dim=embedding_size,
        semantic_groups=semantic_groups,
        group_rules=semantic_group_rules,
        token_projection=token_projection,
    )

    if include_seq_in_tokenization:
        # 将序列池化特征纳入语义 tokenization。
        tokens, token_count = tokenizer.tokenize(
            dense_embeddings, dense_feature_names, seq_embeddings, seq_names
        )
        seq_token_count = 0
    else:
        tokens, token_count = tokenizer.tokenize(
            dense_embeddings, dense_feature_names, None, None
        )
        if seq_embeddings is not None and seq_token_count > 0:
            seq_proj = seq_embeddings
            if embedding_size != d_model:
                seq_proj = tf.compat.v1.layers.dense(seq_proj, units=d_model, activation=None, name="seq_token_proj")
            tokens = tf.concat([tokens, seq_proj], axis=1)
            token_count += seq_token_count

    if token_count != num_tokens:
        logger.warning("token_count=%d differs from configured num_tokens=%d", token_count, num_tokens)

    if add_cls_token:
        cls_embed = tf.compat.v1.get_variable(
            "rankmixer_cls_token", shape=[1, 1, d_model],
            initializer=tf.random_normal_initializer(stddev=0.02))
        cls_token = tf.tile(cls_embed, [tf.shape(tokens)[0], 1, 1])
        tokens = tf.concat([cls_token, tokens], axis=1)
        token_count += 1

    # 无参 token mixing 强制 H = T。
    if num_heads != token_count:
        raise ValueError("paper_strict token mixing requires num_heads == token_count.")

    if not per_token_ffn and not use_moe:
        raise ValueError("RankMixer_Shen0118 expects per-token FFN (or MoE).")

    if use_input_ln:
        input_ln = LayerNorm(name="input_ln")
        tokens = input_ln(tokens)
    if input_dropout and is_training:
        tokens = tf.nn.dropout(tokens, keep_prob=1.0 - input_dropout)

    encoder = RankMixerEncoder(
        num_layers=num_layers,
        num_tokens=token_count,
        d_model=d_model,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        token_dp=token_dp,
        ffn_dp=ffn_dp,
        ln_style=ln_style,
        use_moe=use_moe,
        moe_experts=moe_experts,
        moe_l1_coef=moe_l1_coef,
        moe_sparsity_ratio=moe_sparsity_ratio,
        moe_use_dtsi=moe_use_dtsi,
        moe_routing_type=moe_routing_type,
        use_final_ln=use_final_ln,
        name="rankmixer_encoder",
    )
    encoded = encoder(tokens, training=is_training)
    encoded.set_shape([None, token_count, d_model])

    # 将 tokens 池化为单向量供任务塔使用。
    if output_pooling in ("mean", "avg"):
        head_input = tf.reduce_mean(encoded, axis=1)
    elif output_pooling == "cls":
        if not add_cls_token:
            raise ValueError("output_pooling=cls requires add_cls_token=True.")
        head_input = encoded[:, 0, :]
    else:
        raise ValueError("Unknown output_pooling: %s" % output_pooling)

    if head_dropout and is_training:
        head_input = tf.nn.dropout(head_input, keep_prob=1.0 - head_dropout)

    def _build_task_tower(input_tensor, scope_name):
        # 每个任务使用两层 MLP 头。
        with tf.compat.v1.variable_scope(scope_name):
            net = tf.compat.v1.layers.dense(input_tensor, units=d_model * 2, activation=gelu, name="dense1")
            if head_dropout and is_training:
                net = tf.nn.dropout(net, keep_prob=1.0 - head_dropout)
            net = tf.compat.v1.layers.dense(net, units=d_model, activation=gelu, name="dense2")
            logit = tf.compat.v1.layers.dense(net, units=1, activation=None, name="logit")
        return logit

    ctr_logits = _build_task_tower(head_input, "ctr_tower")
    cvr_logits = _build_task_tower(head_input, "cvr_tower")

    ctr_label_raw = _pick_label(features, labels, ["click_label", "ctr_label", "is_click"])
    if ctr_label_raw is None:
        ctr_label_raw = tf.zeros([tf.shape(ctr_logits)[0]], tf.float32)
    ctr_label = tf.reshape(tf.cast(ctr_label_raw, tf.float32), [-1, 1])

    ctcvr_label_raw = _pick_label(features, labels, ["ctcvr_label", "is_conversion"])
    has_ctcvr = ctcvr_label_raw is not None
    if ctcvr_label_raw is None:
        ctcvr_label_raw = tf.zeros([tf.shape(ctr_logits)[0]], tf.float32)
    ctcvr_label = tf.reshape(tf.cast(ctcvr_label_raw, tf.float32), [-1, 1])

    ctr_loss, ctr_prob = _compute_bce_from_logits(ctr_logits, ctr_label)
    cvr_prob = tf.nn.sigmoid(cvr_logits)
    ctcvr_prob = ctr_prob * cvr_prob
    ctcvr_loss = _binary_cross_entropy_from_probs(ctcvr_label, ctcvr_prob) if has_ctcvr else tf.constant(0.0)

    # 总 loss 可选叠加 conditional CVR 与 MoE 正则。
    total_loss = ctr_loss + (ctcvr_loss if use_ctcvr_loss else 0.0)

    cvr_loss = tf.constant(0.0, dtype=tf.float32)
    if use_conditional_cvr:
        click_mask = tf.reshape(tf.cast(ctr_label, tf.float32), [-1, 1])
        cvr_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctcvr_label, logits=cvr_logits)
        denom = tf.reduce_sum(click_mask) + 1e-6
        cvr_loss = tf.reduce_sum(cvr_ce * click_mask) / denom
        total_loss = total_loss + cvr_loss

    moe_l1_loss = encoder.moe_loss if use_moe else tf.constant(0.0)
    if use_moe:
        total_loss = total_loss + moe_l1_loss

    eval_metric_ops = OrderedDict()
    evaluate(ctr_label, ctr_prob, "task1_ctr", eval_metric_ops)
    evaluate(ctcvr_label, ctcvr_prob, "task1_ctcvr", eval_metric_ops)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings_table.size(),
        "losses": total_loss,
        "ctr_losses": ctr_loss,
        "ctcvr_losses": ctcvr_loss,
        "rankmixer_tokens": tf.constant(int(token_count), dtype=tf.int32),
        "rankmixer_heads": tf.constant(int(num_heads), dtype=tf.int32),
    })
    if use_conditional_cvr:
        loggings["cvr_losses"] = cvr_loss
    if use_moe:
        loggings["moe_l1_loss"] = moe_l1_loss
    for k, (val, up_op) in eval_metric_ops.items():
        loggings[k] = val
        groups.append(up_op)
    groups.extend(update_ops)

    # 输出预测供线上服务与离线评估。
    batch_size = tf.shape(ctr_prob)[0]
    out_tensor = tf.concat([ctr_label, ctcvr_label, ctr_prob, cvr_prob, ctcvr_prob], axis=1)
    predictions = {
        "requestid": features.get("requestid", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "combination_un_id": features.get("combination_un_id", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "out": out_tensor
    }

    class _ModelView(object):
        pass

    model = _ModelView()
    model.losses = total_loss
    model.ctr_losses = ctr_loss
    model.ctcvr_losses = ctcvr_loss
    model.ctr_labels = [ctr_label]
    model.ctr_probs = [ctr_prob]
    model.cvr_probs = [cvr_prob]
    model.ctcvr_labels = [ctcvr_label]
    model.ctcvr_probs = [ctcvr_prob]
    model.predictions = predictions
    model.outputs = {
        "out": out_tensor,
        "ctr_output": ctr_prob,
        "cvr_output": cvr_prob,
        "ctcvr_output": ctcvr_prob
    }

    predict_outputs = model.outputs
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput(predict_outputs)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            export_outputs=export_outputs)

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            loss=model.losses,
            eval_metric_ops=eval_metric_ops)

    # 对可训练变量做 L2 正则（不含 bias）。
    trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    l2_reg = float(params.get("l2_reg", 1e-6)) if params else 1e-6
    if l2_reg > 0:
        l2_loss = l2_reg * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                     for v in trainable_variables if 'bias' not in v.name])
        model.losses = model.losses + l2_loss
        loggings["l2_loss"] = l2_loss

    opt_cfg = params.get("optimize_config", {}) if params else {}
    learning_rate = _resolve_learning_rate(opt_cfg)
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.999))
    epsilon = float(opt_cfg.get("epsilon", 1e-8))

    warmup_steps = int(opt_cfg.get("warmup_steps", 0))
    decay_steps = int(opt_cfg.get("decay_steps", 0))
    decay_type = str(opt_cfg.get("decay_type", "none")).lower()
    min_learning_rate = float(opt_cfg.get("min_learning_rate", 0.0))
    grad_clip_norm = float(opt_cfg.get("grad_clip_norm", 0.0))

    # 学习率调度与可选 warmup。
    lr = tf.constant(learning_rate, dtype=tf.float32)
    if decay_steps and decay_steps > 0 and decay_type in ("cosine", "poly", "polynomial"):
        gs = tf.cast(global_step, tf.int32)
        if decay_type == "cosine":
            lr_decay = tf.compat.v1.train.cosine_decay(
                learning_rate=learning_rate,
                global_step=gs,
                decay_steps=decay_steps,
                alpha=min_learning_rate / learning_rate if learning_rate > 0 else 0.0,
            )
        else:
            lr_decay = tf.compat.v1.train.polynomial_decay(
                learning_rate=learning_rate,
                global_step=gs,
                decay_steps=decay_steps,
                end_learning_rate=min_learning_rate,
                power=1.0,
            )
        lr = lr_decay
    if warmup_steps and warmup_steps > 0:
        warm = tf.cast(tf.minimum(global_step + 1, warmup_steps), tf.float32) / float(warmup_steps)
        lr = tf.where(global_step < warmup_steps, lr * warm, lr)

    loggings["lr"] = lr

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(opt)
    grads_and_vars = opt.compute_gradients(model.losses)
    grads = [g for g, _ in grads_and_vars if g is not None]
    grad_norm = tf.linalg.global_norm(grads) if grads else tf.constant(0.0, tf.float32)
    loggings["grad_norm"] = grad_norm
    if grad_clip_norm and grad_clip_norm > 0 and grads:
        # 裁剪全局范数以提升训练稳定性。
        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_clip_norm)
        it = iter(clipped_grads)
        clipped_gv = []
        for g, v in grads_and_vars:
            if g is None:
                clipped_gv.append((g, v))
            else:
                clipped_gv.append((next(it), v))
        grads_and_vars = clipped_gv
    dense_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    train_op = tf.group(dense_op, *groups)

    log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                      loss=model.losses, train_op=train_op, training_hooks=[log_hook])
