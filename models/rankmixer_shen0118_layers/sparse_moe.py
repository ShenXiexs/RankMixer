import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from models.rankmixer_shen0118_layers.per_token_ffn import gelu


class PerTokenSparseMoE(tf.layers.Layer):
    """
    Per-token Sparse MoE with ReLU routing + optional DTSI.
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        mult=4,
        num_experts=4,
        dropout=0.0,
        l1_coef=0.0,
        sparsity_ratio=1.0,
        use_dtsi=True,
        routing_type="relu_dtsi",
        name=None,
    ):
        super(PerTokenSparseMoE, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight(
            "W1",
            [self.num_tokens, self.num_experts, self.d_model, hidden_dim],
            initializer=init,
        )
        self.b1 = self.add_weight(
            "b1",
            [self.num_tokens, self.num_experts, hidden_dim],
            initializer=tf.zeros_initializer(),
        )
        self.W2 = self.add_weight(
            "W2",
            [self.num_tokens, self.num_experts, hidden_dim, self.d_model],
            initializer=init,
        )
        self.b2 = self.add_weight(
            "b2",
            [self.num_tokens, self.num_experts, self.d_model],
            initializer=tf.zeros_initializer(),
        )
        self.gate_w_train = self.add_weight(
            "gate_w_train",
            [self.num_tokens, self.d_model, self.num_experts],
            initializer=init,
        )
        self.gate_b_train = self.add_weight(
            "gate_b_train",
            [self.num_tokens, self.num_experts],
            initializer=tf.zeros_initializer(),
        )
        if self.use_dtsi:
            self.gate_w_infer = self.add_weight(
                "gate_w_infer",
                [self.num_tokens, self.d_model, self.num_experts],
                initializer=init,
            )
            self.gate_b_infer = self.add_weight(
                "gate_b_infer",
                [self.num_tokens, self.num_experts],
                initializer=tf.zeros_initializer(),
            )
        super(PerTokenSparseMoE, self).build(input_shape)

    def _router_logits(self, x, w, b):
        # 每个 token 的路由 logits，用于专家选择。
        return tf.einsum("btd,tde->bte", x, w) + b

    def call(self, x, training=False):
        # x 形状: [B, T, D]
        # 计算每个 token 的专家输出。
        h = tf.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        expert_out = tf.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout and training:
            expert_out = tf.nn.dropout(expert_out, keep_prob=1.0 - self.dropout)

        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            # 训练阶段使用 soft 路由以提高专家覆盖。
            gate_train = tf.nn.softmax(gate_train_logits, axis=-1)
        elif self.routing_type == "relu":
            gate_train = tf.nn.relu(gate_train_logits)
        else:
            raise ValueError("Unsupported routing_type: %s" % self.routing_type)

        if self.use_dtsi:
            # 推理阶段使用 ReLU gate 以获得稀疏激活。
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = tf.nn.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # 训练/推理选择不同 gate。
        gate = gate_train if training else gate_infer
        y = tf.reduce_sum(expert_out * tf.expand_dims(gate, -1), axis=2)

        if self.l1_coef > 0.0:
            # L1 惩罚鼓励稀疏专家激活。
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            l1_loss = self.l1_coef * scale * tf.reduce_mean(tf.reduce_sum(gate_infer, axis=-1))
        else:
            l1_loss = tf.constant(0.0)
        return y, l1_loss
