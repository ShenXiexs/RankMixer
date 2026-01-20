import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ParameterFreeTokenMixer(tf.layers.Layer):
    """
    Paper-style parameter-free token mixing with strict H = T.
    """

    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0, name=None):
        super(ParameterFreeTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads, got d_model=%d num_heads=%d"
                % (self.d_model, self.num_heads)
            )
        self.d_head = self.d_model // self.num_heads
        super(ParameterFreeTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        # x 形状: [B, T, D]
        batch_size = tf.shape(x)[0]
        t_count = self.num_tokens
        h_count = self.num_heads
        d_head = self.d_head

        # 先拆分 head，再交换 token/head 轴完成混合。
        split = tf.reshape(x, [batch_size, t_count, h_count, d_head])
        shuffled = tf.transpose(split, [0, 2, 1, 3])  # 形状: [B, H, T, D/H]
        merged = tf.reshape(shuffled, [batch_size, h_count, t_count * d_head])  # 形状: [B, H, D]
        mixed = tf.reshape(merged, [batch_size, t_count, self.d_model])  # 形状: [B, T, D]

        if self.dropout and training:
            mixed = tf.nn.dropout(mixed, keep_prob=1.0 - self.dropout)
        return mixed
