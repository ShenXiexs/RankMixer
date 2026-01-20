import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3))))


class PerTokenFFN(tf.layers.Layer):
    """
    Per-token FFN with independent parameters for each token.
    """

    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0, name=None):
        super(PerTokenFFN, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = float(dropout)

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight("W1", [self.num_tokens, self.d_model, hidden_dim], initializer=init)
        self.b1 = self.add_weight("b1", [self.num_tokens, hidden_dim], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight("W2", [self.num_tokens, hidden_dim, self.d_model], initializer=init)
        self.b2 = self.add_weight("b2", [self.num_tokens, self.d_model], initializer=tf.zeros_initializer())
        super(PerTokenFFN, self).build(input_shape)

    def call(self, x, training=False):
        h = tf.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        y = tf.einsum("bth,thd->btd", h, self.W2) + self.b2
        if self.dropout and training:
            y = tf.nn.dropout(y, keep_prob=1.0 - self.dropout)
        return y
