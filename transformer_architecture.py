import tensorflow as tf
import numpy as np
# from keras_mlp.layers import ReversibleEmbedding

class PosEncode(tf.keras.layers.Layer):
    def __init__(self,dims,  max_sep_len,):
        super(PosEncode, self).__init__()
        self.max_sep_len = max_sep_len
        self.pos_enc = self._pos_enc(max_sep_len, dims)

    #Add the Embeddings and the Encodings

    def _pos_enc(self, length, dims):
        pos = np.arange(length)[:, np.newaxis]
        j = np.arange(dims)[np.newaxis, :]
        #distribute the angles according to the formula for Positional Encodings
        angle_rates = 1 / np.power(10000, (2 * j)/ dims)
        angle_rads = pos * angle_rates
        #even coordinates get sin, odd coordinates get cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    @tf.function
    def call(self, x):

        seq_len = x.shape[1]
        return tf.slice(self.pos_enc, [0, 0, 0], [-1, seq_len, -1])

class InverseEmbedding(tf.keras.layers.Embedding):

    def __init__(self, voc_size, dims,  **kwargs):
        super(InverseEmbedding, self).__init__(voc_size, dims, **kwargs)

    def call(self, x, inverse = False, training = True):
        if not inverse:
            return super().call(x)

        trans = tf.transpose(tf.convert_to_tensor(self.embeddings))
        return tf.matmul(x, trans)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dims, num_heads, attn_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        # Initialize the parameters
        self.num_heads = num_heads
        self.attn_size = attn_size
        self.dims = dims

        # Layers for QKV projection and output projection
        self.qkv_projection = tf.keras.layers.Dense(num_heads * attn_size * 3, use_bias=False)
        self.output_projection = tf.keras.layers.Dense(dims)
        self.dropout = tf.keras.layers.Dropout(dropout_prob)

    def call(self, qs, mask=None):
        # Compute batch size and sequence length from inputs
        batch_size = tf.shape(qs)[0]
        seq_length = tf.shape(qs)[1]

        # Project queries, keys, and values
        qkv = self.qkv_projection(qs)
        qkv = tf.reshape(qkv, [batch_size, seq_length, self.num_heads * 3, self.attn_size])
        qs, ks, vs = tf.split(qkv, 3, axis=2)

        # Compute scaled dot-product attention
        qs = tf.transpose(qs, [0, 2, 1, 3])
        ks = tf.transpose(ks, [0, 2, 3, 1])
        attn_product = tf.matmul(qs, ks) / tf.math.sqrt(tf.cast(self.attn_size, tf.float32))

        # Apply mask, if provided
        if mask is not None:
            mask = tf.expand_dims(mask, 1)
            mask = tf.expand_dims(mask, 2)
            mask = tf.broadcast_to(mask, [batch_size, self.num_heads, seq_length, seq_length])
            attn_product = tf.where(mask == 0, tf.fill(tf.shape(attn_product), -1e9), attn_product)

        # Softmax and dropout
        scores = tf.nn.softmax(attn_product, axis=-1)
        scores = self.dropout(scores)

        # Weighted sum of values
        vs = tf.transpose(vs, [0, 2, 1, 3])
        res = tf.matmul(scores, vs)

        # Reshape and project to output size
        res = tf.reshape(tf.transpose(res, [0, 2, 1, 3]), [batch_size, -1, self.num_heads * self.attn_size])
        output = self.output_projection(res)

        return output


class SubLayerLogic(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate):
        super(SubLayerLogic, self).__init__()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization()

    def calls(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(SelfAttentionBlock, self).__init__()
        self.ffn = tf.keras.layers.Dense(d_model, "silu")
        self.attention = MultiHeadAttention(d_model, num_heads,d_model//num_heads, dropout_rate)
        self.sub1 = SubLayerLogic(d_model, dropout_rate)
        self.sub2 = SubLayerLogic(d_model, dropout_rate)

    @tf.function
    def call(self, x, mask):
        x = self.sub1(x, lambda inputs : self.attention(inputs, mask))
        return self.sub2(x, self.attention)

class Transformer(tf.keras.Model):
    def __init__(self, voc_size, dims, maxseqlen, num_heads, dropout_rate, n_blocks):
        super(Transformer, self).__init__()

        self.posemb = PosEncode(dims, maxseqlen)
        self.embedding = InverseEmbedding(voc_size, dims)
        self.blocks = [
            SelfAttentionBlock(dims, num_heads, dropout_rate) for _ in range(n_blocks)
        ]

    def encode(self, x):

        return self.posemb(x) + self.embedding(x, inverse=False)

    def gen_mask(self, seq_len):
        return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)[None,None,:]

    @tf.function
    def call(self, x, maks = None):

        seq_len = x.shape[1]

        x = self.encode(x)

        if mask is None:
          mask = self.gen_mask(seq_len)

        for block in self.blocks:
            x = block(x, mask)

        return self.embedding(x, inverse = True)
