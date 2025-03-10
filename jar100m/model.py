import jax
import jax.numpy as jnp
import equinox as eqx
import math

EMBED_DIMENSIONS = 72

class SelfAttentionHead(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    head_size: int
    affinity_tri: jax.Array

    def __init__(self, in_size: int, head_size: int, context_window_len: int, key):
        qk, kk, vk = jax.random.split(key, 3)
        self.head_size = head_size
        self.query = eqx.nn.Linear(in_size, head_size, use_bias=False, key=qk)
        self.key = eqx.nn.Linear(in_size, head_size, use_bias=False, key=kk)
        self.value = eqx.nn.Linear(in_size, head_size, use_bias=False, key=vk)
        affinity_tri = jnp.tril(jnp.ones((context_window_len, context_window_len))) == 0
        self.affinity_tri = affinity_tri

    def __call__(self, x: jax.Array) -> jax.Array:
        k = jax.vmap(self.key)(x)
        q = jax.vmap(self.query)(x)
        affinities = (q @ k.T) / math.sqrt(self.head_size)
        
        num_toks = x.shape[0]
        affinity_tri = self.affinity_tri[:num_toks, :num_toks]
        decoder_affinities = jnp.where(affinity_tri, float("-inf"), affinities)
        
        normalized = jax.nn.softmax(decoder_affinities, axis=-1)
        return normalized @ jax.vmap(self.value)(x)

class MultiHeadSelfAttention(eqx.Module):
    heads: list

    def __init__(self, num_heads: int, in_size: int, head_size: int, context_window_len: int, key):
        keys = jax.random.split(key, num_heads)
        self.heads = [
            SelfAttentionHead(in_size, head_size, context_window_len, k)
            for k in keys
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.concatenate([head(x) for head in self.heads], axis=-1)

class TransformerBlock(eqx.Module):
    attention: MultiHeadSelfAttention
    mlp: eqx.nn.Sequential

    def __init__(self, context_window_len: int, key):
        attention_key, mlp_key1, mlp_key2 = jax.random.split(key, 3)
        self.attention = MultiHeadSelfAttention(
            num_heads=4,
            in_size=EMBED_DIMENSIONS,
            head_size=EMBED_DIMENSIONS // 4,
            context_window_len=context_window_len,
            key=attention_key,
        )
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(EMBED_DIMENSIONS, EMBED_DIMENSIONS * 4, key=mlp_key1),
            lambda x, key: jax.nn.relu(x),
            eqx.nn.Linear(EMBED_DIMENSIONS * 4, EMBED_DIMENSIONS, key=mlp_key2),
            lambda x, key: jax.nn.relu(x),
        ])

    def __call__(self, x: jax.Array, key) -> jax.Array:
        x = x + self.attention(x)
        x = x + jax.vmap(self.mlp)(x)
        return x

class Model(eqx.Module):
    info_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding
    blocks: eqx.nn.Sequential
    unembed: eqx.nn.Linear
    context_window_len: int

    def __init__(self, vocab_len: int, context_window_len: int, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.context_window_len = context_window_len
        self.info_embedding = eqx.nn.Embedding(vocab_len, EMBED_DIMENSIONS, key=k1)
        self.position_embedding = eqx.nn.Embedding(context_window_len, EMBED_DIMENSIONS, key=k2)
        self.blocks = eqx.nn.Sequential([
            TransformerBlock(context_window_len, k3 + i) for i in range(3)
        ])
        self.unembed = eqx.nn.Linear(EMBED_DIMENSIONS, vocab_len, key=k4)

    def __call__(self, x: jax.Array) -> jax.Array:
        num_toks = x.shape[0]
        embedding_fn = jax.vmap(self.info_embedding)
        embedding = embedding_fn(x[-self.context_window_len:])

        arange = jnp.arange(min(num_toks, self.context_window_len))
        pos_fn = jax.vmap(self.position_embedding)
        position_embeddings = pos_fn(arange)
        
        x = self.blocks(embedding + position_embeddings)
        return jax.vmap(self.unembed)(x)
