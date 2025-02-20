from typing import Tuple

import jax.random
import jax.numpy as np

rng = jax.random.PRNGKey(0)

def fully_connected(num_in, num_out) -> Tuple[np.ndarray, np.ndarray]:
    return (
        jax.random.uniform(rng, (num_in, num_out)),
        jax.random.uniform(rng, (num_out,)),
    )

def embedding(num_embeddings: int, embed_dim: int) -> np.ndarray:
    return jax.random.uniform(rng, (num_embeddings, embed_dim))
