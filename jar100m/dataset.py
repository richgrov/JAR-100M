from typing import Tuple

import jax
import jax.numpy as jnp

class Dataset:
    def __init__(self, text: str, context_window: int):
        self.id_char_map = list(set(text))
        self.char_id_map = {char: id for id, char in enumerate(self.id_char_map)}
        self.encoded_text = self.encode(text)
        self.context_window = context_window

    def encode(self, text: str) -> jax.Array:
        return jnp.array([self.char_id_map[char] for char in text])

    def decode(self, ids) -> str:
        return "".join([self.id_char_map[int(id)] for id in ids])

    @property
    def vocab(self):
        return self.id_char_map

    def get_item(self, index: int) -> Tuple[jax.Array, jax.Array]:
        inp = self.encoded_text[index:index+self.context_window]
        outp = self.encoded_text[index+1:index+1+self.context_window]
        return inp, outp

    def __len__(self):
        return len(self.encoded_text) - self.context_window
