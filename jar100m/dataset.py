from typing import Tuple
import torch

from jar100m.device import device

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text: str, context_window: int) -> None:
        self.id_char_map = sorted(list(set(text)))
        self.char_id_map = { char: id for id, char in enumerate(self.id_char_map) }

        self.encoded_text = self.encode(text)
        self.context_window = context_window

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_id_map[char] for char in text], device=device)

    def decode(self, ids) -> str:
        return "".join([self.id_char_map[id] for id in ids])

    @property
    def vocab(self):
        return self.id_char_map

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = self.encoded_text[index:index+self.context_window]
        outp = self.encoded_text[index+1:index+1+self.context_window]
        return inp, outp

    def __len__(self):
        return len(self.encoded_text) - self.context_window
