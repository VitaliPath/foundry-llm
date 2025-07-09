from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 0
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 64
    block_size: int = 128
