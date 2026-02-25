from dataclasses import dataclass

@dataclass
class Config:
    input_dim: int = 28 * 28
    hidden_enc: int = 128
    hidden_dec: int = 128
    latent_dim: int = 64
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 200
    input_path: str = "./data"