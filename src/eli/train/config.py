from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

SAVE_DIR = Path(__file__).parent / "saved_models"


@dataclass
class TrainConfig:
    s3_bucket: str = "eli-datasets"
    dataset_name: str = "EleutherAI-pythia-70m-resid_post-4"

    num_samples: int = int(2e7)

    seed: int = 42

    wandb_enabled: bool = True

    dataset_loader_batch_size_samples: int = 256
    dataset_loader_shuffle_buffer_size_wds_entries: int = 10

    decoder_model_name: str = "gpt2"
    decoder_model_embed_dim: int = 768

    _dtype: torch.dtype = torch.float16  # For autocast

    save_encoder_path: Optional[Path] = SAVE_DIR / "encoder.pt"
    save_encoder_to_s3: bool = False

    log_loss_control_every_n_iter: int = 10

    @property
    def dtype(self) -> torch.dtype:
        if torch.cuda.is_available():
            return self._dtype
        else:
            return torch.float32


train_cfg = TrainConfig()


@dataclass
class EncoderConfig:
    encoding_len_toks: int = 8

    n_layers: int = 6
    n_heads: int = 16
    d_model: int = 1536
    d_head: int = 128
    d_mlp: int = 6144

    lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)


encoder_cfg = EncoderConfig()
