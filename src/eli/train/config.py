from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

SAVE_DIR = Path(__file__).parent / "saved_models"


@dataclass
class TrainConfig:
    s3_bucket: str = "eli-datasets"
    dataset_name: str = "EleutherAI-pythia-70m-resid_post-4-5-100000000"

    num_samples: int = int(2e7)

    seed: int = 42

    wandb_enabled: bool = True

    dataset_loader_batch_size_samples: int = 256
    loss_control_batch_size_dataset_loader_batch_size_frac: float = 1.0
    dataset_loader_shuffle_buffer_size_wds_entries: int = 2

    decoder_model_name: str = "EleutherAI/pythia-70m"
    decoder_model_embed_dim: int = 512

    # Autocast
    _dtype_decoder: torch.dtype = torch.float16
    _dtype_encoder: torch.dtype = torch.bfloat16

    save_encoder_path: Optional[Path] = SAVE_DIR / "encoder.pt"
    save_encoder_to_s3: bool = True

    log_loss_control_every_n_iter: int = 20
    log_virtual_embeddings_stats_every_n_iter: int = 10

    grad_clip_norm: float = 0.5

    @property
    def dtype_decoder(self) -> torch.dtype:
        if torch.cuda.is_available():
            return self._dtype_decoder
        else:
            return torch.float32

    @property
    def dtype_encoder(self) -> torch.dtype:
        if torch.cuda.is_available():
            return self._dtype_encoder
        else:
            return torch.float32


train_cfg = TrainConfig()


@dataclass
class EncoderConfig:
    encoding_len_toks: int = 8

    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 768
    d_mlp_factor: int = 4

    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)

    @property
    def d_mlp(self) -> int:
        assert hasattr(self, "d_model"), "d_model must be set"
        return int(self.d_model * self.d_mlp_factor)

    @property
    def d_head(self) -> int:
        assert hasattr(self, "d_model"), "d_model must be set"
        return self.d_model


encoder_cfg = EncoderConfig()
