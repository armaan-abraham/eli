from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import transformer_lens

CPU = torch.device("cpu")

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

SAVE_DIR = Path(__file__).parent / "saved_models"


@dataclass
class Config:
    num_train_samples: int = int(1e6)

    seed: int = 42

    use_fake_tokens: bool = True

    # WandB configuration
    wandb_enabled: bool = False
    wandb_project: str = "eli"

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "EleutherAI/pythia-14m"
    decoder_model_name: str = "EleutherAI/pythia-14m"
    vocab_size: int = 50304
    target_ctx_len_toks: int = 4
    decoder_pred_len_toks: int = 2
    encoding_len_toks: int = 2

    train_batch_size_samples: int = 2048  # Per GPU
    control_batch_size_samples: int = 2048  # Per GPU
    target_model_batch_size_samples: int = 8192  # Per GPU

    buffer_size_samples: int = 65536

    target_model_act_dim: int = 128
    decoder_model_embed_dim: int = 128

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = dtypes["float16"]

    site: str = "resid_pre"
    layer: int = 5

    dinalar_weight: float = 0

    save_encoder_path: Optional[Path] = None

    @property
    def target_generation_len_toks(self):
        return self.decoder_pred_len_toks - 1

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def num_train_iter(self):
        return (
            self.num_train_samples + self.buffer_size_samples - 1
        ) // self.buffer_size_samples


cfg = Config()


@dataclass
class EncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16
    d_mlp: int = 256

    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)


encoder_cfg = EncoderConfig()
