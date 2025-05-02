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
    num_train_samples: int = int(2e7)

    seed: int = 42

    use_fake_tokens: bool = False
    use_data_collector_workers: bool = True

    # WandB configuration
    wandb_enabled: bool = True
    wandb_project: str = "eli"

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "gpt2"
    decoder_model_name: str = "gpt2"
    vocab_size_target: int = 50257
    vocab_size_decoder: int = 50257
    target_acts_collect_len_toks: int = 1
    target_ctx_len_toks: int = 64
    decoder_pred_len_toks: int = 1
    encoding_len_toks: int = 1

    train_batch_size_samples: int = 256  # Per GPU
    control_batch_size_samples: int = 256  # Per GPU
    target_model_batch_size_samples: int = 1024  # Per GPU

    buffer_size_samples: int = 32768

    target_model_act_dim: int = 768
    decoder_model_embed_dim: int = 768

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = dtypes["float16"]

    site: str = "resid_post"
    layer: int = 11

    dinalar_weight: float = 0

    save_encoder_path: Optional[Path] = SAVE_DIR / "encoder.pt"

    @property
    def target_generation_len_toks(self):
        return self.decoder_pred_len_toks

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def num_train_iter(self):
        return (
            self.num_train_samples + self.buffer_size_samples - 1
        ) // self.buffer_size_samples

    @property
    def target_model_agg_acts_dim(self):
        return self.target_model_act_dim * self.target_acts_collect_len_toks


cfg = Config()


@dataclass
class EncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 512
    d_head: int = 64
    d_mlp: int = 2048

    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)


encoder_cfg = EncoderConfig()
