from dataclasses import dataclass

import torch
import transformer_lens

CPU = torch.device("cpu")

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@dataclass
class Config:
    num_train_iter: int = 100

    seed: int = 42

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "EleutherAI/pythia-14m"
    decoder_model_name: str = "EleutherAI/pythia-14m"
    vocab_size: int = 50304
    target_ctx_len_toks: int = 4
    decoder_pred_len_toks: int = 2
    encoding_len_toks: int = 2

    train_batch_size_samples: int = 8192
    target_model_batch_size_samples: int = 32768

    buffer_size_samples: int = 131072

    target_model_act_dim: int = 128
    decoder_model_embed_dim: int = 128

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = dtypes["float16"]

    site: str = "resid_pre"
    layer: int = 5

    @property
    def target_generation_len_toks(self):
        return self.decoder_pred_len_toks - 1

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


cfg = Config()


@dataclass
class EncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16
    d_mlp: int = 256

    lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)


encoder_cfg = EncoderConfig()
