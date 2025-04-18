from dataclasses import dataclass

import torch
import transformer_lens

CPU = torch.device("cpu")

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class Config:
    num_train_iter: int = 2

    seed: int = 42

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 2  # TODO: increase

    target_model_name: str = "EleutherAI/pythia-14m"
    decoder_model_name: str = "EleutherAI/pythia-14m"
    vocab_size: int = 50304
    target_ctx_len_toks: int = 4
    decoder_pred_len_toks: int = 2
    encoding_len_toks: int = 2

    train_batch_size_samples: int = 2
    target_model_batch_size_samples: int = 2

    buffer_size_samples: int = 16

    target_model_act_dim: int = 128
    decoder_model_embed_dim: int = 128

    device: torch.device = CPU
    dtype: torch.dtype = dtypes["bfloat16"]

    site: str = "resid_pre"
    layer: int = 1

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
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512

    lr: float = 1e-4
    weight_decay: float = 1e-2


encoder_cfg = EncoderConfig()
