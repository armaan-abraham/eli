# TODO: clean this up

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import transformer_lens

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@dataclass
class DatasetConfig:
    num_samples: int = int(1e6)
    s3_bucket: str = "eli"

    seed: int = 42

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "gpt2"
    vocab_size_target: int = 50257
    target_acts_collect_len_toks: int = 1
    target_ctx_len_toks: int = 64
    target_generation_len_toks: int = 1

    target_model_batch_size_samples: int = 64  # Per GPU

    # Size of each atom returned by target data stream
    target_data_stream_atom_size_samples: int = 32768

    target_model_act_dim: int = 768

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = dtypes["float16"]

    site: str = "resid_post"
    layer: int = 11

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def target_model_agg_acts_dim(self):
        return self.target_model_act_dim * self.target_acts_collect_len_toks


ds_cfg = DatasetConfig()
