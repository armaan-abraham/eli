from dataclasses import dataclass, field

import torch
import transformer_lens

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@dataclass
class DatasetConfig:
    num_samples: int = int(1e8)
    s3_bucket: str = "eli-datasets"

    seed: int = 42

    use_fake_tokens: bool = False

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "EleutherAI/pythia-70m"
    vocab_size_target: int = 50304
    target_ctx_len_toks: int = 64
    target_generation_len_toks: int = 32

    target_model_batch_size_samples: int = 2048  # Per device

    max_shard_size_bytes: int = 1024**3

    # Size of each atom returned by target data stream
    dataset_entry_size_samples: int = 8192

    target_model_act_dim: int = 512

    _device_str: str = "cuda"
    _dtype_str: str = "float16"

    _act_storage_dtype_str: str = "float16"

    target_acts_collect_len_toks: int = 16
    site: str = "resid_post"
    target_acts_layer_range: list[int, int] = field(
        default_factory=lambda: [4, 5]
    )  # [start, end] inclusive

    @property
    def device(self) -> torch.device:
        return torch.device(self._device_str)

    @property
    def dtype(self) -> torch.dtype:
        return dtypes[self._dtype_str]

    @property
    def act_storage_dtype(self) -> torch.dtype:
        return dtypes[self._act_storage_dtype_str]

    @property
    def num_act_layers(self) -> int:
        return self.target_acts_layer_range[1] - self.target_acts_layer_range[0] + 1


ds_cfg = DatasetConfig()
