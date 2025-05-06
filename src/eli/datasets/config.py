from dataclasses import dataclass

import torch
import transformer_lens

dtypes = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

CPU = torch.device("cpu")


@dataclass
class DatasetConfig:
    num_samples: int = int(1e3)
    s3_bucket: str = "eli-datasets"

    seed: int = 42

    use_fake_tokens: bool = (
        True  # Whether to use fake random tokens instead of real data
    )

    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    target_model_name: str = "EleutherAI/pythia-14m"
    vocab_size_target: int = 50304
    target_acts_collect_len_toks: int = 1
    target_ctx_len_toks: int = 8
    target_generation_len_toks: int = 1

    target_model_batch_size_samples: int = 16  # Per GPU

    desired_shard_size_bytes: int = 1024

    # Size of each atom returned by target data stream
    stream_atom_size_samples: int = 32

    target_model_act_dim: int = 128

    device: torch.device = CPU
    dtype: torch.dtype = dtypes["float32"]

    site: str = "resid_post"
    layer: int = 1

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


ds_cfg = DatasetConfig()
