import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, cfg


class DataCollectorEncoderContext:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        tokenizer_target = AutoTokenizer.from_pretrained(cfg.target_model_name)
        tokenizer_decoder = AutoTokenizer.from_pretrained(cfg.decoder_model_name)

        assert (
            type(tokenizer_target) == type(tokenizer_decoder)
        ), f"Tokenizer types must match, got {type(tokenizer_target)} and {type(tokenizer_decoder)}"

        self.tokenizer = tokenizer_target
