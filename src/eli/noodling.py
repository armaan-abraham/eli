# %%

import unicodedata
from dataclasses import dataclass

import torch
import torch.nn.init as init
import transformer_lens
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    num_iter: int = 100

    seed: int = 42
    target_model_name: str = "EleutherAI/pythia-14m"
    decoder_model_name: str = "EleutherAI/pythia-14m"
    target_model_act_dim: int = 128
    decoder_model_embed_dim: int = 128
    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 1
    target_ctx_len_toks: int = 256
    decoder_pred_len_toks: int = 32
    encoding_len_toks: int = 16
    batch_size_samples: int = 128

    site: str = "resid_pre"
    layer: int = 1

    @property
    def sample_len_toks(self) -> int:
        return self.target_ctx_len_toks + self.decoder_pred_len_toks

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


cfg = Config()

# Load model twice for transformer lens and inference. Arrest me.
tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)
model = AutoModelForCausalLM.from_pretrained(cfg.target_model_name)
model_lens = transformer_lens.HookedTransformer.from_pretrained(cfg.target_model_name)

# %%

# Example usage
text = "Once upon a time,"
input = tokenizer(text, return_tensors="pt")["input_ids"]

num_generated_tokens = cfg.decoder_pred_len_toks - 1

print("Input shape:", input.shape)
# Generate (decoder_pred_len_toks - 1) tokens
generate_max_length = input.shape[1] + num_generated_tokens

# Force min_length to ensure we get at least the number of tokens we need
sequence_with_inference = model.generate(
    input,
    max_length=generate_max_length,
    min_length=generate_max_length,  # Force generating the full sequence
    do_sample=False,  # Use greedy decoding to ensure deterministic length
    pad_token_id=tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id,
)

# Now we can safely extract the last decoder_pred_len_toks tokens
generated_outputs = sequence_with_inference[:, -num_generated_tokens:]

print(
    "Generated text:",
    tokenizer.decode(sequence_with_inference[0], skip_special_tokens=False),
)

# Get logits for combined sequence
with torch.no_grad():
    logits = model(sequence_with_inference).logits

print("Logits shape:", logits.shape)

target_logits = logits[:, -cfg.decoder_pred_len_toks :, :]

print("Target logits shape:", target_logits.shape)

print(
    "Generated outputs:",
    tokenizer.decode(generated_outputs[0], skip_special_tokens=False),
)

# %%

# Get the logits for the last token
target_logits_first = target_logits[:, 0, :]

# Get the top predicted tokens
top_k = 5
top_k_values, top_k_indices = torch.topk(target_logits_first, top_k, dim=-1)

# Print the top predicted tokens
for i in range(top_k):
    token_id = top_k_indices[0, i].item()
    token = tokenizer.decode(token_id)
    probability = torch.softmax(target_logits_first, dim=-1)[0, token_id].item()
    print(f"Token: '{token}', Probability: {probability:.4f}")

# %%

_, cache = model_lens.run_with_cache(
    input,
    stop_at_layer=cfg.layer + 1,
    names_filter=cfg.act_name,
    return_cache_object=True,
)
acts = cache.cache_dict[cfg.act_name]

# %%

print("Acts shape:", acts.shape)

acts_last_token = acts[:, -1, :]

print("Acts last token shape:", acts_last_token.shape)

print("Result dtype:", target_logits.dtype)
print("Acts dtype:", acts.dtype)

# %%

result_bfloat = target_logits.to(torch.bfloat16)
acts_bfloat = acts.to(torch.bfloat16)

print("Result bfloat dtype:", result_bfloat.dtype)
print("Acts bfloat dtype:", acts_bfloat.dtype)

# %%


@dataclass
class EncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512


encoder_cfg = EncoderConfig()


class Attention(torch.nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.n_heads = cfg.n_heads

        self.W_Q = torch.nn.Parameter(
            torch.empty(self.n_heads, self.d_head, self.d_model)
        )
        self.b_Q = torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.W_K = torch.nn.Parameter(
            torch.empty(self.n_heads, self.d_head, self.d_model)
        )
        self.b_K = torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.W_V = torch.nn.Parameter(
            torch.empty(self.n_heads, self.d_head, self.d_model)
        )
        self.b_V = torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.W_O = torch.nn.Parameter(
            torch.empty(self.n_heads, self.d_model, self.d_head)
        )
        self.b_O = torch.nn.Parameter(torch.zeros(self.d_model))

        init.kaiming_normal_(self.W_Q)
        init.kaiming_normal_(self.W_K)
        init.kaiming_normal_(self.W_V)
        init.kaiming_normal_(self.W_O)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        Q = (
            einsum(
                x,
                self.W_Q,
                "batch tok d_model, n_heads d_head d_model -> batch tok n_heads d_head",
            )
            + self.b_Q
        )
        K = (
            einsum(
                x,
                self.W_K,
                "batch tok d_model, n_heads d_head d_model -> batch tok n_heads d_head",
            )
            + self.b_K
        )
        V = (
            einsum(
                x,
                self.W_V,
                "batch tok d_model, n_heads d_head d_model -> batch tok n_heads d_head",
            )
            + self.b_V
        )

        QK = einsum(
            Q,
            K,
            "batch tok_Q n_heads d_head, batch tok_K n_heads d_head -> batch n_heads tok_Q tok_K",
        )

        QK_scaled = QK / self.d_head**0.5

        # No causal mask here is intentional

        attn_weights = torch.nn.functional.softmax(QK_scaled, dim=-1)

        result_d_head = einsum(
            attn_weights,
            V,
            "batch n_heads tok_Q tok_K, batch tok_K n_heads d_head -> batch tok_Q n_heads d_head",
        )

        result_d_model = (
            einsum(
                result_d_head,
                self.W_O,
                "batch tok n_heads d_head, n_heads d_model d_head -> batch tok d_model",
            )
            + self.b_O
        )

        return result_d_model


class MLP(torch.nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.W_in = torch.nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_in = torch.nn.Parameter(torch.zeros(self.cfg.d_mlp))
        self.W_out = torch.nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_out = torch.nn.Parameter(torch.zeros(self.cfg.d_model))

        init.kaiming_normal_(
            self.W_in
        )  # Using 'gelu' as closest to the actual activation
        init.kaiming_normal_(self.W_out)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        acts = (
            einsum(x, self.W_in, "batch tok d_model, d_mlp d_model -> batch tok d_mlp")
            + self.b_in
        )
        acts = torch.nn.functional.gelu(acts)
        acts = (
            einsum(
                acts, self.W_out, "batch tok d_mlp, d_model d_mlp -> batch tok d_model"
            )
            + self.b_out
        )
        return acts


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()

        self.layernorm_1 = torch.nn.LayerNorm(cfg.d_model)
        self.attention = Attention(cfg)
        self.layernorm_2 = torch.nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        x_ln = self.layernorm_1(x)
        x_attn_resid = self.attention(x_ln) + x
        x_ln_2 = self.layernorm_2(x_attn_resid)
        x_mlp_resid = self.mlp(x_ln_2) + x_attn_resid
        return x_mlp_resid


class Encoder(torch.nn.Module):
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_cfg = encoder_cfg

        self.multiplex_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(cfg.target_model_act_dim, encoder_cfg.d_model)
                for _ in range(cfg.encoding_len_toks)
            ]
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(encoder_cfg) for _ in range(encoder_cfg.n_layers)]
        )
        self.output_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_cfg.d_model, cfg.decoder_model_embed_dim)
                for _ in range(cfg.encoding_len_toks)
            ]
        )

        for head in self.multiplex_heads:
            init.kaiming_normal_(head.weight)
            init.zeros_(head.bias)

        for head in self.output_heads:
            init.kaiming_normal_(head.weight)
            init.zeros_(head.bias)

    def forward(
        self, x: Float[Tensor, "batch d_in"]
    ) -> Float[Tensor, "batch tok d_out"]:
        x_toks = torch.stack(
            [head(x) for head in self.multiplex_heads], dim=1
        )  # [batch tok d_out]
        assert x_toks.shape == (
            x.shape[0],
            self.cfg.encoding_len_toks,
            self.encoder_cfg.d_model,
        )

        for block in self.transformer_blocks:
            x_toks = block(x_toks)

        # Apply each output head to its corresponding token
        x_out = torch.stack(
            [head(x_toks[:, i, :]) for i, head in enumerate(self.output_heads)], dim=1
        )
        assert x_out.shape == (
            x.shape[0],
            self.cfg.encoding_len_toks,
            self.cfg.decoder_model_embed_dim,
        )

        return x_out


# %%

encoder = Encoder(cfg, encoder_cfg)
encoder.to("mps")

virtual_embeddings = encoder(acts_last_token)

print("Virtual embeddings shape:", virtual_embeddings.shape)
print("Virtual embeddings device:", virtual_embeddings.device)


# %%
decoder = AutoModelForCausalLM.from_pretrained(cfg.decoder_model_name).to("mps")
decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name)

# %%

system_msg = "You are an expert at predicting what a language model will say next."
# Tokenize just the prefix
prefix_text = f"""<|system|>
{system_msg}
<|user|>
Your task is to predict what another LLM will say, given the following description of what the LLM is thinking: """
prefix_tokens = decoder_tokenizer(prefix_text, return_tensors="pt").input_ids

# Tokenize just the constant part of the suffix (without the generated text)
suffix_start_text = "<|assistant|>"
suffix_start_tokens = decoder_tokenizer(
    suffix_start_text, return_tensors="pt"
).input_ids

# Assert that tokenizers are the same (since we're using the same model for both)
# TODO: just directly compare the tokenizers here
assert type(tokenizer) == type(decoder_tokenizer)
assert (
    tokenizer.vocab_size == decoder_tokenizer.vocab_size
), "Tokenizers have different vocabulary sizes"
assert (
    tokenizer.get_vocab() == decoder_tokenizer.get_vocab()
), "Tokenizers have different vocabularies"

print("Prefix tokens shape:", prefix_tokens.shape)
print("Suffix start tokens shape:", suffix_start_tokens.shape)
print("Generated outputs shape:", generated_outputs.shape)

# Concatenate tokens directly (no need for batch dimension handling)
input_tokens = torch.cat([prefix_tokens, suffix_start_tokens, generated_outputs], dim=1)

# Convert to embeddings
word_embeddings = decoder.get_input_embeddings()
input_embeds = word_embeddings(input_tokens.to(virtual_embeddings.device))

print("Input embeds shape:", input_embeds.shape)
print("Virtual embeddings shape:", virtual_embeddings.shape)

# Add the virtual embeddings between prefix and suffix
combined_embeds = torch.cat(
    [
        input_embeds[:, : prefix_tokens.shape[1], :],
        virtual_embeddings,
        input_embeds[:, prefix_tokens.shape[1] :, :],
    ],
    dim=1,
)

# Get the second segment of input (after virtual embeddings)
second_segment_tokens = input_tokens[:, prefix_tokens.shape[1] :]

# Detokenize the second segment to see what it contains
second_segment_text = decoder_tokenizer.decode(
    second_segment_tokens[0], skip_special_tokens=False
)

print(f"Suffix text: {second_segment_text}")

logit_tokens = input_tokens[:, -cfg.decoder_pred_len_toks :]

# Detokenize the second segment to see what it contains
logit_tokens_text = decoder_tokenizer.decode(logit_tokens[0], skip_special_tokens=False)

# Print first character with its Unicode name
first_char = logit_tokens_text[0]
try:
    char_name = unicodedata.name(first_char)
except ValueError:
    # If character doesn't have a Unicode name, show its hex value
    char_name = f"(hex: {ord(first_char):04x})"

print(f"Logit tokens first token: '{first_char}' (Unicode: {char_name})")

# Print all characters with their names for better debugging
print("All characters in logit tokens:")
for i, char in enumerate(logit_tokens_text):
    try:
        char_name = unicodedata.name(char)
    except ValueError:
        char_name = f"(hex: {ord(char):04x})"
    print(f"  Character {i}: '{char}' (Unicode: {char_name})")

# Verify the combined embeddings shape is correct
expected_length = (
    prefix_tokens.shape[1]
    + virtual_embeddings.shape[1]
    + suffix_start_tokens.shape[1]
    + generated_outputs.shape[1]
)
assert (
    combined_embeds.shape[1] == expected_length
), f"Combined embeddings length mismatch: {combined_embeds.shape[1]} vs expected {expected_length}"

print("Combined embeds shape:", combined_embeds.shape)

# Run the model with the custom embeddings
with torch.no_grad():
    outputs = decoder(inputs_embeds=combined_embeds, return_dict=True)

decoder_logits = outputs.logits[:, -cfg.decoder_pred_len_toks :, :]

print("Decoder logits shape:", decoder_logits.shape)

# %%


def compute_kl_divergence(
    proposed_logits: Float[Tensor, "batch tok vocab"],
    target_logits: Float[Tensor, "batch tok vocab"],
):
    assert (
        proposed_logits.shape == target_logits.shape
    ), f"Proposed logits shape: {proposed_logits.shape}, Target logits shape: {target_logits.shape}"
    assert proposed_logits.ndim == 3

    proposed_log_probs = F.log_softmax(proposed_logits, dim=-1)
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    # Compute KL divergence with batchmean reduction
    kl_div = F.kl_div(
        proposed_log_probs,
        target_log_probs,
        reduction="sum",
        log_target=True,
    ) / (proposed_logits.shape[0] * proposed_logits.shape[1])

    return kl_div


# %%

print("Decoder logits device:", decoder_logits.device)
print("Target logits device:", target_logits.device)

target_logits = target_logits.to(decoder_logits.device)

kl_div = compute_kl_divergence(decoder_logits, target_logits)

print(f"KL divergence: {kl_div.item()}")
# %%
