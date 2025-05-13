import einops
import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import init
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.datasets.config import DatasetConfig
from eli.train.config import EncoderConfig, TrainConfig, encoder_cfg, train_cfg
from eli.train.utils import log_decoded_tokens

PROMPT_DECODER = """## role:system
You are going to follow the instructions EXACTLY. Your task is simply to REPEAT
THE GIVEN TEXT. No commentary, no tags. Shown below are examples. You will be
given some text, labeled "GIVEN TEXT", and you will need to repeat it exactly.
If the given text includes spaces or special characters, you will need to repeat
them exactly, including leading and trailing spaces.

## role:example
GIVEN TEXT:
Alice
ANSWER:
Alice

## role:example
GIVEN TEXT:
nucleus
ANSWER:
nucleus

## role:example
GIVEN TEXT:
 ..**<
ANSWER:
 ..**<

## role:example
GIVEN TEXT:
cular
ANSWER:
cular

## role:test
GIVEN TEXT:
<thought>
ANSWER:
"""

PROMPT_CONTROL = """## role:system
You are going to follow the instructions EXACTLY. Your task is simply to say a
random word. No commentary, no tags. Shown below are examples.

## role:example
ANSWER:
Alice

## role:example
ANSWER:
nucleus

## role:example
ANSWER:
the

## role:test
ANSWER:
"""


def prepend_bos_token(toks: Int[Tensor, "batch tok"], tokenizer: AutoTokenizer):
    bos = torch.ones_like(toks[:, :1])
    bos[:] = tokenizer.bos_token_id
    return torch.cat([bos, toks], dim=1)


class Attention(torch.nn.Module):
    def __init__(self, encoder_cfg: EncoderConfig):
        super().__init__()
        self.d_model = encoder_cfg.d_model
        self.d_head = encoder_cfg.d_head
        self.n_heads = encoder_cfg.n_heads

        # Initialize projection matrices and biases
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

        # Initialize weights with Kaiming initialization
        init.kaiming_normal_(self.W_Q)
        init.kaiming_normal_(self.W_K)
        init.kaiming_normal_(self.W_V)
        init.kaiming_normal_(self.W_O)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        # Project inputs to queries, keys, and values
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

        # Calculate attention scores
        QK = einsum(
            Q,
            K,
            "batch tok_Q n_heads d_head, batch tok_K n_heads d_head -> batch n_heads tok_Q tok_K",
        )

        # Scale attention scores
        QK_scaled = QK / self.d_head**0.5

        # No causal mask here is intentional; no autoregression.
        attn_weights = torch.nn.functional.softmax(QK_scaled, dim=-1)

        # Apply attention weights to values
        result_d_head = einsum(
            attn_weights,
            V,
            "batch n_heads tok_Q tok_K, batch tok_K n_heads d_head -> batch tok_Q n_heads d_head",
        )

        # Project back to model dimension
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
    def __init__(self, encoder_cfg: EncoderConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg

        # Initialize input and output projections
        self.W_in = torch.nn.Parameter(
            torch.empty(self.encoder_cfg.d_mlp, self.encoder_cfg.d_model)
        )
        self.b_in = torch.nn.Parameter(torch.zeros(self.encoder_cfg.d_mlp))
        self.W_out = torch.nn.Parameter(
            torch.empty(self.encoder_cfg.d_model, self.encoder_cfg.d_mlp)
        )
        self.b_out = torch.nn.Parameter(torch.zeros(self.encoder_cfg.d_model))

        # Initialize weights with Kaiming initialization
        init.kaiming_normal_(self.W_in)
        init.kaiming_normal_(self.W_out)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        # Project to hidden dimension
        acts = (
            einsum(x, self.W_in, "batch tok d_model, d_mlp d_model -> batch tok d_mlp")
            + self.b_in
        )
        # Apply GELU activation
        acts = torch.nn.functional.gelu(acts)
        # Project back to model dimension
        acts = (
            einsum(
                acts, self.W_out, "batch tok d_mlp, d_model d_mlp -> batch tok d_model"
            )
            + self.b_out
        )
        return acts


class TransformerBlock(torch.nn.Module):
    def __init__(self, encoder_cfg: EncoderConfig):
        super().__init__()

        self.layernorm_1 = torch.nn.LayerNorm(encoder_cfg.d_model)
        self.attention = Attention(encoder_cfg)
        self.layernorm_2 = torch.nn.LayerNorm(encoder_cfg.d_model)
        self.mlp = MLP(encoder_cfg)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        # Apply attention with residual connection
        x_ln = self.layernorm_1(x)
        x_attn_resid = self.attention(x_ln) + x

        # Apply MLP with residual connection
        x_ln_2 = self.layernorm_2(x_attn_resid)
        x_mlp_resid = self.mlp(x_ln_2) + x_attn_resid

        return x_mlp_resid


class Encoder(torch.nn.Module):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        train_cfg: TrainConfig,
        encoder_cfg: EncoderConfig,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.encoder_cfg = encoder_cfg
        self.dataset_cfg = dataset_cfg

        # Multiplexing heads convert input activations to separate token embeddings
        self.multiplex_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    dataset_cfg.target_model_act_dim
                    * dataset_cfg.target_acts_collect_len_toks,
                    encoder_cfg.d_model,
                )
                for _ in range(encoder_cfg.encoding_len_toks)
            ]
        )

        # Transformer blocks for processing the token sequence
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(encoder_cfg) for _ in range(encoder_cfg.n_layers)]
        )

        # Output heads convert transformer outputs to decoder embeddings
        self.output_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_cfg.d_model, train_cfg.decoder_model_embed_dim)
                for _ in range(encoder_cfg.encoding_len_toks)
            ]
        )

        # Initialize weights
        for head in self.multiplex_heads:
            init.kaiming_normal_(head.weight)
            init.zeros_(head.bias)

        for head in self.output_heads:
            init.kaiming_normal_(head.weight)
            init.zeros_(head.bias)

    def forward(
        self, x: Float[Tensor, "batch tok d_target_model"]
    ) -> Float[Tensor, "batch tok d_decoder_model"]:
        x = einops.rearrange(x, "batch tok d_model -> batch (tok d_model)")

        # Apply multiplex heads to create a sequence of tokens
        x_toks = torch.stack(
            [head(x) for head in self.multiplex_heads], dim=1
        )  # [batch tok d_model]

        assert (
            x_toks.shape
            == (
                x.shape[0],
                self.encoder_cfg.encoding_len_toks,
                self.encoder_cfg.d_model,
            )
        ), f"Expected shape {(x.shape[0], self.encoder_cfg.encoding_len_toks, self.encoder_cfg.d_model)}, got {x_toks.shape}"

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_toks = block(x_toks)

        # Apply output heads to each token
        x_out = torch.stack(
            [head(x_toks[:, i, :]) for i, head in enumerate(self.output_heads)], dim=1
        )

        assert (
            x_out.shape
            == (
                x.shape[0],
                self.encoder_cfg.encoding_len_toks,
                self.train_cfg.decoder_model_embed_dim,
            )
        ), f"Expected shape {(x.shape[0], self.encoder_cfg.encoding_len_toks, self.train_cfg.decoder_model_embed_dim)}, got {x_out.shape}"

        return x_out

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


class EncoderDecoder(torch.nn.Module):
    """Combined encoder-decoder model that maps target activations to text."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_cfg: DatasetConfig,
        encoder_cfg: EncoderConfig = encoder_cfg,
        train_cfg: TrainConfig = train_cfg,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.encoder_cfg = encoder_cfg

        # Initialize submodules
        self.encoder = Encoder(dataset_cfg, train_cfg, encoder_cfg)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            train_cfg.decoder_model_name
        )

        # Freeze the decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        target_acts: Float[Tensor, "batch tok d_model"],
        target_generated_tokens: Int[Tensor, "batch tok"],
        attention_mask: Int[Tensor, "batch tok"],
        # encoder_output_logits: Float[Tensor, "batch tok vocab"] | None = None,
        train_iter: int = -1,
    ):
        """Forward pass for encoder-decoder.

        Args:
            target_acts: Target model activations
            target_generated_tokens: Tokens generated by target model
            train_iter: Current training iteration (for logging)

        Returns:
            Tuple of (decoder logits for target tokens, decoder logits for encoding tokens, virtual embeddings)
        """
        assert (
            target_acts.shape[0]
            == target_generated_tokens.shape[0]
            == attention_mask.shape[0]
        )
        assert target_acts.ndim == 3
        assert target_generated_tokens.ndim == 2
        assert target_acts.shape[1] == self.dataset_cfg.target_acts_collect_len_toks
        assert target_acts.shape[2] == self.dataset_cfg.target_model_act_dim

        # Generate virtual embeddings with the encoder
        virtual_embeddings = self.encoder(target_acts)  # [batch tok d_embed]

        # Assemble input embeddings for the decoder
        decoder_context_embeddings, attention_mask, fixed_token_lens = (
            self.assemble_decoder_context_embeddings(
                target_generated_tokens, attention_mask, virtual_embeddings, train_iter
            )
        )

        # Run the decoder
        decoder_logits = self.decoder(
            inputs_embeds=decoder_context_embeddings, attention_mask=attention_mask
        ).logits

        # Extract target logits (for prediction loss)
        decoder_logits_target_tokens = decoder_logits[
            :, -target_generated_tokens.shape[1] - 1 : -1, :
        ]

        # Extract encoding logits (for regularization)
        prefix_len = fixed_token_lens["prefix_tokens_len"]
        decoder_logits_encoding_tokens = decoder_logits[
            :, prefix_len - 1 : (prefix_len + self.encoder_cfg.encoding_len_toks) - 1, :
        ]

        return (
            decoder_logits_target_tokens,
            decoder_logits_encoding_tokens,
            virtual_embeddings,
        )

    def assemble_decoder_context_embeddings(
        self,
        target_generated_tokens: Int[Tensor, "batch tok"],
        attention_mask: Int[Tensor, "batch tok"],
        virtual_embeddings: Float[Tensor, "batch tok d_embed"],
        train_iter: int = -1,
    ):
        """Assemble input embeddings for the decoder.

        Args:
            target_generated_tokens: Tokens generated by target model
            virtual_embeddings: Virtual embeddings generated by encoder
            train_iter: Current training iteration (for logging)

        Returns:
            Tuple of (combined embeddings, attention mask, token lengths dictionary)
        """
        device = target_generated_tokens.device

        # Generate tokens for prompt components
        prompt_prefix, prompt_suffix = PROMPT_DECODER.split("<thought>")
        prefix_tokens = self.tokenizer(
            prompt_prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        suffix_start_tokens = self.tokenizer(
            prompt_suffix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        # Repeat for batch size
        batch_size = target_generated_tokens.shape[0]
        prefix_tokens = prefix_tokens.repeat(batch_size, 1)
        prefix_tokens = prepend_bos_token(prefix_tokens, self.tokenizer)
        suffix_start_tokens = suffix_start_tokens.repeat(batch_size, 1)

        # Concatenate all tokens
        input_tokens = torch.cat(
            [prefix_tokens, suffix_start_tokens, target_generated_tokens], dim=1
        )

        # Log decoded tokens for debugging (only on first iteration)
        if train_iter == 0:
            try:
                log_decoded_tokens(
                    self.tokenizer,
                    input_tokens,
                    "decoded_tokens_encoder.txt",
                    "assemble_decoder_context_embeddings",
                )
            except Exception as e:
                print(f"Failed to log decoded tokens: {e}")

        # Get embeddings from decoder
        embeddings = self.decoder.get_input_embeddings()
        input_embeds = embeddings(input_tokens)

        # Verify shapes match
        if input_embeds.shape[0] != virtual_embeddings.shape[0]:
            raise ValueError(
                f"Input embeds shape: {input_embeds.shape}, Virtual embeddings shape: {virtual_embeddings.shape}"
            )

        # Combine embeddings: prefix + virtual + suffix + target
        combined_embeds = torch.cat(
            [
                input_embeds[:, : prefix_tokens.shape[1], :],
                virtual_embeddings,
                input_embeds[:, prefix_tokens.shape[1] :, :],
            ],
            dim=1,
        )

        # Create attention mask (all 1s since we're not using padding)
        attention_mask = torch.cat(
            (
                torch.ones(
                    combined_embeds.shape[0],
                    (
                        prefix_tokens.shape[1]
                        + virtual_embeddings.shape[1]
                        + suffix_start_tokens.shape[1]
                    ),
                    device=combined_embeds.device,
                ),
                attention_mask,
            ),
            dim=1,
        )

        # Verify the combined embeddings shape is correct
        expected_length = (
            prefix_tokens.shape[1]
            + virtual_embeddings.shape[1]
            + suffix_start_tokens.shape[1]
            + target_generated_tokens.shape[1]
        )

        if combined_embeds.shape[1] != expected_length:
            raise ValueError(
                f"Combined embeddings length mismatch: {combined_embeds.shape[1]} vs expected {expected_length}"
            )

        return (
            combined_embeds,
            attention_mask,
            {
                "prefix_tokens_len": prefix_tokens.shape[1],
                "suffix_start_tokens_len": suffix_start_tokens.shape[1],
            },
        )


def get_target_prediction_loss(
    decoder_logits_target_tokens: Float[Tensor, "batch tok vocab"],
    target_generated_tokens: Int[Tensor, "batch tok"],
    tokenizer: AutoTokenizer,
) -> Float[Tensor, ""]:
    """Calculate target prediction loss.

    Args:
        decoder_logits_target_tokens: Logits from the decoder for target tokens
        target_generated_tokens: Tokens generated by target model
    """

    print(f"decoder logits target tokens device: {decoder_logits_target_tokens.device}")
    print(f"target generated tokens device: {target_generated_tokens.device}")

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        decoder_logits_target_tokens.permute(0, 2, 1),
        target_generated_tokens.long(),
        reduction="mean",
        ignore_index=tokenizer.pad_token_id,
    )

    return loss


def get_loss(
    train_cfg: TrainConfig,
    device: torch.device,
    encoder_decoder: EncoderDecoder,
    target_generated_tokens: Int[Tensor, "batch tok"],
    attention_mask: Int[Tensor, "batch tok"],
    target_acts: Float[Tensor, "batch tok d_model_target"],
    tokenizer: AutoTokenizer,
    train_iter: int = -1,
):
    """Calculate loss for the encoder-decoder model.
    Tuple
        Args:
            target_generated_tokens: Tokens generated by target model
            target_acts: Target model activations
            train_iter: Current training iteration (for logging)

        Returns:
            Tuple of (total loss, target prediction loss, dinalar loss)
    """
    # Use autocast for mixed precision
    with torch.autocast(device_type=device.type, dtype=train_cfg.dtype):
        # Forward pass
        (
            decoder_logits_target_tokens,
            decoder_logits_encoding_tokens,
            virtual_embeddings,
        ) = encoder_decoder(
            target_acts,
            target_generated_tokens,
            attention_mask,
            train_iter=train_iter,
        )

        # Compute KL loss between decoder predictions and target generations
        target_prediction_loss = get_target_prediction_loss(
            decoder_logits_target_tokens, target_generated_tokens, tokenizer
        )

        return target_prediction_loss


def get_loss_control(
    train_cfg: TrainConfig,
    device: torch.device,
    encoder_decoder: EncoderDecoder,
    target_generated_tokens: Int[Tensor, "batch tok"],
    attention_mask: Int[Tensor, "batch tok"],
    tokenizer: AutoTokenizer,
    train_iter: int = -1,
):
    """Evaluate loss without using the encoder."""

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=train_cfg.dtype):
            # Create input tokens with control prompt
            prefix_tokens = tokenizer(
                PROMPT_CONTROL, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            prefix_tokens = prefix_tokens.repeat(target_generated_tokens.shape[0], 1)
            prefix_tokens = prepend_bos_token(prefix_tokens, tokenizer)
            input_tokens = torch.cat([prefix_tokens, target_generated_tokens], dim=1)

            # Log decoded tokens for debugging (only on first iteration)
            if train_iter == 0:
                try:
                    log_decoded_tokens(
                        tokenizer,
                        input_tokens,
                        "decoded_tokens_control.txt",
                        "loss_control",
                    )
                except Exception as e:
                    print(f"Failed to log decoded tokens: {e}")

            # Get decoder logits and compute loss
            decoder_logits = encoder_decoder.module.decoder(
                input_ids=input_tokens,
                attention_mask=attention_mask,
            ).logits[:, -target_generated_tokens.shape[1] - 1 : -1, :]

            loss = get_target_prediction_loss(
                decoder_logits, target_generated_tokens, tokenizer
            )

    return loss
