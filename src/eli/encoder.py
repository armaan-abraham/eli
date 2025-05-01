import gc

import torch
from einops import einsum
from jaxtyping import Float, Int
from networkx import hits
from torch import Tensor
from torch.amp import GradScaler
from torch.nn import init
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, EncoderConfig
from eli.data import DataCollector
from eli.utils import calculate_gini, log_decoded_tokens, print_gpu_memory_usage_fn

PROMPT_DECODER = """## role:system
You are going to follow the instructions EXACTLY. Your task is simply to repeat
the given text. No commentary, no tags. Shown below are examples. You will be
given some text, labeled "GIVEN TEXT", and you will need to repeat it exactly.

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
    """Multi-head attention module with separate Q, K, V projections."""

    def __init__(self, cfg: EncoderConfig):
        """Initialize attention module.

        Args:
            cfg: Configuration parameters for the encoder
        """
        super().__init__()
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.n_heads = cfg.n_heads

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
        """Forward pass for attention.

        Args:
            x: Input tensor with shape [batch, token, d_model]

        Returns:
            Output tensor with shape [batch, token, d_model]
        """
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
    """Multi-layer perceptron with GELU activation."""

    def __init__(self, cfg: EncoderConfig):
        """Initialize MLP module.

        Args:
            cfg: Configuration parameters for the encoder
        """
        super().__init__()
        self.cfg = cfg

        # Initialize input and output projections
        self.W_in = torch.nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_in = torch.nn.Parameter(torch.zeros(self.cfg.d_mlp))
        self.W_out = torch.nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_out = torch.nn.Parameter(torch.zeros(self.cfg.d_model))

        # Initialize weights with Kaiming initialization
        init.kaiming_normal_(self.W_in)
        init.kaiming_normal_(self.W_out)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        """Forward pass for MLP.

        Args:
            x: Input tensor with shape [batch, token, d_model]

        Returns:
            Output tensor with shape [batch, token, d_model]
        """
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
    """Transformer block with pre-normalization."""

    def __init__(self, cfg: EncoderConfig):
        """Initialize transformer block.

        Args:
            cfg: Configuration parameters for the encoder
        """
        super().__init__()

        self.layernorm_1 = torch.nn.LayerNorm(cfg.d_model)
        self.attention = Attention(cfg)
        self.layernorm_2 = torch.nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(
        self, x: Float[Tensor, "batch tok d_model"]
    ) -> Float[Tensor, "batch tok d_model"]:
        """Forward pass for transformer block.

        Args:
            x: Input tensor with shape [batch, token, d_model]

        Returns:
            Output tensor with shape [batch, token, d_model]
        """
        # Apply attention with residual connection
        x_ln = self.layernorm_1(x)
        x_attn_resid = self.attention(x_ln) + x

        # Apply MLP with residual connection
        x_ln_2 = self.layernorm_2(x_attn_resid)
        x_mlp_resid = self.mlp(x_ln_2) + x_attn_resid

        return x_mlp_resid


class Encoder(torch.nn.Module):
    """Encoder that maps target model activations to decoder embeddings."""

    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig):
        """Initialize encoder.

        Args:
            cfg: Global configuration parameters
            encoder_cfg: Configuration parameters for the encoder
        """
        super().__init__()
        self.cfg = cfg
        self.encoder_cfg = encoder_cfg

        # Multiplexing heads convert input activations to separate token embeddings
        self.multiplex_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(cfg.target_model_agg_acts_dim, encoder_cfg.d_model)
                for _ in range(cfg.encoding_len_toks)
            ]
        )

        # Transformer blocks for processing the token sequence
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(encoder_cfg) for _ in range(encoder_cfg.n_layers)]
        )

        # Output heads convert transformer outputs to decoder embeddings
        self.output_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_cfg.d_model, cfg.decoder_model_embed_dim)
                for _ in range(cfg.encoding_len_toks)
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
        self, x: Float[Tensor, "batch d_in"]
    ) -> Float[Tensor, "batch tok d_embed"]:
        """Forward pass for encoder.

        Args:
            x: Input tensor with shape [batch, d_in]

        Returns:
            Output tensor with shape [batch, tok, d_embed]
        """
        # Apply multiplex heads to create a sequence of tokens
        x_toks = torch.stack(
            [head(x) + x for head in self.multiplex_heads], dim=1
        )  # [batch tok d_model]

        assert (
            x_toks.shape
            == (
                x.shape[0],
                self.cfg.encoding_len_toks,
                self.encoder_cfg.d_model,
            )
        ), f"Expected shape {(x.shape[0], self.cfg.encoding_len_toks, self.encoder_cfg.d_model)}, got {x_toks.shape}"

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
                self.cfg.encoding_len_toks,
                self.cfg.decoder_model_embed_dim,
            )
        ), f"Expected shape {(x.shape[0], self.cfg.encoding_len_toks, self.cfg.decoder_model_embed_dim)}, got {x_out.shape}"

        return x_out


def get_embeddings_from_decoder(decoder: torch.nn.Module):
    """Get embedding layer from decoder, handling DataParallel."""
    decoder_base = (
        decoder.module if isinstance(decoder, torch.nn.DataParallel) else decoder
    )
    return decoder_base.get_input_embeddings()


class EncoderDecoder(torch.nn.Module):
    """Combined encoder-decoder model that maps target activations to text."""

    def __init__(
        self, cfg: Config, encoder_cfg: EncoderConfig, tokenizer: AutoTokenizer
    ):
        """Initialize encoder-decoder.

        Args:
            cfg: Global configuration parameters
            encoder_cfg: Configuration parameters for the encoder
            tokenizer: Tokenizer for the decoder model
        """
        super().__init__()
        self.encoder = Encoder(cfg, encoder_cfg).to(device=CPU)
        self.decoder = AutoModelForCausalLM.from_pretrained(cfg.decoder_model_name).to(
            CPU
        )

        # Freeze the decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

    def forward(
        self,
        target_acts: Float[Tensor, "batch d_model"],
        target_generated_tokens: Int[Tensor, "batch tok"],
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
        # Generate virtual embeddings with the encoder
        # if encoder_output_logits is None:
        # encoder_output_logits = self.encoder(target_acts)  # [batch tok d_embed]
        # encoder_output_probs = torch.nn.functional.softmax(
        #     encoder_output_logits, dim=-1
        # )
        # embeddings = get_embeddings_from_decoder(self.decoder).weight  # [vocab d_embed]
        # virtual_embeddings = einsum(
        #     encoder_output_probs,
        #     embeddings,
        #     "batch tok vocab, vocab d_embed -> batch tok d_embed",
        # )
        virtual_embeddings_enc = self.encoder(target_acts)  # [batch tok d_embed]

        virtual_embeddings = target_acts[:, None, :]

        # Assemble input embeddings for the decoder
        decoder_context_embeddings, attention_mask, fixed_token_lens = (
            self.assemble_decoder_context_embeddings(
                target_generated_tokens, virtual_embeddings, train_iter
            )
        )

        # Run the decoder
        decoder_logits = self.decoder(
            inputs_embeds=decoder_context_embeddings, attention_mask=attention_mask
        ).logits

        # Extract target logits (for prediction loss)
        decoder_logits_target_tokens = decoder_logits[
            :, -self.cfg.decoder_pred_len_toks - 1 : -1, :
        ]

        # Extract encoding logits (for regularization)
        prefix_len = fixed_token_lens["prefix_tokens_len"]
        decoder_logits_encoding_tokens = decoder_logits[
            :, prefix_len - 1 : (prefix_len + self.cfg.encoding_len_toks) - 1, :
        ]

        return (
            decoder_logits_target_tokens,
            decoder_logits_encoding_tokens,
            virtual_embeddings_enc,
            # encoder_output_logits,
        )

    def assemble_decoder_context_embeddings(
        self,
        target_generated_tokens: Int[Tensor, "batch tok"],
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
        embeddings = get_embeddings_from_decoder(self.decoder)
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
        attention_mask = torch.ones(
            combined_embeds.shape[0],
            combined_embeds.shape[1],
            device=combined_embeds.device,
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


# def calculate_dinalar_loss(
#     decoder_logits_encoding_tokens: Float[Tensor, "batch tok vocab"],
#     encoder_output_logits: Float[Tensor, "batch tok vocab"],
# ) -> Float[Tensor, ""]:
#     """Calculate Direct Natural Language Regularization (dinalar) loss.

#     Args:
#         decoder_logits_encoding_tokens: Logits from the decoder for encoding tokens
#         encoder_output_logits: Logits from the encoder

#     Returns:
#         Dinalar loss
#     """
#     # Skip the first token
#     decoder_logits_encoding_tokens = decoder_logits_encoding_tokens[:, 1:, :]
#     encoder_output_logits = encoder_output_logits[:, 1:, :]

#     # Compute cross entropy loss between top prediction of decoder and decoder outputs
#     decoder_top_preds = torch.argmax(
#         decoder_logits_encoding_tokens, dim=-1
#     )  # [batch tok]
#     loss = torch.nn.functional.cross_entropy(
#         encoder_output_logits.permute(0, 2, 1),  # [batch vocab tok]
#         decoder_top_preds.long(),
#         reduction="mean",
#     )
#     return loss


def calculate_target_prediction_loss(
    decoder_logits_target_tokens: Float[Tensor, "batch tok vocab"],
    target_generated_tokens: Int[Tensor, "batch tok"],
) -> Float[Tensor, ""]:
    """Calculate target prediction loss.

    Args:
        decoder_logits_target_tokens: Logits from the decoder for target tokens
        target_generated_tokens: Tokens generated by target model
    """

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        decoder_logits_target_tokens.permute(0, 2, 1),
        target_generated_tokens.long(),
        reduction="mean",
    )

    return loss


class EncoderTrainer:
    """Trainer for the encoder-decoder model."""

    def __init__(
        self,
        cfg: Config,
        encoder_cfg: EncoderConfig,
    ):
        """Initialize trainer.

        Args:
            cfg: Global configuration parameters
            encoder_cfg: Configuration parameters for the encoder
        """
        self.cfg = cfg
        self.encoder_cfg = encoder_cfg

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name)
        self.encoder_decoder = EncoderDecoder(cfg, encoder_cfg, self.tokenizer)

        # Set up multi-GPU if available
        self.device_count = 1
        if cfg.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for encoder decoder")
            self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
            self.device_count = torch.cuda.device_count()

        # Move model to device
        self.encoder_decoder.to(self.cfg.device)

        for param in self.encoder_decoder.parameters():
            if param.requires_grad:
                print(param)

        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            [param for param in self.encoder_decoder.parameters() if param.requires_grad],
            lr=encoder_cfg.lr,
            betas=encoder_cfg.betas,
            weight_decay=encoder_cfg.weight_decay,
        )

        # Move model back to CPU to save memory
        self.encoder_decoder.to(CPU)

        # Set up gradient scaler for mixed precision
        self.scaler = GradScaler()

    def move_models_to_device(self, device: torch.device):
        """Move models to specified device."""
        self.encoder_decoder.to(device)

    @classmethod
    def loss(
        cls,
        cfg: Config,
        encoder_decoder: EncoderDecoder,
        target_generated_tokens: Int[Tensor, "batch tok"],
        target_acts: Float[Tensor, "batch tok d_model_target"],
        train_iter: int = -1,
    ):
        """Calculate loss for the encoder-decoder model.

        Args:
            target_generated_tokens: Tokens generated by target model
            target_acts: Target model activations
            train_iter: Current training iteration (for logging)

        Returns:
            Tuple of (total loss, target prediction loss, dinalar loss)
        """
        # Use autocast for mixed precision
        with torch.autocast(device_type=cfg.device.type, dtype=cfg.dtype):
            # Forward pass
            (
                decoder_logits_target_tokens,
                decoder_logits_encoding_tokens,
                virtual_embeddings,
                # encoder_output_logits,
            ) = encoder_decoder(
                target_acts, target_generated_tokens, train_iter=train_iter
            )

            # Compute KL loss between decoder predictions and target generations
            target_prediction_loss = calculate_target_prediction_loss(
                decoder_logits_target_tokens, target_generated_tokens
            )

            # Initialize total loss with prediction loss
            loss = 0
            loss += target_prediction_loss

            # Calculate Direct Natural Language Regularization (dinalar) if enabled
            # dinalar_loss = calculate_dinalar_loss(
            #     decoder_logits_encoding_tokens,
            #     encoder_output_logits,
            # )
            dinalar_loss = torch.tensor(1)

            # Add dinalar loss if weight is positive
            if cfg.dinalar_weight > 0:
                loss += cfg.dinalar_weight * dinalar_loss

            loss += (virtual_embeddings - torch.randn_like(virtual_embeddings)).pow(2).mean()

            return (
                loss,
                target_prediction_loss,
                dinalar_loss,
                # encoder_output_logits,
            )

    @print_gpu_memory_usage_fn
    def train(self, data_collector: DataCollector, train_iter: int = -1):
        """Train the encoder-decoder model for one epoch.

        Args:
            data_collector: Data collector with training data
            train_iter: Current training iteration

        Returns:
            Dictionary of training results
        """
        # Load all data
        data = data_collector.data

        target_generated_tokens = data["target_generated_tokens"]
        target_acts = data["target_acts"]

        # Calculate batch information
        buffer_size = target_acts.shape[0]
        batch_size = self.cfg.train_batch_size_samples * self.device_count
        num_batches = buffer_size // batch_size

        # Initialize results dictionary
        results = {
            "loss": [],
            "target_prediction_loss": [],
            "dinalar_loss": [],
            "grad_norm": [],
            "grad_abs_max": [],
            "grad_abs_min": [],
        }

        # Training loop
        print(f"Training for {num_batches} batches")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Extract batch data and move to device
            batch_tokens = target_generated_tokens[start_idx:end_idx].to(
                self.cfg.device
            )
            batch_acts = target_acts[start_idx:end_idx].to(self.cfg.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Calculate loss
            loss, target_prediction_loss, dinalar_loss = (
                self.loss(
                    self.cfg,
                    self.encoder_decoder,
                    batch_tokens,
                    batch_acts,
                    train_iter,
                )
            )

            # if batch_idx == num_batches - 1:
            #     encoder_output_logits_last = encoder_output_logits

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.encoder_decoder.parameters(), max_norm=0.5
            )

            # Get gradient statistics
            grad_stats = get_gradient_stats(self.encoder_decoder.parameters())

            # Save results for logging
            results["loss"].append(loss.item())
            results["target_prediction_loss"].append(target_prediction_loss.item())
            results["dinalar_loss"].append(dinalar_loss.item())
            results["grad_norm"].append(grad_stats["grad_norm"].item())
            results["grad_abs_max"].append(grad_stats["grad_abs_max"].item())
            results["grad_abs_min"].append(grad_stats["grad_abs_min"].item())

            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Clean up memory
            del (
                batch_tokens,
                batch_acts,
                loss,
                target_prediction_loss,
                dinalar_loss,
                grad_stats,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Zero gradients after training
        self.optimizer.zero_grad(set_to_none=True)

        # Verify all result lists have the same length
        lens = [len(results[key]) for key in results]
        if not all(l == lens[0] for l in lens):
            raise ValueError(
                f"All result lists should have the same length, got {lens}"
            )

        # Add per-buffer metrics
        # results["encoder_output_logits_gini"] = calculate_gini(
        #     encoder_output_logits_last
        # )

        return results

    def loss_control(self, data_collector: DataCollector, train_iter: int = -1):
        """Evaluate loss without using the encoder (control experiment).

        Args:
            data_collector: Data collector with evaluation data
            train_iter: Current training iteration

        Returns:
            Control loss value
        """
        data = data_collector.data

        target_generated_tokens = data["target_generated_tokens"]

        # Evaluate on one batch
        tokens = target_generated_tokens[: self.cfg.control_batch_size_samples].to(
            self.cfg.device
        )

        with torch.autocast(device_type=self.cfg.device.type, dtype=self.cfg.dtype):
            # Create input tokens with control prompt
            prefix_tokens = self.tokenizer(
                PROMPT_CONTROL, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(tokens.device)
            prefix_tokens = prefix_tokens.repeat(tokens.shape[0], 1)
            prefix_tokens = prepend_bos_token(prefix_tokens, self.tokenizer)
            input_tokens = torch.cat([prefix_tokens, tokens], dim=1)
            # Log decoded tokens for debugging (only on first iteration)
            if train_iter == 0:
                try:
                    log_decoded_tokens(
                        self.tokenizer,
                        input_tokens,
                        "decoded_tokens_control.txt",
                        "loss_control",
                    )
                except Exception as e:
                    print(f"Failed to log decoded tokens: {e}")

            with torch.no_grad():
                # Get decoder logits
                decoder_logits = self.get_decoder(self.encoder_decoder)(
                    input_ids=input_tokens
                ).logits[:, -self.cfg.decoder_pred_len_toks - 1 : -1, :]
                # Calculate KL loss
                loss = calculate_target_prediction_loss(decoder_logits, tokens)

        return loss.item()

    @classmethod
    def get_decoder(cls, encoder_decoder: EncoderDecoder):
        """Get the decoder model, handling DataParallel if present."""
        if isinstance(encoder_decoder, torch.nn.DataParallel):
            return encoder_decoder.module.decoder
        return encoder_decoder.decoder

    def save_encoder(self, save_path):
        """Save just the encoder part of the encoder-decoder model.

        Args:
            save_path: Path where the model should be saved
        """
        # Get the encoder, accounting for DataParallel
        if isinstance(self.encoder_decoder, torch.nn.DataParallel):
            encoder = self.encoder_decoder.module.encoder
        else:
            encoder = self.encoder_decoder.encoder

        # Move to CPU before saving
        encoder = encoder.to(CPU)

        # Save the model
        try:
            torch.save(encoder.state_dict(), save_path)
            print(f"Encoder saved to {save_path}")
        except Exception as e:
            print(f"Failed to save encoder: {e}")


def get_gradient_stats(parameters):
    """Calculate gradient statistics for the given parameters.

    Args:
        parameters: Iterator over parameters with gradients

    Returns:
        Dictionary containing gradient norm, absolute max, and absolute min
    """
    # Collect all gradients
    grads = [p.grad.detach() for p in parameters if p.grad is not None]

    if not grads:
        return {
            "grad_norm": torch.tensor(0.0),
            "grad_abs_max": torch.tensor(0.0),
            "grad_abs_min": torch.tensor(0.0),
        }

    # Flatten gradients
    flat_grads = torch.cat([g.flatten() for g in grads])

    # Calculate statistics
    grad_norm = torch.norm(flat_grads, p=2)
    grad_abs_max = torch.max(torch.abs(flat_grads))

    # Handle case where all gradients might be zero
    non_zero_grads = (
        flat_grads[flat_grads != 0] if torch.any(flat_grads != 0) else flat_grads
    )
    grad_abs_min = torch.min(torch.abs(non_zero_grads))

    return {
        "grad_norm": grad_norm,
        "grad_abs_max": grad_abs_max,
        "grad_abs_min": grad_abs_min,
    }
