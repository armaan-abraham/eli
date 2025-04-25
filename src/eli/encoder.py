import gc

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from torch.amp import GradScaler
from torch.nn import init
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, EncoderConfig
from eli.data import DataCollector
from eli.utils import print_gpu_memory_usage_fn

# Constants for prompts
PROMPT_PREFIX = """<|system|>
You are an expert at predicting what a language model will say next.
<|user|> Your task is to predict what another LLM will say, given the following
explanation of what the LLM is currently thinking: \" 
"""

PROMPT_SUFFIX = """\". Provide your prediction and nothing else.
<|assistant|>"""

PROMPT_PREFIX_CONTROL = """<|system|>
You are an expert at predicting what a language model will say next.
<|user|>
Your task is to predict what another LLM will say. Provide your prediction and nothing else.
<|assistant|>
"""


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
                torch.nn.Linear(cfg.target_model_act_dim, encoder_cfg.d_model)
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
    ) -> Float[Tensor, "batch tok d_out"]:
        """Forward pass for encoder.

        Args:
            x: Input tensor with shape [batch, d_in]

        Returns:
            Output tensor with shape [batch, token, d_out]
        """
        # Apply multiplex heads to create a sequence of tokens
        x_toks = torch.stack(
            [head(x) for head in self.multiplex_heads], dim=1
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
        target_generated_tokens: Float[Tensor, "batch tok"],
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
        virtual_embeddings = self.encoder(target_acts)

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
            :, -self.cfg.decoder_pred_len_toks :, :
        ]

        # Extract encoding logits (for regularization)
        prefix_len = fixed_token_lens["prefix_tokens_len"]
        decoder_logits_encoding_tokens = decoder_logits[
            :, prefix_len : (prefix_len + self.cfg.encoding_len_toks), :
        ]

        return (
            decoder_logits_target_tokens,
            decoder_logits_encoding_tokens,
            virtual_embeddings,
        )

    def assemble_decoder_context_embeddings(
        self,
        target_generated_tokens: Float[Tensor, "batch tok"],
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
        prefix_tokens = self.tokenizer(PROMPT_PREFIX, return_tensors="pt").input_ids.to(
            device
        )
        suffix_start_tokens = self.tokenizer(
            PROMPT_SUFFIX, return_tensors="pt"
        ).input_ids.to(device)

        # Repeat for batch size
        batch_size = target_generated_tokens.shape[0]
        prefix_tokens = prefix_tokens.repeat(batch_size, 1)
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


def kl_div(
    proposed_logits: Float[Tensor, "batch tok vocab"],
    target_logits: Float[Tensor, "batch tok vocab"],
):
    """Calculate KL divergence between two sets of logits.

    Args:
        proposed_logits: Logits from the proposed model
        target_logits: Logits from the target model

    Returns:
        KL divergence value
    """
    if proposed_logits.shape != target_logits.shape:
        raise ValueError(
            f"Proposed logits shape: {proposed_logits.shape}, Target logits shape: {target_logits.shape}"
        )

    if proposed_logits.ndim != 3:
        raise ValueError(f"Expected 3D logits, got shape {proposed_logits.shape}")

    proposed_logits = proposed_logits.float()
    target_logits = target_logits.float()

    # Add small epsilon to prevent numerical issues
    proposed_probs = torch.nn.functional.softmax(proposed_logits, dim=-1) + 1e-9
    target_probs = torch.nn.functional.softmax(target_logits, dim=-1) + 1e-9

    # Calculate KL divergence
    kl = torch.nn.functional.kl_div(
        torch.log(proposed_probs),
        target_probs,
        reduction="sum",
        log_target=False,
    ) / (proposed_logits.shape[0] * proposed_logits.shape[1])

    return kl


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

        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.encoder_decoder.parameters(),
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

    def loss(
        self,
        target_generated_tokens: Float[Tensor, "batch tok"],
        target_logits: Float[Tensor, "batch tok vocab"],
        target_acts: Float[Tensor, "batch tok d_model_target"],
        train_iter: int = -1,
    ):
        """Calculate loss for the encoder-decoder model.

        Args:
            target_generated_tokens: Tokens generated by target model
            target_logits: Logits from the target model
            target_acts: Target model activations
            train_iter: Current training iteration (for logging)

        Returns:
            Tuple of (total loss, target prediction loss, dinalar loss)
        """
        # Use autocast for mixed precision
        with torch.autocast(device_type=self.cfg.device.type, dtype=self.cfg.dtype):
            # Forward pass
            (
                decoder_logits_target_tokens,
                decoder_logits_encoding_tokens,
                virtual_embeddings,
            ) = self.encoder_decoder(target_acts, target_generated_tokens, train_iter)

            # Compute KL loss between decoder predictions and target generations
            target_prediction_loss = kl_div(decoder_logits_target_tokens, target_logits)

            # Initialize total loss with prediction loss
            loss = target_prediction_loss

            # Calculate Direct Natural Language Regularization (dinalar) if enabled
            decoder_probs_encoding_tokens = torch.nn.functional.softmax(
                decoder_logits_encoding_tokens, dim=-1
            )

            # Get token embeddings from decoder
            token_embeddings = get_embeddings_from_decoder(
                self.get_decoder()
            ).weight  # [vocab_size, d_embed]

            # Compute weighted sum of token embeddings by probability
            weighted_token_embeddings = einsum(
                decoder_probs_encoding_tokens,
                token_embeddings,
                "batch tok vocab, vocab d_embed -> batch tok d_embed",
            )

            # Compute MSE loss between virtual embeddings and weighted token embeddings
            dinalar_loss = torch.nn.functional.mse_loss(
                virtual_embeddings, weighted_token_embeddings, reduction="mean"
            )

            # Add dinalar loss if weight is positive
            if self.cfg.dinalar_weight > 0:
                loss += self.cfg.dinalar_weight * dinalar_loss

            return (
                loss,
                target_prediction_loss,
                dinalar_loss,
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
        target_logits = data["target_logits"]
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
            "logits_max": [],
            "logits_min": [],
        }

        # Training loop
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Extract batch data and move to device
            batch_tokens = target_generated_tokens[start_idx:end_idx].to(
                self.cfg.device
            )
            batch_logits = target_logits[start_idx:end_idx].to(self.cfg.device, dtype=torch.float32)
            batch_acts = target_acts[start_idx:end_idx].to(self.cfg.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Calculate loss
            loss, target_prediction_loss, dinalar_loss = self.loss(
                batch_tokens, batch_logits, batch_acts, train_iter
            )

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
            results["logits_max"].append(torch.max(batch_logits).item())
            results["logits_min"].append(torch.min(batch_logits).item())

            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Clean up memory
            del (
                batch_tokens,
                batch_logits,
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
        target_logits = data["target_logits"]

        # Evaluate on one batch
        tokens = target_generated_tokens[: self.cfg.control_batch_size_samples].to(
            self.cfg.device
        )
        logits = target_logits[: self.cfg.control_batch_size_samples].to(
            self.cfg.device, dtype=torch.float32
        )

        with torch.autocast(device_type=self.cfg.device.type, dtype=self.cfg.dtype):
            # Create input tokens with control prompt
            prefix_tokens = self.tokenizer(
                PROMPT_PREFIX_CONTROL, return_tensors="pt"
            ).input_ids.to(tokens.device)
            prefix_tokens = prefix_tokens.repeat(tokens.shape[0], 1)
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
                decoder_logits = self.get_decoder()(input_ids=input_tokens).logits[
                    :, -self.cfg.decoder_pred_len_toks :, :
                ]
                # Calculate KL loss
                loss = kl_div(decoder_logits, logits)

        return loss.item()

    def get_decoder(self):
        """Get the decoder model, handling DataParallel if present."""
        if isinstance(self.encoder_decoder, torch.nn.DataParallel):
            return self.encoder_decoder.module.decoder
        return self.encoder_decoder.decoder

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


def log_decoded_tokens(tokenizer, tokens, file_path, source_name):
    """Decode and log tokens to a file for debugging purposes.

    Args:
        tokenizer: The tokenizer to use for decoding
        tokens: The tokens to decode (first batch entry will be used)
        file_path: Path to the output file
        source_name: Name of the source function/method for context
    """
    try:
        decoded_tokens = tokenizer.decode(tokens[0])
        with open(file_path, "w") as f:
            f.write(f"=== Decoded Tokens from {source_name} ===\n")
            f.write(decoded_tokens)
            f.write("\n\n")
    except Exception as e:
        print(f"Error in log_decoded_tokens: {e}")
