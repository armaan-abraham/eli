import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from torch.nn import init
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, EncoderConfig
from eli.data import DataCollector
from eli.utils import log_gpu_memory_usage


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

        # No causal mask here is intentional; no autoregression.

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


def kl_div(
    proposed_logits: Float[Tensor, "batch tok vocab"],
    target_logits: Float[Tensor, "batch tok vocab"],
):
    assert (
        proposed_logits.shape == target_logits.shape
    ), f"Proposed logits shape: {proposed_logits.shape}, Target logits shape: {target_logits.shape}"
    assert proposed_logits.ndim == 3

    proposed_log_probs = torch.nn.functional.log_softmax(proposed_logits, dim=-1)
    target_log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)

    kl_div = torch.nn.functional.kl_div(
        proposed_log_probs,
        target_log_probs,
        reduction="sum",
        log_target=True,
    ) / (proposed_logits.shape[0] * proposed_logits.shape[1])

    return kl_div


class EncoderTrainer:
    def __init__(
        self,
        cfg: Config,
        encoder_cfg: EncoderConfig,
    ):
        self.cfg = cfg
        self.encoder_cfg = encoder_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name)
        self.encoder = Encoder(cfg, encoder_cfg).to(dtype=cfg.dtype, device=CPU)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            cfg.decoder_model_name, torch_dtype=cfg.dtype
        ).to(CPU)
        # We'll keep the optimizer's state on the cfg-specified device, so
        # transfer encoder to that device before initializing the optimizer.
        self.encoder.to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=encoder_cfg.lr,
            betas=encoder_cfg.betas,
            weight_decay=encoder_cfg.weight_decay,
        )
        self.encoder.to(CPU)

    def move_models_to_device(self, device: torch.device):
        self.encoder.to(device)
        self.decoder.to(device)

    def assemble_decoder_context_embeddings(
        self,
        target_generated_tokens: Float[Tensor, "batch tok"],
        virtual_embeddings: Float[Tensor, "batch tok d_embed"],
    ) -> Float[Tensor, "batch tok d_embed"]:
        # Generate tokens before virtual embeddings
        system_msg = (
            "You are an expert at predicting what a language model will say next."
        )
        prefix_text = f"""<|system|>
        {system_msg}
        <|user|>
        Your task is to predict what another LLM will say, given the following description of what the LLM is thinking. Provide your prediction and nothing else."""
        prefix_tokens = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(target_generated_tokens.device)

        # Generate tokens after virtual embeddings (excluding target model generation)
        suffix_start_text = "<|assistant|>"
        suffix_start_tokens = self.tokenizer(
            suffix_start_text, return_tensors="pt"
        ).input_ids.to(target_generated_tokens.device)

        prefix_tokens = prefix_tokens.repeat(target_generated_tokens.shape[0], 1)
        suffix_start_tokens = suffix_start_tokens.repeat(
            target_generated_tokens.shape[0], 1
        )

        input_tokens = torch.cat(
            [prefix_tokens, suffix_start_tokens, target_generated_tokens], dim=1
        )

        embeddings = self.decoder.get_input_embeddings()

        input_embeds = embeddings(input_tokens)

        assert (
            input_embeds.shape[0] == virtual_embeddings.shape[0]
        ), f"Input embeds shape: {input_embeds.shape}, Virtual embeddings shape: {virtual_embeddings.shape}"

        combined_embeds = torch.cat(
            [
                input_embeds[:, : prefix_tokens.shape[1], :],
                virtual_embeddings,
                input_embeds[:, prefix_tokens.shape[1] :, :],
            ],
            dim=1,
        )

        # Create attention mask (all 1s since you're not using padding)
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
        assert (
            combined_embeds.shape[1] == expected_length
        ), f"Combined embeddings length mismatch: {combined_embeds.shape[1]} vs expected {expected_length}"

        return combined_embeds, attention_mask

    def loss(
        self,
        target_generated_tokens: Float[Tensor, "batch tok"],
        target_logits: Float[Tensor, "batch tok vocab"],
        target_acts: Float[Tensor, "batch tok d_model_target"],
    ):
        virtual_embeddings = self.encoder(target_acts)

        decoder_context_embeddings, attention_mask = (
            self.assemble_decoder_context_embeddings(
                target_generated_tokens, virtual_embeddings
            )
        )

        decoder_logits = self.decoder(
            inputs_embeds=decoder_context_embeddings, attention_mask=attention_mask
        ).logits[:, -self.cfg.decoder_pred_len_toks :, :]

        return kl_div(decoder_logits, target_logits)

    @log_gpu_memory_usage
    def train(self, data_collector: DataCollector):
        # Load all data
        data = data_collector.data

        target_generated_tokens = data["target_generated_tokens"]
        target_logits = data["target_logits"]
        target_acts = data["target_acts"]

        buffer_size = target_acts.shape[0]
        batch_size = self.cfg.train_batch_size_samples
        num_batches = buffer_size // batch_size

        results = {
            "loss": [],
        }

        # Training loop
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Extract batch data
            batch_tokens = target_generated_tokens[start_idx:end_idx].to(
                self.cfg.device
            )
            batch_logits = target_logits[start_idx:end_idx].to(self.cfg.device)
            batch_acts = target_acts[start_idx:end_idx].to(self.cfg.device)

            self.optimizer.zero_grad()

            batch_loss = self.loss(batch_tokens, batch_logits, batch_acts)
            results["loss"].append(batch_loss.item())

            batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
            
            self.optimizer.step()

        return results

    def loss_control(self, data_collector: DataCollector):
        """Evaluates the loss of the decoder predictions without virtual
        embeddings produced by the encoder. This is to determine whether the
        encoder is actually useful to the decoder's predictions."""

        data = data_collector.data

        target_generated_tokens = data["target_generated_tokens"]
        target_logits = data["target_logits"]
        # No need to load activations as we are not using the encoder

        # Evaluate on one batch
        tokens = target_generated_tokens[: self.cfg.train_batch_size_samples].to(
            self.cfg.device
        )
        logits = target_logits[: self.cfg.train_batch_size_samples].to(self.cfg.device)

        system_msg = (
            "You are an expert at predicting what a language model will say next."
        )
        prefix_text = f"""<|system|>
        {system_msg}
        <|user|>
        Your task is to predict what another LLM will say. Provide your prediction and nothing else.
        <|assistant|>"""
        prefix_tokens = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(tokens.device)

        prefix_tokens = prefix_tokens.repeat(tokens.shape[0], 1)

        input_tokens = torch.cat([prefix_tokens, tokens], dim=1)

        with torch.no_grad():
            decoder_logits = self.decoder(input_ids=input_tokens).logits[
                :, -self.cfg.decoder_pred_len_toks :, :
            ]

            loss = kl_div(decoder_logits, logits)

        return loss.item()
