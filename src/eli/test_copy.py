# %%

import torch
import transformer_lens
from einops import einsum
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.encoder import Encoder, EncoderDecoder, calculate_target_prediction_loss

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

from eli.config import cfg, encoder_cfg

encoder_decoder = EncoderDecoder(cfg, encoder_cfg, tokenizer).to("cuda")

# %%


def eval(tok):
    tok = torch.tensor([[tok]])
    tok = tok.to("cuda")
    # target_generated_tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    # assert target_generated_tokens.shape == (1, 1)

    # print(target_generated_tokens)
    # print(target_generated_tokens.shape)
    # print(tokenizer.decode(target_generated_tokens[0]))

    # embedding_text = text
    # embedding_tokens = tokenizer(embedding_text, add_special_tokens=False, return_tensors="pt").input_ids
    embeddings = encoder_decoder.decoder.get_input_embeddings()
    virtual_embeddings = embeddings(tok)
    # print(virtual_embeddings.shape)
    attention_mask = torch.ones_like(tok)

    decoder_context_embeddings, attention_mask, fixed_token_lens = (
        encoder_decoder.assemble_decoder_context_embeddings(
            tok,
            attention_mask,
            virtual_embeddings,
        )
    )

    decoder_logits = encoder_decoder.decoder(
        inputs_embeds=decoder_context_embeddings.to("cuda"),
        attention_mask=attention_mask.to("cuda"),
    ).logits

    decoder_logits_target_tokens = decoder_logits[:, -2:-1]

    loss = calculate_target_prediction_loss(decoder_logits_target_tokens, tok, tokenizer)

    return loss


toks = torch.randint(0, cfg.vocab_size_decoder, (500,))

# losses = [eval(tokenizer.decode(tok)) for tok in toks if not tokenizer.decode(tok).startswith(" ")]
losses = [eval(tok) for tok in toks if not tokenizer.decode(tok).startswith(" ")]

print(sum(losses) / len(losses))

# %%

for tok, loss in zip(toks, losses):
    print(tokenizer.decode(tok), round(loss.item(), 2))

# %%
# Get the logits for the target token "Bob"
bob_token_id = target_generated_tokens[0, 0].item()
bob_logit = decoder_logits_target_tokens[0, 0, bob_token_id].item()
print(
    f"Token ID: {bob_token_id}, Token: '{tokenizer.decode([bob_token_id])}', Logit: {bob_logit}"
)

# Get the top 3 token indices and their corresponding logits
k = 10
top_values, top_indices = torch.topk(decoder_logits_target_tokens[0, 0], k)

# Print the top 3 tokens and their logits
print(f"\nTop {k} tokens by logit:")
for i, (token_id, logit_value) in enumerate(zip(top_indices, top_values)):
    token_text = tokenizer.decode([token_id.item()])
    print(
        f"  {i+1}. Token ID: {token_id.item()}, Token: '{token_text}', Logit: {logit_value.item():.4f}"
    )

# %%


prompt = """## role:system
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
Bob
ANSWER:
"""

tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), tokens], dim=1)
embeddings_control = encoder_decoder.decoder.get_input_embeddings()(tokens)

print(embeddings_control.shape)
print(decoder_context_embeddings.shape)

# %%

model_lens = transformer_lens.HookedTransformer.from_pretrained("EleutherAI/pythia-70m")

output, cache = model_lens.run_with_cache(tokens, return_cache_object=True)


print(cache["normalized"])


acts = cache["normalized"][:, -1, :]

print(acts.shape)

logits = (
    einsum(acts, model_lens.W_U, "batch d_model, d_model vocab -> batch vocab")
    + model_lens.b_U
)

print(logits.shape)

print(torch.allclose(logits, output[:, -1, :]))

print(logits[0, :5])

print(output[0, -1, :5])


# %%
