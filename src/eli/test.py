# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# %%
    
def print_tokens(tokens):
    tokens = tokens[0]
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id.item()])
        print(f"Token {i+1}: ID = {token_id.item()}, Text = '{token_text}'")

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
print_tokens(tokens)
tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), tokens], dim=1)
print_tokens(tokens)

# Get model outputs
outputs = model(tokens)
logits = outputs.logits

# Get the logits for the last token
last_token_logits = logits[0, -1, :]
print(last_token_logits.shape)

# Get the top 3 token indices and their corresponding logits
k = 10
top_values, top_indices = torch.topk(last_token_logits, k)

# Print the top 3 tokens and their logits
print(f"Top {k} tokens by logit:")
for i, (token_id, logit_value) in enumerate(zip(top_indices, top_values)):
    token_text = tokenizer.decode([token_id.item()])
    print(f"{i+1}. Token: '{token_text}', ID: {token_id.item()}, Logit: {logit_value.item():.4f}")