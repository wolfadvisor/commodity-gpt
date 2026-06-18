import torch

from vocabulary import CreateVocabulary
from model import CommodityGPT

# ==================================
# LOAD DATASET
# ==================================

with open(
    "data/commodities.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

# ==================================
# VOCABULARY
# ==================================

tokens, word_to_id, id_to_word = (
    CreateVocabulary(text)
)

vocab_size = len(word_to_id)

print("Vocabulary Size:", vocab_size)

# ==================================
# MODEL
# ==================================

model = CommodityGPT(
    vocab_size=vocab_size
)

# ==================================
# INPUT EXAMPLE
# ==================================

sample = torch.tensor([
    [
        word_to_id["soybean"],
        word_to_id["FOB"],
        word_to_id["price"]
    ]
])

print(f"Sample:{sample.shape}")

# ==================================
# FORWARD PASS
# ==================================

logits = model(sample)

print()

print("Logits Shape")
print(logits.shape)

# ==================================
# LAST TOKEN
# ==================================

last_token_logits = logits[0, -1]

prediction_id = torch.argmax(
    last_token_logits
).item()

prediction_word = id_to_word[
    prediction_id
]

print()

print(
    "Predicted Token:",
    prediction_word
)