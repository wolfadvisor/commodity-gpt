import torch
from torch.utils.data import DataLoader
from dataset.commodity_dataset import CommodityDataset
from model.create_vocabulary import CreateVocabulary
from model.model import CommodityGPT
from train import train_model
from generate import Generate_Text
from rich import print
from rich.traceback import install

install()


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
# DataSet
# ==================================

dataset = CommodityDataset(tokens=tokens.tolist(),
                           context_size=3)

# ==================================
# DataLoader
# ==================================

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# ==================================
# MODEL
# ==================================

model = CommodityGPT(
    vocab_size=vocab_size
)

# ==================================
# LOSS
# ==================================
trained_model = train_model(
    model=model,
    loader=loader,
    epochs=500,
    lr=0.001,
    print_every=50
)

# ==================================
# SAVE MODEL
# ==================================

torch.save(
    trained_model.state_dict(),
    "commodity_gpt_v1.pt"
)

print(
    "\nModel saved successfully!"
)

# ==================================
# MODEL Load
# ==================================

model.load_state_dict(
    torch.load("commodity_gpt_v1.pt")
)

model.eval()
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

# ==================================
# Text Generation
# ==================================

print()
print("="*50)
print("TEXT GENERATION")
print("="*50)

generated =  Generate_Text(
    model= trained_model,
    prompt= "soybean FOB",
    word_to_id= word_to_id,
    id_to_word= id_to_word,
    max_new_tokens=8
 )
print(f'\n{generated}\n')