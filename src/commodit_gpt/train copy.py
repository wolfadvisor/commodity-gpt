import torch
import torch.nn as nn
from model import CommodityGPT

from rich.traceback import install
from rich import print

install()

# Tiny dataset

X = torch.tensor([
    [0,1,2],     # soybean FOB price
    [5,6,2],     # corn CIF price
])



y = torch.tensor([
    3,  #increased 
    7   #decresead
])

# using model gpt

model = CommodityGPT(
    vocab_size=13
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)

#Training Loop

for epoch in range(500):
    optimizer.zero_grad()
    logits = model(X)
   
    last_token_logits = logits[:,-1,:]
    loss = criterion(last_token_logits,y)

    loss.backward()
    optimizer.step()

    if epoch %50 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.4f}"
        )

print("\nTesting...\n")

with torch.no_grad():

    logits = model(X)

    predictions = torch.argmax(
        logits[:, -1, :],
        dim=1
    )

    print(predictions)