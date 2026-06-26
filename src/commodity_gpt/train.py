import torch
import torch.nn as nn
from model.model import CommodityGPT

from rich.traceback import install
from rich import print

install()

# Tiny dataset

X = torch.tensor([
    [0,1,2],     # soybean FOB price
    [4,5,2],     # corn CIF price
])



y = torch.tensor([
    3,  #increased 
    6   #decresead
])

# using model gpt

model = CommodityGPT(
    vocab_size=7
)


import torch
import torch.nn as nn


def train_model(
    model,
    loader,
    epochs=100,
    lr=0.001,
    print_every=10
):
    """
    Trains CommodityGPT using batches from a DataLoader.

    Args:
        model: CommodityGPT model
        loader: PyTorch DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        print_every: Print loss every N epochs

    Returns:
        Trained model
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in loader:

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(X_batch)

            # Shape:
            # [batch_size, context_size, vocab_size]
            #
            # We only care about the prediction of the
            # last token position

            last_token_logits = logits[:, -1, :]

            # Compute loss
            loss = criterion(
                last_token_logits,
                y_batch
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % print_every == 0:

            print(
                f"Epoch {epoch:4d} "
                f"| Loss: {avg_loss:.4f}"
            )

    return model