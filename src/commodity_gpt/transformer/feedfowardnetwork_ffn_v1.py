import torch
import torch.nn as nn

from rich import print
from rich.traceback import install

install()

# ======================================
# INPUT
# ======================================

X = torch.tensor([
    [1., 0., 2.],
    [1., 1., 2.],
    [2., 1., 1.],
    [0., 2., 1.]
])

print("\n[green]Input[/green]")
print(X)

print("Shape:", X.shape)

# ======================================
# FEED FORWARD NETWORK
# ======================================

ffn = nn.Sequential(
    nn.Linear(
        in_features=3,
        out_features=12
    ),

    nn.ReLU(),

    nn.Linear(
        in_features=12,
        out_features=3
    )
)

output= ffn(X)

print("\n[cyan]FFN Output[/cyan]")
print(output)

print("Shape:", output.shape)