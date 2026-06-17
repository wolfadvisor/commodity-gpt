"Creating Q,K and V with Pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from rich.traceback import install

install()

# =====================================================
# STEP 1
# INPUT EMBEDDINGS
# =====================================================

X = torch.tensor([
    [1., 0., 2.],  # soybean
    [1., 1., 2.],  # FOB
    [2., 1., 1.],  # price
    [0., 2., 1.]   # increased
])

print(f"\n[bold green]Original Embeddings (X)")
print(X)
print(f"Shape: {X.shape}")

# =====================================================
# STEP 2
# CREATE TRAINABLE MATRICES
# =====================================================

#Creates Wq internally
query_layer =nn.Linear(
    in_features= 3,
    out_features= 3,
    bias= False
)

#Creates Wk internally
key_layer = nn.Linear(
    in_features=3,
    out_features=3,
    bias=False
)

#Creates Wv internally
values_layer = nn.Linear(
    in_features=3,
    out_features=3,
    bias=False
)
# =====================================================
# STEP 3
# GENERATE Q K V
# =====================================================
Q = query_layer(X)

K = key_layer(X)

V = values_layer(X)

print("\n[bold cyan]Query Matrix (Q)[/bold cyan]")
print(Q)

print("\n[bold cyan]Key Matrix (K)[/bold cyan]")
print(K)

print("\n[bold cyan]Value Matrix (V)[/bold cyan]")
print(V)

# =====================================================
# STEP 4
# ATTENTION SCORES
# =====================================================

scores = Q @ K.T

print("\n[bold yellow]Attention Scores[/bold yellow]")
print(scores)

# =====================================================
# STEP 5
# SOFTMAX
# =====================================================

weights = F.softmax(
    scores,
    dim=-1
)

print("\n[bold magenta]Attention Weights[/bold magenta]")
print(f"{weights}\n")

# =====================================================
# STEP 6
# FINAL ATTENTION OUTPUT
# =====================================================

output = weights @ V

print("\n[bold red]Attention Output[/bold red]")
print(output)

print("\nOutput Shape:")
print(output.shape)

# =====================================================
# STEP 7
# ADDITIONAL INFORMATION
# =====================================================

