import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.traceback import install

install()

# ==========================================
# INPUT EMBEDDINGS
# ==========================================

X = torch.tensor([
    [1., 0., 2.],
    [1., 1., 2.],
    [2., 1., 1.],
    [0., 2., 1.]
])

print("\n[green]Original Input[/green]")
print(X)

# ==========================================
# ATTENTION
# ==========================================

scores = X @ X.T

weights = F.softmax(
    scores / torch.sqrt(torch.tensor(3.0)),
    dim=-1
)

attention_output = weights @ X

print("\n[cyan]Attention Output[/cyan]")
print(attention_output)

# ==========================================
# RESIDUAL
# ==========================================

residual_output = attention_output + X

print("\n[yellow]Residual Output[/yellow]")
print(residual_output)

# ==========================================
# LAYER NORMALIZATION
# ==========================================

layer_norm = nn.LayerNorm(
    normalized_shape=3
)


normalized_output = layer_norm(residual_output)

print("\n[magenta]Normalized Output[/magenta]")
print(normalized_output)

print("\nShape:")
print(normalized_output.shape)