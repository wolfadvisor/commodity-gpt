import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.traceback import install

install()

# =====================================
# EMBEDDINGS tokens and dimensions
# =====================================

X = torch.tensor([
    [1., 0., 2.],
    [1., 1., 2.],
    [2., 1., 1.],
    [0., 2., 1.]
])

print("\n[green]Input Embeddings[/green]")
print(X)

# ==========================================
# ATTENTION
# ==========================================
d_k= X.shape[-1]
scores = X @ X.T

weights = F.softmax(
    scores / torch.sqrt(torch.tensor(d_k)), 
    dim=-1
)

attention_output = weights @ X

print("\n[cyan]Attention Output[/cyan]")
print(attention_output)

# ==========================================
# RESIDUAL CONNECTION
# ==========================================

residual_output = attention_output + X

print("\n[yellow]Residual Output[/yellow]")
print(residual_output)

print("\nShape:")
print(residual_output.shape)