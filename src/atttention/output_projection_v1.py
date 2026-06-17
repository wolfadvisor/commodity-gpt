import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.traceback import install

install()

X = torch.tensor([
    [1., 0., 2.],
    [1., 1., 2.],
    [2., 1., 1.],
    [0., 2., 1.]
])

K = X
Q = X
V = X

d_k = K.shape[-1]

# ==================================================
# HEAD 1
# ==================================================
scores1 = X @ X.T

weights1 = F.softmax(
    scores1 / torch.sqrt(torch.tensor(d_k)),
    dim= -1
)

head1 = weights1 @ X

# ==================================================
# HEAD 2
# ==================================================
scores2 = (X*2) @ (X*2).T

weights2 = F.softmax(
    scores2 / torch.sqrt(torch.tensor(d_k)),
    dim=-1
)

head2 = weights2 @ (X*2)

# ==================================================
# CONCATENATION
# ==================================================

multi_head = torch.cat(
    [head1,head2],
    dim=-1
)

print("\n[cyan]Multi Head Output[/cyan]")
print(multi_head)

print("Shape:", multi_head.shape)

# ==================================================
# OUTPUT PROJECTION
# ==================================================

projection = nn.Linear(
    in_features=6,
    out_features=3,
    bias=False
)

final_output = projection(
    multi_head
)

print("\n[green]Projected Output[/green]")
print(final_output)

print("Shape:", final_output.shape)

# ==================================================
# INSPECT Wo
# ==================================================

print("\n[yellow]Projection Matrix Wo[/yellow]")
print(projection.weight)

print("Shape:", projection.weight.shape)