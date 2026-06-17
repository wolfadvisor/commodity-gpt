import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.traceback import install

install()

# =====================================
# EMBEDDINGS
# =====================================

X = torch.tensor([
    [1., 0., 2.],
    [1., 1., 2.],
    [2., 1., 1.],
    [0., 2., 1.]
])

print("\n[green]Input Embeddings[/green]")
print(X)

# =====================================
# HEAD 1
# =====================================

Q1 = X
K1 = X
V1 = X

scores1 = Q1 @ K1.T
d_k =  K1.shape[-1]
weights1 = F.softmax(
    scores1 / torch.sqrt(torch.tensor(d_k)),
    dim=-1
)

head1 = weights1 @ V1

print("\n[cyan]Head 1 Output[/cyan]")
print(head1)

# =====================================
# HEAD 2 nos multiplicamos por dois 
# =====================================

Q2 = X * 2

K2 = X * 2

V2 = X * 2

scores2 = Q2 @ K2.T

weights2 = F.softmax(
    scores2 / torch.sqrt(torch.tensor(d_k)),
    dim=-1
)

head2 = weights2 @ V2

print("\n[yellow]Head 2 Output[/yellow]")
print(head2)

# =====================================
# CONCATENATION
# =====================================
"""
Head 1 learns: Relationship A - soybean

Head 2 learns: Relationship B - FOB 

Combining: A + B -> gives richer information.
"""
multi_head_output = torch.cat(
    [head1, head2],
    dim=-1
)

print("\n[magenta]Multi Head Output[/magenta]")
print(multi_head_output)

print("\nShape:")
print(multi_head_output.shape)
