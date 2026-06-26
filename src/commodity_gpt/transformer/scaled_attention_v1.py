import torch
from torch.nn import functional as F
from rich import print
from rich.traceback import install

install()

"Partiremos dos valores (scores) para valores grandes onde o softmax não é a melhor pedida"


scores = torch.tensor([
    10.0,
    15.0,
    20.0
])

print(F.softmax(scores, dim=0))

# ==================================================
# EMBEDDINGS tokens and dimensions
# ==================================================

X = torch.tensor([
    [1., 0., 2.],  # soybean = tolken
    [1., 1., 2.],  # FOB
    [2., 1., 1.],  # price
    [0., 2., 1.]   # increased
])

# ==================================================
# Q K V
# ==================================================

Q = X
K = X
V = X

# ==================================================
# RAW SCORES
# ==================================================

scores = Q @ K.T

print("\n[bold yellow]Raw Scores[/bold yellow]")
print(scores)

# ==================================================
# SCALE (BALANCEAR OS DADOS)
# ==================================================

d_k = K.shape[-1]  #Last key dimensions

print(f"\nd_k = {d_k}")

scale = torch.sqrt(
    torch.tensor(
        d_k,
        dtype= torch.float32
    )
)

print(f"Scale = {scale}")

# ==================================================
# SCALED SCORES (Valores Balanceados)
# ==================================================

scaled_scores = scores / scale

print("\n[bold cyan]Scaled Scores[/bold cyan]")
print(scaled_scores)

# ==================================================
# SOFTMAX usando o scaled_scores assim fica mais facil
# ==================================================

weights = F.softmax(
    scaled_scores,
    dim=-1
)

print("\n[bold green]Attention Weights[/bold green]")
print(weights)

# ==================================================
# FINAL OUTPUT
# ==================================================

output= weights @ V

print("\n[bold magenta]Attention Output[/bold magenta]")
print(output)

print("\nShape:")
print(output.shape)
