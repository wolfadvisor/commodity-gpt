from transfrom_block_v1 import TransformerBlock
import torch
from rich import print 
from rich.traceback import install

install()

# ==========================================
# TEST
# ==========================================

X = torch.randn(
    1,      # batch
    4,      # tokens
    3       # embedding dimension
)

print("\nInput Shape")
print(X.shape)

block = TransformerBlock(
    embedding_dim=3,
    hidden_dim=12
)

output = block(X)

print("\nOutput Shape")
print(output.shape)

print("\nOutput")
print(output)