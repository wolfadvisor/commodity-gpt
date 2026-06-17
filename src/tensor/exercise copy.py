from rich import print
from rich.traceback import install
import torch 

install()

#Create a randon Tensor 

X = torch.tensor([10,20,30], dtype=torch.float32)

print(X.mean())
print(X.std())

