"""Linear Algebra para multiplicação de matrizes
    onde os angulos dos vetores irãodeterminar a direção e 
    logo se existe similariedade entre os fatos...
"""

import torch
from torch.nn import functional as F
from commodity_gpt.embeddings.embedding_v1 import similarity
from rich import print
from rich.traceback import install

install()

soybean = torch.tensor([
    1.0,2.0,3.0
])

corn = torch.tensor([
    4.0,5.0,6.0
])

sugar = torch.tensor([
    -7.0,-8.0,-9.0
])

result = torch.dot(soybean,corn)
print(result)
result = torch.dot(soybean,sugar)
print(result)

x = similarity(soybean,sugar)

print(x)

#Fazendo matriz com pytorch

matriz_tensor = torch.stack([soybean,corn,sugar],dim= 0)
print(matriz_tensor)

scores = matriz_tensor @ matriz_tensor.T
print(scores)

commodity_embeddings = torch.tensor([
    [1., 0., 2.],  # soybean
    [1., 1., 2.],  # corn
    [5., 4., 3.]   # coffee
])

s = commodity_embeddings @ commodity_embeddings.T
print(s)

