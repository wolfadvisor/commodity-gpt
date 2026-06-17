from rich import print, inspect
from rich.traceback import install
import torch

install()

# Create a tensor

prices = torch.tensor([
    [120,250,200],
    [240,320,325],
    [360,400,450]
],dtype=torch.float32)

print(prices * 1.10)
print(prices.shape)

soybeans_price = torch.tensor([
    120.50,
    125.80,
    130.20
])

print(soybeans_price)
print(soybeans_price.shape)

#Matrix Creation

a = torch.tensor(
    [[120, 125, 130],
    [220, 225, 230],
    [320, 325, 330]
],dtype= torch.float32)

print(a)
print(a.shape)

C = torch.matmul(prices, a)
print(C)

#Dot product
soybean = torch.tensor([
    1.0,
    2.0,
    3.0
])

corn = torch.tensor([
    4.0,
    5.0,
    6.0
])

sugar = torch.tensor([
    10,
    15,
    20
])

result = torch.dot(
    soybean,
    corn    
)

print(result)

#Tensor Shapes

x = torch.rand(4, 8)
print(x.shape)