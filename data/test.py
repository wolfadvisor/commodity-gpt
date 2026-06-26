import torch
from dataset import CommodityDataset as ds
from torch.utils.data import DataLoader
from rich import print
from rich.traceback import install

install()

X = torch.tensor([
 [0,0,0],
 [0,0,1],
 [0,1,2],
 [1,2,3]
])

y = torch.tensor([
 1,
 2,
 3,
 4
])

dataset = ds(X, y)

print(f"len dataset:{len(dataset)}")
print()
print(dataset[0])
print()
print(dataset[1])

loader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle=True
    )
for X_bacth, y_bacth in loader:
    print(X_bacth.shape)
    print()
    print(y_bacth)
    break   