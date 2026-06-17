import torch
from rich import print
from rich.panel import Panel
from rich.traceback import install

install()
commodities_price = torch.tensor(
    [120, 220, 320, 420],
    dtype=torch.float32
)

print(Panel.fit(
    f"""
Commodities : {commodities_price}
Shape       : {commodities_price.shape}

+10% Prices : {commodities_price * 1.10}

Average     : {commodities_price.mean():.2f}
Maximum     : {commodities_price.max():.2f}
""",
    title="Commodity Tensor Analysis"
))

vocabulary = {
    "soybeans" : 0,
    "corn": 1,
    "sugar":2,
    "coffee":3
}
commodities = [
    "soybean",
    "corn",
    "sugar",
    "coffee"
]
def enum_vocabulary(text = commodities):
    vocab = {
        word : index 
        for index, word in enumerate(text)
    }

    return vocab

new_vocabulary = enum_vocabulary()
print (new_vocabulary)


