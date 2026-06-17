from rich import print
from rich.panel import Panel
import torch

commodities = torch.tensor(
    [120, 220, 320, 420],
    dtype=torch.float32
)

print(Panel.fit(
    f"""
Commodities : {commodities}
Shape       : {commodities.shape}

+10% Prices : {commodities * 1.10}

Average     : {commodities.mean():.2f}
Maximum     : {commodities.max():.2f}
""",
    title="Commodity Tensor Analysis"
))

