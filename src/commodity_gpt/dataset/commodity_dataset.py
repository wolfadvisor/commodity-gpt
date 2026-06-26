import torch
from torch.utils.data import Dataset

class CommodityDataset(Dataset):

    def __init__(self, tokens, context_size=3):
        
        self.X = []
        self.y = []

        for i in range(
            len(tokens) - context_size
        ):
            input_sequence = tokens[
                i:i+context_size
            ]
            target = tokens[
                i+context_size
            ]
            self.X.append(input_sequence)
            self.y.append(target)

        self.X =torch.tensor(
            self.X,
            dtype=torch.long
        )
        self.y = torch.tensor(
            self.y,
            dtype=torch.long
        )


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return(
            self.X[idx],
            self.y[idx]
        )
    
   