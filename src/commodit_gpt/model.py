import torch
import torch.nn as nn


class CommodityGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=16,
        hidden_dim=64
    ):
        super().__init__()

        # Embeddings

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Transformer Block

        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=1,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        # LM Head

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.transformer(x)

        logits = self.lm_head(x)

        return logits