import torch
import torch.nn as nn


class CommodityGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=16,
        hidden_dim=64,
        max_seq_length=128
    ):
        super().__init__()

        # ==========================
        # Token Embedding
        # ==========================

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ==========================
        # Position Embedding
        # ==========================

        self.position_embedding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )

        # ==========================
        # Transformer
        # ==========================

        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=1,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        # ==========================
        # LM Head
        # ==========================

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, tokens):

        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device=tokens.device
        )

        pos_emb = self.position_embedding(
            positions
        )

        token_emb = self.embedding(
            tokens
        )

        x = token_emb + pos_emb

        x = self.transformer(x)

        logits = self.lm_head(x)

        return logits