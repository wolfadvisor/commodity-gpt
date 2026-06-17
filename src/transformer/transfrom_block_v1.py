import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        # ==================================
        # ATTENTION
        # ==================================

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            batch_first=True
        )

        # ==================================
        # LAYER NORM 1
        # ==================================

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        # ==================================
        # FEED FORWARD
        # ==================================

        self.ffn = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )

        )

        # ==================================
        # LAYER NORM 2
        # ==================================

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # ==================================
        # ATTENTION
        # ==================================

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # ==================================
        # RESIDUAL + NORM
        # ==================================

        x = self.norm1(
            x + attention_output
        )

        # ==================================
        # FFN
        # ==================================

        ffn_output = self.ffn(x)

        # ==================================
        # RESIDUAL + NORM
        # ==================================

        x = self.norm2(
            x + ffn_output
        )

        return x