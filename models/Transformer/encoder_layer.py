import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    An encoder layer.

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    n_heads : int
        Number of attention heads

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """
    def __init__(
        self, d_model: int, n_heads: int, hidden_size: int, dropout: float = 0.5
    ) -> None:
        super(EncoderLayer, self).__init__()

        # an encoder layer has two sub-layers:
        #   - multi-head self-attention
        #   - positon-wise fully connected feed-forward network
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_size, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data

        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of the current encoder layer

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att_out, att = self.attention(x, mask=mask)  # (batch_size, word_pad_len, d_model), (batch_size, n_heads, word_pad_len, word_pad_len)
        out = self.feed_forward(att_out)  # (batch_size, word_pad_len, d_model)
        return out, att
