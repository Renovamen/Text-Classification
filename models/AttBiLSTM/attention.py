import torch
from torch import nn
from typing import Tuple

class Attention(nn.Module):
    """
    Attention network

    Parameters
    ----------
    rnn_size : int
        Size of Bi-LSTM
    """
    def __init__(self, rnn_size: int) -> None:
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor (batch_size, word_pad_len, hidden_size)
            Output of Bi-LSTM

        Returns
        -------
        r : torch.Tensor (batch_size, rnn_size)
            Sentence representation

        alpha : torch.Tensor (batch_size, word_pad_len)
            Attention weights
        """
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim = 1)  # (batch_size, rnn_size)

        return r, alpha
