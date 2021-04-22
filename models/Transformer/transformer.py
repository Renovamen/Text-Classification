import copy
import torch
from torch import nn

from .pe import PositionalEncoding
from .encoder_layer import EncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Mask tokens that are pads (not pad: 1, pad: 0)

    Parameters
    ----------
    seq : torch.Tensor (batch_size, word_pad_len)
        The sequence which needs masking

    pad_idx: index of '<pad>' (default is 0)

    Returns
    -------
    mask : torch.Tensor (batch_size, 1, word_pad_len)
        A padding mask metrix
    """
    mask = (seq != pad_idx).unsqueeze(-2).to(device)  # (batch_size, 1, word_pad_len)
    return mask


class Transformer(nn.Module):
    """
    Implementation of Transformer proposed in paper [1]. Only the encoder part
    is used here.

    `Here <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py>`_
    is the official TensorFlow implementation of Transformer.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    d_model : int
        Size of word embeddings

    word_pad_len : int
        Length of the padded sequence

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    hidden_size : int
        Size of position-wise feed forward network

    n_heads : int
        Number of attention heads

    n_encoders : int
        Number of encoder layers

    dropout : float
        Dropout

    References
    ----------
    1. "`Attention Is All You Need. <https://arxiv.org/abs/1706.03762>`_" \
        Ashish Vaswani, et al. NIPS 2017.
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        d_model: torch.Tensor,
        word_pad_len: int,
        fine_tune: bool,
        hidden_size: int,
        n_heads: int,
        n_encoders: int,
        dropout: float = 0.5
    ) -> None:
        super(Transformer, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.set_embeddings(embeddings, fine_tune)
        # postional coding layer
        self.postional_encoding = PositionalEncoding(d_model, word_pad_len, dropout)

        # an encoder layer
        self.encoder = EncoderLayer(d_model, n_heads, hidden_size, dropout)
        # encoder is composed of a stack of n_encoders identical encoder layers
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(n_encoders)
        ])

        # classifier
        self.fc = nn.Linear(word_pad_len * d_model, n_classes)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = True) -> None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad = fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        # get padding mask
        mask = get_padding_mask(text)

        # word embedding
        embeddings = self.embeddings(text) # (batch_size, word_pad_len, emb_size)
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask = mask)  # (batch_size, word_pad_len, d_model)

        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # (batch_size, word_pad_len * d_model)
        scores = self.fc(encoder_out)  # (batch_size, n_classes)

        return scores #, att
