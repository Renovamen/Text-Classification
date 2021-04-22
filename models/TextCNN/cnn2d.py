import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class TextCNN2D(nn.Module):
    """
    Implementation of 2D version of TextCNN proposed in paper [1].

    `Here <https://github.com/yoonkim/CNN_sentence>`_ is the official
    implementation of TextCNN.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    n_kernels : int
        Number of kernels

    kernel_sizes : List[int]
        Size of each kernel

    dropout : float
        Dropout

    n_channels : int
        Number of channels (1 / 2)

    References
    ----------
    1. "`Convolutional Neural Networks for Sentence Classification. \
        <https://www.aclweb.org/anthology/D14-1181.pdf>`_" Yoon Kim. EMNLP 2014.
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        n_kernels: int,
        kernel_sizes: List[int],
        dropout: float,
        n_channels = 1
    ) -> None:
        super(TextCNN2D, self).__init__()

        # embedding layer
        self.embedding1 = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, 1, fine_tune)

        if n_channels == 2:
            # multichannel: a static channel and a non-static channel
            # which means embedding2 is frozen
            self.embedding2 = nn.Embedding(vocab_size, emb_size)
            self.set_embeddings(embeddings, 1, False)
        else:
            self.embedding2 = None

        # 2d conv layer
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = n_channels,
                out_channels = n_kernels,
                kernel_size = (size, emb_size)
            )
            for size in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def set_embeddings(
        self,
        embeddings: torch.Tensor,
        layer_id: int = 1,
        fine_tune: bool = True
    ) -> None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        layer_id : int
            Embedding layer 1 or 2 (when adopting multichannel architecture)

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            if layer_id == 1:
                self.embedding1.weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding2.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            if layer_id == 1:
                self.embedding1.weight = nn.Parameter(embeddings, requires_grad = fine_tune)
            else:
                self.embedding2.weight = nn.Parameter(embeddings, requires_grad = fine_tune)

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
        # word embedding
        embeddings = self.embedding1(text).unsqueeze(1)  # (batch_size, 1, word_pad_len, emb_size)
        # multichannel
        if self.embedding2:
            embeddings2 = self.embedding2(text).unsqueeze(1)  # (batch_size, 1, word_pad_len, emb_size)
            embeddings = torch.cat((embeddings, embeddings2), dim = 1) # (batch_size, 2, word_pad_len, emb_size)

        # conv
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]  # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]

        # pooling
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]

        # flatten
        flattened = self.dropout(torch.cat(pooled, dim = 1))  # (batch size, n_kernels * len(kernel_sizes))
        scores = self.fc(flattened)  # (batch size, n_classes)

        return scores
