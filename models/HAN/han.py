import torch
import torch.nn as nn
from typing import Tuple

from .sent_encoder import *

class HAN(nn.Module):
    """
    Implementation of Hierarchial Attention Network (HAN) proposed in paper [1].

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

    word_rnn_size : int
        Size of (bidirectional) word-level RNN

    sentence_rnn_size : int
        Size of (bidirectional) sentence-level RNN

    word_rnn_layers : int
        Number of layers in word-level RNN

    sentence_rnn_layers : int
        Number of layers in sentence-level RNN

    word_att_size : int
        Size of word-level attention layer

    sentence_att_size : int
        Size of sentence-level attention layer

    dropout : float, optional, default=0.5
        Dropout
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        word_rnn_size: int,
        sentence_rnn_size: int,
        word_rnn_layers: int,
        sentence_rnn_layers: int,
        word_att_size: int,
        sentence_att_size: int,
        dropout: float = 0.5
    ) -> None:
        super(HAN, self).__init__()

        # sentence encoder
        self.sentence_encoder = SentenceEncoder(
            vocab_size, embeddings, emb_size, fine_tune,
            word_rnn_size, sentence_rnn_size,
            word_rnn_layers, sentence_rnn_layers,
            word_att_size, sentence_att_size,
            dropout
        )

        # classifier
        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        documents: torch.Tensor,
        sentences_per_document: torch.Tensor,
        words_per_sentence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        documents : torch.Tensor (n_documents, sent_pad_len, word_pad_len)
            Encoded document-level data

        sentences_per_document : torch.Tensor (n_documents)
            Document lengths

        words_per_sentence : torch.Tensor (n_documents, sent_pad_len)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores

        word_alphas : torch.Tensor
            Attention weights on each word

        sentence_alphas : torch.Tensor
            Attention weights on each sentence
        """
        # sentence encoder, get document vectors
        document_embeddings, word_alphas, sentence_alphas = self.sentence_encoder(
            documents,
            sentences_per_document,
            words_per_sentence
        )  # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))

        # classify
        # eq.11: p = softmax(W_c v + b_c)
        scores = self.fc(self.dropout(document_embeddings))  # (n_documents, n_classes)

        return scores, word_alphas, sentence_alphas
