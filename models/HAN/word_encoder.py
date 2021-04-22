import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Tuple

class WordEncoder(nn.Module):
    """
    Word-level attention module

    Parameters
    ----------
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

    word_rnn_layers : int
        Number of layers in word-level RNN

    word_att_size : int
        Size of word-level attention layer

    dropout : float
        Dropout
    """
    def __init__(
        self,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        word_rnn_size: int,
        word_rnn_layers: int,
        word_att_size: int,
        dropout: float
    ) -> None:
        super(WordEncoder, self).__init__()

        # word embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # word-level RNN (bidirectional GRU)
        self.word_rnn = nn.GRU(
            emb_size, word_rnn_size,
            num_layers = word_rnn_layers,
            bidirectional = True,
            dropout = (0 if word_rnn_layers == 1 else dropout),
            batch_first = True
        )

        # word-level attention network
        self.W_w = nn.Linear(2 * word_rnn_size, word_att_size)

        # word context vector u_w
        self.u_w = nn.Linear(word_att_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

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

    def forward(
        self, sentences: torch.Tensor, words_per_sentence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentences : torch.Tensor (n_sentences, word_pad_len, emb_size)
            Encoded sentence-level data

        words_per_sentence : torch.Tensor (n_sentences)
            Sentence lengths

        Returns
        -------
        sentences : torch.Tensor
            Sentence embeddings

        word_alphas : torch.Tensor
            Attention weights on each word
        """
        # word embedding, apply dropout
        sentences = self.dropout(self.embeddings(sentences)) # (n_sentences, word_pad_len, emb_size)

        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            sentences,
            lengths = words_per_sentence.tolist(),
            batch_first = True,
            enforce_sorted = False
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # run through word-level RNN (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(packed_words) # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        # we do unpacking here because attention weights have to be computed only over words in the same sentence
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # word-level attention
        # eq.5: u_it = tanh(W_w h_it + b_w)
        u_it = self.W_w(sentences)  # (n_sentences, max(words_per_sentence), att_size)
        u_it = self.tanh(u_it)  # (n_sentences, max(words_per_sentence), att_size)

        # eq.6: alpha_it = softmax(u_it u_w)
        word_alphas = self.u_w(u_it).squeeze(2)  # (n_sentences, max(words_per_sentence))
        word_alphas = self.softmax(word_alphas)  # (n_sentences, max(words_per_sentence))

        # form sentence vectors
        # eq.7: s_i = \sum_t α_it h_it
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas
