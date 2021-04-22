import torch
from torch import nn

class fastText(nn.Module):
    """
    Implementation of fastText proposed in paper [1].

    `Here <https://github.com/facebookresearch/fastText>`_ is the official
    implementation of fastText.

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

    hidden_size : int
        Size of the hidden layer

    References
    ----------
    1. "`Bag of Tricks for Efficient Text Classification. \
        <https://arxiv.org/abs/1607.01759>`_" Armand Joulin, et al. EACL 2017.
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        hidden_size: int
    ) -> None:
        super(fastText, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # hidden layer
        self.hidden = nn.Linear(emb_size, hidden_size)

        # output layer
        self.fc = nn.Linear(hidden_size, n_classes)

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
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

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
        embeddings = self.embeddings(text)  # (batch_size, word_pad_len, emb_size)

        # average word embeddings in to sentence erpresentations
        avg_embeddings = embeddings.mean(dim=1).squeeze(1)  # (batch_size, emb_size)
        hidden = self.hidden(avg_embeddings)  # (batch_size, hidden_size)

        # compute probability
        scores = self.fc(hidden)  # (batch_size, n_classes)

        return scores
