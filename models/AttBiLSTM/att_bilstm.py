import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from .attention import *

'''
attention-based bidirectional LSTM

attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    rnn_size: size of bi-LSTM
    rnn_layers: number of layers in bi-LSTM
    dropout: dropout
'''
class AttBiLSTM(nn.Module):

    def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune,
                 rnn_size, rnn_layers, dropout):

        super(AttBiLSTM, self).__init__()

        self.rnn_size = rnn_size
        
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size, 
            num_layers = rnn_layers, 
            bidirectional = True,
            dropout = (0 if rnn_layers == 1 else dropout), 
            batch_first = True
        )

        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 1)


    '''
    set weights of embedding layer

    input param:
        embeddings: word embeddings
        fine_tune: allow fine-tuning of embedding layer? 
                   (only makes sense when using pre-trained embeddings)
    '''
    def set_embeddings(self, embeddings, fine_tune = True):
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad = fine_tune)


    '''
    input param:
        text: input data (batch_size, word_pad_len)
        words_per_sentence: sentence lengths (batch_size)

    return: 
        scores: class scores (batch_size, n_classes)
    '''
    def forward(self, text, words_per_sentence):
        
        # word embedding, apply dropout
        embeddings = self.dropout(self.embeddings(text)) # (batch_size, word_pad_len, emb_size)
        
        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            embeddings,
            lengths = words_per_sentence.tolist(),
            batch_first = True,
            enforce_sorted = False
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, emb_size)
        
        # run through bidirectional LSTM (PyTorch automatically applies it on the PackedSequence)
        rnn_out, _ = self.BiLSTM(packed_words)  # a PackedSequence object, where 'data' is the output of the LSTM (n_words, 2 * rnn_size)
        
        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first = True)  # (batch_size, word_pad_len, 2 * word_rnn_size)

        # eq.8: h_i = [\overrightarrow{h}_i ⨁ \overleftarrow{h}_i ]
        # H = {h_1, h_2, ..., h_T}
        H = rnn_out[ :, :, : self.rnn_size] + rnn_out[ :, :, self.rnn_size : ] # (batch_size, word_pad_len, rnn_size)

        # attention module
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores #, alphas