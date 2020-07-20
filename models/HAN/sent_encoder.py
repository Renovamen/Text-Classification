import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from .word_encoder import *

'''
sentence-level attention module

attributes:
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    word_rnn_size: size of (bidirectional) word-level RNN
    sentence_rnn_size: size of (bidirectional) sentence-level RNN
    word_rnn_layers: number of layers in word-level RNN
    sentence_rnn_layers: number of layers in sentence-level RNN
    word_att_size: size of word-level attention layer
    sentence_att_size: size of sentence-level attention layer
    dropout: dropout
'''
class SentenceEncoder(nn.Module):

    def __init__(self, vocab_size, embeddings, emb_size, fine_tune, 
                 word_rnn_size, sentence_rnn_size, 
                 word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):

        super(SentenceEncoder, self).__init__()

        # word encoder
        self.word_encoder = WordEncoder(
            vocab_size = vocab_size, 
            embeddings = embeddings, 
            emb_size = emb_size, 
            fine_tune = fine_tune, 
            word_rnn_size = word_rnn_size, 
            word_rnn_layers = word_rnn_layers, 
            word_att_size = word_att_size, 
            dropout = dropout
        )

        # sentence-level RNN (bidirectional GRU)
        self.sentence_rnn = nn.GRU(
            2 * word_rnn_size, sentence_rnn_size, 
            num_layers = sentence_rnn_layers,
            bidirectional = True, 
            dropout = (0 if sentence_rnn_layers == 1 else dropout), 
            batch_first = True
        )

        # sentence-level attention network
        self.W_s = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # sentence context vector u_s to take dot-product with
        self.u_s = nn.Linear(sentence_att_size, 1, bias = False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)


    '''
    input param:
        documents: encoded document-level data (n_documents, sent_pad_len, word_pad_len)
        sentences_per_document: document lengths (n_documents)
        words_per_sentence: sentence lengths (n_documents, sent_pad_len)
    
    return: 
        documents: document embeddings
        word_alphas: attention weights of words
        sentence_alphas: attention weights of sentences
    '''
    def forward(self, documents, sentences_per_document, words_per_sentence):

        # pack sequences (remove word-pads, DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(
            documents,
            lengths = sentences_per_document.tolist(),
            batch_first = True,
            enforce_sorted = False
        )  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(
            words_per_sentence,
            lengths = sentences_per_document.tolist(),
            batch_first = True,
            enforce_sorted = False
        )  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)

        # word encoder, get sentence vectors
        sentences, word_alphas = self.word_encoder(
            packed_sentences.data,
            packed_words_per_sentence.data
        )  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)

        # run through sentence-level RNN (PyTorch automatically applies it on the PackedSequence)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(
            data = sentences,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # unpack sequences (re-pad with 0s, SENTENCES -> DOCUMENTS)
        # we do unpacking here because attention weights have to be computed only over sentences in the same document
        documents, _ = pad_packed_sequence(packed_sentences, batch_first = True)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        # sentence-level attention
        # eq.8: u_i = tanh(W_s h_i + b_s)
        u_i = self.W_s(documents)  # (n_documents, max(sentences_per_document), att_size)
        u_i = self.tanh(u_i)  # (n_documents, max(sentences_per_document), att_size)

        # eq.9: alpha_i = softmax(u_i u_s)
        sent_alphas = self.u_s(u_i).squeeze(2)  # (n_documents, max(sentences_per_document))
        sent_alphas = self.softmax(sent_alphas)  # (n_documents, max(sentences_per_document))
        
        # form document vectors
        # eq.10: v = \sum_i α_i h_i
        documents = documents * sent_alphas.unsqueeze(2)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)
        documents = documents.sum(dim = 1)  # (n_documents, 2 * sentence_rnn_size)

        # also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(
            data = word_alphas,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ), batch_first = True)  # (n_documents, max(sentences_per_document), max(words_per_sentence))

        return documents, word_alphas, sent_alphas