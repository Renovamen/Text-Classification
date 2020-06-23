import torch
import torch.nn as nn
from .sent_encoder import *

'''
the overarching Hierarchial Attention Network (HAN)

attributes:
    n_classes: number of classes
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
class HAN(nn.Module):

    def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune,
                 word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers, 
                 word_att_size, sentence_att_size, dropout = 0.5):

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


    '''
    input param:
        documents: encoded document-level data (n_documents, sent_pad_len, word_pad_len)
        sentences_per_document: document lengths (n_documents)
        words_per_sentence: sentence lengths (n_documents, sent_pad_len)
    
    return: 
        scores: class scores
        word_alphas: attention weights of words
        sentence_alphas: attention weights of sentences
    '''
    def forward(self, documents, sentences_per_document, words_per_sentence):
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