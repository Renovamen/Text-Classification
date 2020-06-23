import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

'''
the overarching Hierarchial Attention Network (HAN)

attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    emb_size: size of word embeddings
    word_rnn_size: size of (bidirectional) word-level RNN
    sentence_rnn_size: size of (bidirectional) sentence-level RNN
    word_rnn_layers: number of layers in word-level RNN
    sentence_rnn_layers: number of layers in sentence-level RNN
    word_att_size: size of word-level attention layer
    sentence_att_size: size of sentence-level attention layer
    dropout: dropout
'''
class HierarchialAttentionNetwork(nn.Module):

    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, dropout = 0.5):

        super(HierarchialAttentionNetwork, self).__init__()

        # sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.sentence_attention = SentenceAttention(
            vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
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
        # apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(
            documents, 
            sentences_per_document,
            words_per_sentence
        )  # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))

        # classify
        scores = self.fc(self.dropout(document_embeddings))  # (n_documents, n_classes)

        return scores, word_alphas, sentence_alphas


'''
the sentence-level attention module

attributes:
    vocab_size: number of words in the vocabulary of the model
    emb_size: size of word embeddings
    word_rnn_size: size of (bidirectional) word-level RNN
    sentence_rnn_size: size of (bidirectional) sentence-level RNN
    word_rnn_layers: number of layers in word-level RNN
    sentence_rnn_layers: number of layers in sentence-level RNN
    word_att_size: size of word-level attention layer
    sentence_att_size: size of sentence-level attention layer
    dropout: dropout
'''
class SentenceAttention(nn.Module):

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):

        super(SentenceAttention, self).__init__()

        # word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,dropout)

        # bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(
            2 * word_rnn_size, sentence_rnn_size, 
            num_layers = sentence_rnn_layers,
            bidirectional = True, 
            dropout = dropout, 
            batch_first = True
        )

        # sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1, bias = False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector
        # you could also do this with:
        # self.sentence_context_vector = nn.Parameter(torch.FloatTensor(1, sentence_att_size))
        # self.sentence_context_vector.data.uniform_(-0.1, 0.1)
        # and then take the dot-product

        # dropout
        self.dropout = nn.Dropout(dropout)


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

        # re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
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

        # find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(
            packed_sentences.data,
            packed_words_per_sentence.data
        )  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)

        # apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(
            data = sentences,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # find attention vectors by applying the attention linear layer on the output of the RNN
        att_s = self.sentence_attention(packed_sentences.data)  # (n_sentences, att_size)
        att_s = torch.tanh(att_s)  # (n_sentences, att_size)
        # take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # compute softmax over the dot-product manually
        # manually because they have to be computed only over sentences in the same document

        # first, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(
            data = att_s,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ), batch_first = True)  # (n_documents, max(sentences_per_document))

        # calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / torch.sum(att_s, dim = 1, keepdim = True)  # (n_documents, max(sentences_per_document))

        # similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(packed_sentences, batch_first = True)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        # find document embeddings
        documents = documents * sentence_alphas.unsqueeze(2)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)
        documents = documents.sum(dim = 1)  # (n_documents, 2 * sentence_rnn_size)

        # also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(
            data = word_alphas,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ), batch_first = True)  # (n_documents, max(sentences_per_document), max(words_per_sentence))

        return documents, word_alphas, sentence_alphas


'''
the word-level attention module

attributes:
    vocab_size: number of words in the vocabulary of the model
    emb_size: size of word embeddings
    word_rnn_size: size of (bidirectional) word-level RNN
    word_rnn_layers: number of layers in word-level RNN
    word_att_size: size of word-level attention layer
    dropout: dropout
'''
class WordAttention(nn.Module):

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        
        super(WordAttention, self).__init__()

        # embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # bidirectional word-level RNN
        self.word_rnn = nn.GRU(
            emb_size, word_rnn_size, 
            num_layers = word_rnn_layers, 
            bidirectional = True,
            dropout = dropout, 
            batch_first = True
        )

        # word-level attention network
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

        # word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        # you could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # and then take the dot-product

        self.dropout = nn.Dropout(dropout)


    '''
    initialize embedding and fc layer with the uniform distribution
    '''
    def init_embedding_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)


    '''
    initialize embedding layer with pre-computed embeddings

    input param:
        embeddings: pre-computed embeddings
    '''
    def load_pretrained_embeddings(self, embeddings):
        self.embeddings.weight = nn.Parameter(embeddings)


    '''
    allow fine-tuning of embedding layer? 
    (only makes sense to not-allow if using pre-trained embeddings)

    input param:
        fine_tune: allow?
    '''
    def fine_tune_embeddings(self, fine_tune=False):
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune


    '''
    input param:
        sentences: encoded sentence-level data (n_sentences, word_pad_len, emb_size)
        words_per_sentence: sentence lengths (n_sentences)
    
    return: 
        sentences: sentence embeddings
        word_alphas: attention weights of words
    '''
    def forward(self, sentences, words_per_sentence):
        # get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_sentences, word_pad_len, emb_size)

        # re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            sentences,
            lengths = words_per_sentence.tolist(),
            batch_first = True,
            enforce_sorted = False
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(packed_words)  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # compute softmax over the dot-product manually
        # manually because they have to be computed only over words in the same sentence

        # first, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(
            data = att_w,
            batch_sizes = packed_words.batch_sizes,
            sorted_indices = packed_words.sorted_indices,
            unsorted_indices = packed_words.unsorted_indices
        ), batch_first = True)  # (n_sentences, max(words_per_sentence))

        # calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim = 1, keepdim = True)  # (n_sentences, max(words_per_sentence))

        # similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words, batch_first = True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim = 1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas