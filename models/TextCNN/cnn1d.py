import torch
import torch.nn as nn
import torch.nn.functional as F

'''
TextCNN1D

attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    n_kernels: number of kernels
    kernel_sizes (list): size of each kernel
    dropout: dropout
    n_channels: number of channels (1 / 2)
'''
class TextCNN1D(nn.Module):
    def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune, 
                 n_kernels, kernel_sizes, dropout, n_channels = 1):

        super(TextCNN1D, self).__init__()

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

        # 1d conv layer
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels = n_channels, 
                out_channels = n_kernels, 
                kernel_size = size * emb_size,
                stride = emb_size
            ) 
            for size in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes) 
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    '''
    set weights of embedding layer

    input param:
        embeddings: word embeddings
        layer_id: embedding layer 1 or 2 (when adopting multichannel architecture)
        fine_tune: allow fine-tuning of embedding layer? 
                   (only makes sense when using pre-trained embeddings)
    '''
    def set_embeddings(self, embeddings, layer_id = 1, fine_tune = True):
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


    '''
    input param:
        text: input data (batch_size, word_pad_len)
        words_per_sentence: sentence lengths (batch_size)

    return: 
        scores: class scores (batch_size, n_classes)
    '''
    def forward(self, text, words_per_sentence):

        batch_size = text.size(0)

        # word embedding
        embeddings = self.embedding1(text).view(batch_size, 1, -1)  # (batch_size, 1, word_pad_len * emb_size)
        # multichannel
        if self.embedding2:
            embeddings2 = self.embedding2(text).view(batch_size, 1, -1)  # (batch_size, 1, word_pad_len * emb_size)
            embeddings = torch.cat((embeddings, embeddings2), dim = 1) # (batch_size, 2, word_pad_len * emb_size)

        # conv
        conved = [self.relu(conv(embeddings)) for conv in self.convs]  # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]

        # pooling
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]
        
        # flatten
        flattened = self.dropout(torch.cat(pooled, dim = 1))  # (batch size, n_kernels * len(kernel_sizes))
        scores = self.fc(flattened)  # (batch size, n_classes)
        
        return scores