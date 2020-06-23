import torch
from torch import nn

'''
fastText

attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    hidden_size: size of hidden layer
'''
class fastText(nn.Module):

    def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune, hidden_size):
        super(fastText, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # hidden layer
        self.hidden = nn.Linear(emb_size, hidden_size)
        
        # output layer
        self.fc = nn.Linear(hidden_size, n_classes)
        
        # softmax
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
        # word embedding
        embeddings = self.embeddings(text) # (batch_size, word_pad_len, emb_size)
        
        # average word embeddings in to sentence erpresentations
        avg_embeddings = embeddings.mean(dim = 1).squeeze(1) # (batch_size, emb_size)
        hidden = self.hidden(avg_embeddings) # (batch_size, hidden_size)
        
        # compute probability
        output = self.fc(hidden) # (batch_size, n_classes)
        scores = self.softmax(output) # (batch_size, n_classes)
        
        return scores