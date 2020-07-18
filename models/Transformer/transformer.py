import torch
from torch import nn
import copy
from .pe import PositionalEncoding
from .encoder_layer import EncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
mask tokens that are pads (not pad: 1, pad: 0)

input:
    seq: the sequence which needs masking (batch_size, word_pad_len)
    pad_idx: index of '<pad>' (default is 0)

return:
    mask: a padding mask metrix (batch_size, 1, word_pad_len)
'''
def get_padding_mask(seq, pad_idx = 0):
    mask = (seq != pad_idx).unsqueeze(-2).to(device)  # (batch_size, 1, word_pad_len)
    return mask


'''
Transformer, we only use the encoder part here.

attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    d_model: size of word embeddings
    word_pad_len: length of padded sequence
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    hidden_size: size of position-wise feed forward network
    n_heads: number of attention heads
    n_encoders: number of encoder layers
    dropout: dropout
'''
class Transformer(nn.Module):

    def __init__(self, n_classes, vocab_size, embeddings, d_model, word_pad_len, fine_tune, 
                    hidden_size, n_heads, n_encoders, dropout = 0.5):
        super(Transformer, self).__init__()
        
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.set_embeddings(embeddings, fine_tune)
        # postional coding layer
        self.postional_encoding = PositionalEncoding(d_model, word_pad_len, dropout)
        
        # an encoder layer
        self.encoder = EncoderLayer(d_model, n_heads, hidden_size, dropout)
        # encoder is composed of a stack of n_encoders identical encoder layers
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(n_encoders)
        ])

        # classifier
        self.fc = nn.Linear(word_pad_len * d_model, n_classes)


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

        # get padding mask
        mask = get_padding_mask(text)

        # word embedding
        embeddings = self.embeddings(text) # (batch_size, word_pad_len, emb_size)
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask = mask)  # (batch_size, word_pad_len, d_model)

        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # (batch_size, word_pad_len * d_model)
        scores = self.fc(encoder_out)  # (batch_size, n_classes)
        
        return scores #, att