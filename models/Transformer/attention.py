import torch
import torch.nn as nn

'''
scaled dot-product attention

attributes:
    scale: scale factor (sqrt(d_k))
    dropout: dropout
'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout = 0.5):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    '''
    input param:
        Q: query (batch_size, n_heads, word_pad_len, d_k)
        K: key
        V: value
        mask: padding mask metrix, None if it is not needed (batch_size, 1, 1, word_pad_len)
    return:
        context: context vector (batch_size, n_heads, word_pad_len, d_k)
        att: attention weights (batch_size, n_heads, word_pad_len, word_pad_len)
    '''
    def forward(self, Q, K, V, mask = None):
        # Q·K^T / sqrt(d_k)
        att = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size, n_heads, word_pad_len, word_pad_len)

        # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)

        # eq.1: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k))·V
        att = self.dropout(self.softmax(att))  # (batch_size, n_heads, word_pad_len, word_pad_len)
        context = torch.matmul(att, V)  # (batch_size, n_heads, word_pad_len, d_k)
        
        return context, att


'''
multi-head self-attention

attributes:
    d_model: size of word embeddings
    n_heads: number of attention heads
    dropout: dropout
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.5):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0

        # we assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # linear projections
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k)

        # scaled dot-product attention
        scale = self.d_k ** 0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale = scale)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * self.d_k, d_model)
    
        self.dropout = nn.Dropout(dropout)


    '''
    input param:
        x: input data (batch_size, word_pad_len, d_model)
        mask: padding mask metrix, None if it is not needed (batch_size, 1, word_pad_len)

    return: 
        out: output of multi-head self-attention network (batch_size, word_pad_len, d_model)
        att: attention weights (batch_size, n_heads, word_pad_len, word_pad_len)
    '''
    def forward(self, x, mask):
        
        batch_size = x.size(0)
        
        Q = self.W_Q(x)  # (batch_size, word_pad_len, n_heads * d_k)
        K = self.W_K(x)
        V = self.W_V(x)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)  # (batch_size, word_pad_len, n_heads, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (batch_size, n_heads, word_pad_len, d_k)
        
        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, d_k)

        context, att = self.attention(Q, K, V, mask = mask)  # (batch_size, n_heads, word_pad_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)  # (batch_size, word_pad_len, n_heads * d_k)

        out = self.dropout(self.fc(context))  # (batch_size, word_pad_len, d_model)

        out = out + x  # residual connection
        out = self.layer_norm(out)  # LayerNorm
        
        return out, att