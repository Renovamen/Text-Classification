import torch

from .HAN import HAN
from .fastText import fastText
from .AttBiLSTM import AttBiLSTM
from .TextCNN import TextCNN1D, TextCNN2D
from .Transformer import Transformer
from utils.opts import Config

def make(
    config: Config,
    n_classes: int,
    vocab_size: int,
    embeddings: torch.Tensor,
    emb_size: int
) -> torch.nn.Module:
    """
    Make a model

    Parameters
    ----------
    config : Config
        Configuration settings

    n_classes : int
        Number of classes

    vocab_size : int
        Size of vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings
    """
    if config.model_name == 'han':
        model = HAN(
            n_classes = n_classes,
            vocab_size = vocab_size,
            embeddings = embeddings,
            emb_size = emb_size,
            fine_tune = config.fine_tune_word_embeddings,
            word_rnn_size = config.word_rnn_size,
            sentence_rnn_size = config.sentence_rnn_size,
            word_rnn_layers = config.word_rnn_layers,
            sentence_rnn_layers = config.sentence_rnn_layers,
            word_att_size = config.word_att_size,
            sentence_att_size = config.sentence_att_size,
            dropout = config.dropout
        )
    elif config.model_name == 'fasttext':
        model = fastText(
            n_classes = n_classes,
            vocab_size = vocab_size,
            embeddings = embeddings,
            emb_size = emb_size,
            fine_tune = config.fine_tune_word_embeddings,
            hidden_size = config.hidden_size
        )
    elif config.model_name == 'attbilstm':
        model = AttBiLSTM(
            n_classes = n_classes,
            vocab_size = vocab_size,
            embeddings = embeddings,
            emb_size = emb_size,
            fine_tune = config.fine_tune_word_embeddings,
            rnn_size = config.rnn_size,
            rnn_layers = config.rnn_layers,
            dropout = config.dropout
        )
    elif config.model_name == 'textcnn':
        if config.conv_layer == '2D':
            model = TextCNN2D(
                n_classes = n_classes,
                vocab_size = vocab_size,
                embeddings = embeddings,
                emb_size = emb_size,
                fine_tune = config.fine_tune_word_embeddings,
                n_kernels = config.n_kernels,
                kernel_sizes = config.kernel_sizes,
                n_channels = config.n_channels,
                dropout = config.dropout
            )
        elif config.conv_layer == '1D':
            model = TextCNN1D(
                n_classes = n_classes,
                vocab_size = vocab_size,
                embeddings = embeddings,
                emb_size = emb_size,
                fine_tune = config.fine_tune_word_embeddings,
                n_kernels = config.n_kernels,
                kernel_sizes = config.kernel_sizes,
                n_channels = config.n_channels,
                dropout = config.dropout
            )
        else:
            raise Exception("Convolution layer not supported: ", config.conv_layer)
    elif config.model_name == 'transformer':
        model = Transformer(
            n_classes = n_classes,
            vocab_size = vocab_size,
            embeddings = embeddings,
            d_model = emb_size,
            word_pad_len = config.word_limit,
            fine_tune = config.fine_tune_word_embeddings,
            hidden_size = config.hidden_size,
            n_heads = config.n_heads,
            n_encoders = config.n_encoders,
            dropout = config.dropout
        )
    else:
        raise Exception("Model not supported: ", config.model_name)

    return model
