'''
script for loading data for sentence classification using torchtext (never used)

I abandon this because torchtext loads all data in one go, which occupies 
too much memory and slows down the training speed, expecially when the dataset 
is big.

So I finally choose to preprocess data manually (see datasets/prepocess/) and 
load data dynamically via Pytorch's Dataloader.
'''

import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
load data using torchtext (only for sentence classification)

input param:
    config (Class): config settings
    split: 'trian' / 'test'
    build_vocab: build vocabulary?
                 only makes sense when split = 'train'

return:
    split = 'test':
        test_loader: data iterator for test data
    split = 'train':
        build_vocab = Flase:
            train_loader: data iterator for train data 
        build_vocab = True:
            train_loader: data iterator for train data 
            embeddings: pre-trained word embeddings (None if config.word2vec='random')
            emb_size: embedding size (config.emb_size if config.word2vec='random')
            word_map: word2ix map
            n_classes: number of classes
            vocab_size: size of vocabulary
'''
def load_data(config, split, build_vocab = True):

    split = split.lower()
    assert split in {'train', 'test'}

    tokenizer = lambda x: x.split()

    # Field: a class that storing information about the way of preprocessing
    TEXT = data.Field(sequential = True, tokenize = tokenizer, lower = True, include_lengths = True, batch_first = True, fix_length = config.word_limit)
    LABEL = data.Field(sequential = False, unk_token = None) # we don't need <unk> in label

    # Yahoo Answers
    if config.dataset == 'yahoo_answers':
        # |  label  |  text1  |  text2  |  text3  |
        fields = [('label', LABEL), ('text1', TEXT), ('text2', TEXT), ('text3', TEXT)]
    # Yelp Review Full, Yelp Review Polarity
    elif config.dataset in ['yelp_full', 'yelp_polarity']:
        # |  label  |  text  |
        fields = [('label', LABEL), ('text', TEXT)]
    # AG News, DBpedia, Amazon Review Full, Amazon Review Polarity   
    else:
        # |  label  |  we don't need  |  text  |
        fields = [('label', LABEL), (None, None), ('text', TEXT)]

    # load data
    train_data, test_data = data.TabularDataset.splits(
        path = config.dataset_path,
        train = 'train.csv',
        test = 'test.csv',
        format = 'csv',
        fields = fields,
        skip_header = False
    )
    
    # concatenate all text to a single piece of text
    if config.dataset == 'yahoo_answers':
        for train_item in train_data:
            train_item.text = train_item.text1 + train_item.text2 + train_item.text3
            del train_item.text1, train_item.text2, train_item.text3   
        train_data.fields = {
            'label': LABEL, 
            'text': TEXT
        }
        
    if config.word2vec == 'glove':
        # build word2ix map
        # and load Glove as pre-trained word embeddings for words in the word map
        vectors = Vectors(name = config.word2vec_name, cache = config.word2vec_folder)
        TEXT.build_vocab(train_data, vectors = vectors)
        embeddings = TEXT.vocab.vectors
        emb_size = TEXT.vocab.vectors.size()[1]
    else:
        # build word2ix map only
        TEXT.build_vocab(train_data)
        embeddings = None
        emb_size = config.emb_size
    # size of vocabulary
    vocab_size = len(TEXT.vocab)
    
    # number of classes
    LABEL.build_vocab(train_data)
    n_classes = len(LABEL.vocab)

    # word map
    word_map = TEXT.vocab.stoi

    # BucketIterator: defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed
    train_loader, test_loader = data.BucketIterator.splits(
        (train_data, test_data), 
        batch_size = config.batch_size, 
        sort_key = lambda x: len(x.text), 
        device = device, 
        repeat = False, 
        shuffle = True
    )

    if split == 'test':
        return test_loader
    else:
        if build_vocab == False:
            return train_loader
        else:
            return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size