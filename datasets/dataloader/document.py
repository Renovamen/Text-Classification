'''
load data from manually preprocessed data (see prepo/document.py)
(for document classification)
'''

from torch.utils.data import Dataset
import torch
import os
import json
from datasets.info import *
from utils.embedding import *

'''
a PyTorch Dataset class to be used in a PyTorch DataLoader to create batches 
(for document classification)

attributes:
    data_folder: folder where data files are stored
    split: split, one of 'TRAIN' or 'TEST'
'''
class DocDataset(Dataset):

    def __init__(self, data_folder, split):
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self):
        return len(self.data['labels'])


'''
load data from files output by prepo/document.py (for document classification)

input param:
    config (Class): config settings
    split: 'trian' / 'test'
    build_vocab: build vocabulary?
                 only makes sense when split = 'train'

return:
    split = 'test':
        test_loader: dataloader for test data
    split = 'train':
        build_vocab = Flase:
            train_loader: dataloader for train data 
        build_vocab = True:
            train_loader: dataloader for train data 
            embeddings: pre-trained word embeddings (None if config.word2vec='random')
            emb_size: embedding size (config.emb_size if config.word2vec='random')
            word_map: word2ix map
            n_classes: number of classes
            vocab_size: size of vocabulary
'''
def load_data(config, split, build_vocab = True):

    split = split.lower()
    assert split in {'train', 'test'}

    # test
    if split == 'test':
        test_loader = torch.utils.data.DataLoader(
            DocDataset(config.output_path, 'test'), 
            batch_size = config.batch_size, 
            shuffle = False,
            num_workers = config.workers, 
            pin_memory = True
        )
        return test_loader

    # train
    else:
        # dataloaders
        train_loader = torch.utils.data.DataLoader(
            DocDataset(config.output_path, 'train'), 
            batch_size = config.batch_size,
            shuffle = True,
            num_workers = config.workers,
            pin_memory = True
        )

        if build_vocab == False:
            return train_loader

        else:
            # load word2ix map
            with open(os.path.join(config.output_path, 'word_map.json'), 'r') as j:
                word_map = json.load(j)
            # size of vocabulary
            vocab_size = len(word_map)
            
            # number of classes
            label_map, _ = get_label_map(config.dataset)
            n_classes = len(label_map)

            # word embeddings
            if config.word2vec == 'glove':
                # load Glove as pre-trained word embeddings for words in the word map
                word2vec_path = os.path.join(config.word2vec_folder, config.word2vec_name)
                embeddings, emb_size = load_embeddings(word2vec_path, word_map)
            else:
                embeddings = None
                emb_size = config.emb_size

            return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size