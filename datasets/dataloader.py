"""
Load data from manually preprocessed data (see ``datasets/prepocess/``).
"""

import os
import json
from typing import Dict, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_embeddings
from utils.opts import Config
from .info import get_label_map

class DocDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
    (for document classification).

    Parameters
    ----------
    data_folder : str
        Path to folder where data files are stored

    split : str
        Split, one of 'TRAIN' or 'TEST'
    """
    def __init__(self, data_folder: str, split: str) -> None:
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


class SentDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
    (for sentence classification).

    Parameters
    ----------
    data_folder : str
        Path to folder where data files are stored

    split : str
        Split, one of 'TRAIN' or 'TEST'
    """
    def __init__(self, data_folder: str, split: str) -> None:
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data['sents'][i]), \
               torch.LongTensor([self.data['words_per_sentence'][i]]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


def load_data(
    config: Config, split: str, build_vocab: bool = True
) -> Union[DataLoader, Tuple[DataLoader, torch.Tensor, int, Dict[str, int], int, int]]:
    """
    Load data from files output by ``prepocess.py``.

    Parameters
    ----------
    config : Config
        Configuration settings

    split : str
        'trian' / 'test'

    build_vocab : bool
        Build vocabulary or not. Only makes sense when split = 'train'.

    Returns
    -------
    split = 'test':
        test_loader : DataLoader
            Dataloader for test data

    split = 'train':
        build_vocab = Flase:
            train_loader : DataLoader
                Dataloader for train data

        build_vocab = True:
            train_loader : DataLoader
                Dataloader for train data

            embeddings : torch.Tensor
                Pre-trained word embeddings (None if config.emb_pretrain = False)

            emb_size : int
                Embedding size (config.emb_size if config.emb_pretrain = False)

            word_map : Dict[str, int]
                Word2ix map

            n_classes : int
                Number of classes

            vocab_size : int
                Size of vocabulary
    """
    split = split.lower()
    assert split in {'train', 'test'}

    # test
    if split == 'test':
        test_loader = DataLoader(
            DocDataset(config.output_path, 'test') if config.model_name in ['han'] else SentDataset(config.output_path, 'test'),
            batch_size = config.batch_size,
            shuffle = False,
            num_workers = config.workers,
            pin_memory = True
        )
        return test_loader

    # train
    else:
        # dataloaders
        train_loader = DataLoader(
            DocDataset(config.output_path, 'train') if config.model_name in ['han'] else SentDataset(config.output_path, 'train'),
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
            if config.emb_pretrain == True:
                # load Glove as pre-trained word embeddings for words in the word map
                emb_path = os.path.join(config.emb_folder, config.emb_filename)
                embeddings, emb_size = load_embeddings(
                    emb_file = os.path.join(config.emb_folder, config.emb_filename),
                    word_map = word_map,
                    output_folder = config.output_path
                )
            # or initialize embedding weights randomly
            else:
                embeddings = None
                emb_size = config.emb_size

            return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size
