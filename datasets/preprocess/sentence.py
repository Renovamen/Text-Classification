"""
Preprocess data for sentence classification.
"""

import torch
from typing import Tuple, Dict
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import pandas as pd
import os
import json

from .utils import get_clean_text

# tokenizers
word_tokenizer = TreebankWordTokenizer()

def read_csv(csv_folder: str, split: str, word_limit: int) -> Tuple[list, list, Counter]:
    """
    Read CSVs containing raw training data, clean sentences and labels, and do
    a word-count.

    Parameters
    ----------
    csv_folder : str
        Folder containing the dataset in CSV format files

    split : str
        'train' or 'test' split?

    word_limit : int
        Truncate long sentences to these many words

    Returns
    -------
    sents : list
        Sentences ([ word1, ..., wordn ])

    labels : list
        List of label of each sentence

    word_counter : Counter
    """
    assert split in {'train', 'test'}

    sents = []
    labels = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header = None)
    for i in tqdm(range(data.shape[0])):
        row = list(data.loc[i, :])

        s = ''

        for text in row[1:]:
            text = get_clean_text(text)
            s = s + text

        words = word_tokenizer.tokenize(s)[:word_limit]
        # if sentence is empty (due to removing punctuation, digits, etc.)
        if len(words) == 0:
            continue
        word_counter.update(words)

        labels.append(int(row[0]) - 1) # since labels are 1-indexed in the CSV
        sents.append(words)

    return sents, labels, word_counter

def encode_and_pad(
    input_sents: list, word_map: Dict[str, int], word_limit: int
) -> Tuple[list, list]:
    """
    Encode sentences, and pad them to fit word_limit.

    Parameters
    ----------
    input_sents : list
        Sentences ([ word1, ..., wordn ])

    word_map : Dict[str, int]
        Word2ix map

    word_limit : int
        Max number of words in a sentence

    Returns
    -------
    encoded_sents : list
        Encoded and padded sentences

    words_per_sentence : list
        Number of words per sentence
    """
    encoded_sents = list(
        map(lambda s: list(
            map(lambda w: word_map.get(w, word_map['<unk>']), s)
        ) + [0] * (word_limit - len(s)), input_sents)
    )
    words_per_sentence = list(map(lambda s: len(s), input_sents))
    return encoded_sents, words_per_sentence

def run_prepro(
    csv_folder: str, output_folder: str, word_limit: int, min_word_count: int = 5
) -> None:
    """
    Create data files to be used for training the model.

    Parameters
    ----------
    csv_folder : str
        Folder where the CSVs with the raw data are located

    output_folder : str
        Folder where files must be created

    word_limit : int
        Truncate long sentences to these many words

    min_word_count : int
        Discard rare words which occur fewer times than this number
    """
    # --------------------- training data ---------------------
    print('\nTraining data: reading and preprocessing...\n')
    train_sents, train_labels, word_counter = read_csv(csv_folder, 'train', word_limit)

    # create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('\nTraining data: discarding words with counts less than %d, the size of the vocabulary is %d.\n' % (min_word_count, len(word_map)))
    # save word map
    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)
    print('Training data: word map saved to %s.\n' % os.path.abspath(output_folder))

    # encode and pad
    print('Training data: encoding and padding...\n')
    encoded_train_sents, words_per_train_sent = encode_and_pad(train_sents, word_map, word_limit)

    # save
    print('Training data: saving...\n')
    assert len(encoded_train_sents) == len(train_labels) == len(words_per_train_sent)
    # because of the large data, saving as a JSON can be very slow
    torch.save({
        'sents': encoded_train_sents,
        'labels': train_labels,
        'words_per_sentence': words_per_train_sent
    }, os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Training data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    # free some memory
    del train_sents, encoded_train_sents, train_labels, words_per_train_sent

    # --------------------- test data ---------------------
    print('Test data: reading and preprocessing...\n')
    test_sents, test_labels, _ = read_csv(csv_folder, 'test', word_limit)

    # encode and pad
    print('\nTest data: encoding and padding...\n')
    encoded_test_sents, words_per_test_sent = encode_and_pad(test_sents, word_map, word_limit)

    # save
    print('Test data: saving...\n')
    assert len(encoded_test_sents) == len(test_labels) == len(words_per_test_sent)
    torch.save({
        'sents': encoded_test_sents,
        'labels': test_labels,
        'words_per_sentence': words_per_test_sent
    }, os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Test data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    print('All done!\n')
