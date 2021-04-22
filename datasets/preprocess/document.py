"""
Preprocess data for document classification.
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
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

def read_csv(
    csv_folder: str, split, sentence_limit: int, word_limit: int
) -> Tuple[list, list, Counter]:
    """
    Read CSVs containing raw training data, clean documents and labels, and do
    a word-count.

    Parameters
    ----------
    csv_folder : str
        Folder containing the dataset in CSV format files

    split : str
        'train' or 'test' split?

    sentence_limit : int
        Truncate long documents to these many sentences

    word_limit : int
        Truncate long sentences to these many words

    Returns
    -------
    docs : list
        Documents ([ [word1a, ... ], ..., [wordna, ... ] ])

    labels : list
        List of label of each document

    word_counter : Counter
    """
    assert split in {'train', 'test'}

    docs = []
    labels = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header = None)
    for i in tqdm(range(data.shape[0])):
        row = list(data.loc[i, :])

        sentences = list()

        for text in row[1:]:
            for paragraph in get_clean_text(text).splitlines():
                sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        words = list()
        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            # if sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)

        # if all sentences were empty
        if len(words) == 0:
            continue

        labels.append(int(row[0]) - 1) # since labels are 1-indexed in the CSV
        docs.append(words)

    return docs, labels, word_counter

def encode_and_pad(
    input_docs: list, word_map: Dict[str, int], sentence_limit: int, word_limit: int
) -> Tuple[list, list, list]:
    """
    Encode sentences, and pad them to fit word_limit.

    Parameters
    ----------
    input_docs : list
        Documents ([ [word1a, ... ], ..., [wordna, ... ] ])

    word_map : Dict[str, int]
        Word2ix map

    sentence_limit : int
        Max number of sentences in a document

    word_limit : int
        Max number of words in a sentence

    Returns
    -------
    encoded_docs : list
        Encoded and padded document

    sentences_per_document : list
        Number of sentences per document

    words_per_sentence : list
        Number of words per sentence
    """
    encoded_docs = list(
        map(lambda doc: list(
            map(lambda s: list(
                map(lambda w: word_map.get(w, word_map['<unk>']), s)
            ) + [0] * (word_limit - len(s)), doc)
        ) + [[0] * word_limit] * (sentence_limit - len(doc)), input_docs)
    )
    sentences_per_document = list(map(lambda doc: len(doc), input_docs))
    words_per_sentence = list(
        map(lambda doc: list(
            map(lambda s: len(s), doc)
        ) + [0] * (sentence_limit - len(doc)), input_docs)
    )
    return encoded_docs, sentences_per_document, words_per_sentence

def run_prepro(
    csv_folder: str,
    output_folder: int,
    sentence_limit: int,
    word_limit: int,
    min_word_count: int = 5
) -> None:
    """
    Create data files to be used for training the model.

    Parameters
    ----------
    csv_folder : str
        Folder where the CSVs with the raw data are located

    output_folder : str
        Folder where files must be created

    sentence_limit : int
        Truncate long documents to these many sentences

    word_limit : int
        Truncate long sentences to these many words

    min_word_count : int
        Discard rare words which occur fewer times than this number
    """
    # --------------------- training data ---------------------
    print('\nTraining data: reading and preprocessing...\n')
    train_docs, train_labels, word_counter = read_csv(csv_folder, 'train', sentence_limit, word_limit)

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
    encoded_train_docs, sentences_per_train_document, words_per_train_sentence = \
        encode_and_pad(train_docs, word_map, sentence_limit, word_limit)

    # save
    print('Training data: saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(words_per_train_sentence)
    # because of the large data, saving as a JSON can be very slow
    torch.save({
        'docs': encoded_train_docs,
        'labels': train_labels,
        'sentences_per_document': sentences_per_train_document,
        'words_per_sentence': words_per_train_sentence
    }, os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Training data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    # free some memory
    del train_docs, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    # --------------------- test data ---------------------
    print('Test data: reading and preprocessing...\n')
    test_docs, test_labels, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)

    # encode and pad
    print('\nTest data: encoding and padding...\n')
    encoded_test_docs, sentences_per_test_document, words_per_test_sentence = \
        encode_and_pad(test_docs, word_map, sentence_limit, word_limit)

    # save
    print('Test data: saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(words_per_test_sentence)
    torch.save({
        'docs': encoded_test_docs,
        'labels': test_labels,
        'sentences_per_document': sentences_per_test_document,
        'words_per_sentence': words_per_test_sentence
    }, os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Test data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    print('All done!\n')
