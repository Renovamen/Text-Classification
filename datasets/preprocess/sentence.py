'''
preprocess data for sentence classification
'''

import torch
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import pandas as pd
import os
import json

# tokenizers
word_tokenizer = TreebankWordTokenizer()

'''
preprocess text for being used in the model, including lower-casing, standardizing newlines and removing junk

input param:
    text: a string
return: 
    clean_text: a cleaner string
'''
def get_clean_text(text):

    if isinstance(text, float):
        return ''

    clean_text = text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')
    return clean_text


'''
read CSVs containing raw training data, clean documents and labels, and do a word-count

input param:
    csv_folder: folder containing the CSV
    split: train or test CSV?
    word_limit: truncate long sentences to these many words
return: 
    sents(list): sentences ([ word1, ..., wordn ])
    labels(list): labels for each sentence
    word_counter: a word-count
'''
def read_csv(csv_folder, split, word_limit):

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


'''
encode sentences, and pad them to fit word_limit

input param: 
    input_sents(list): sentences ([ word1, ..., wordn ])
    word_map: word map (word2ix)
    word_limit: max number of words in a sentence

return:
    encoded_sents: encoded and padded sentences
    words_per_sentence
'''
def encode_and_pad(input_sents, word_map, word_limit):
    encoded_sents = list(
        map(lambda s: list(
            map(lambda w: word_map.get(w, word_map['<unk>']), s)
        ) + [0] * (word_limit - len(s)), input_sents)
    )
    words_per_sentence = list(map(lambda s: len(s), input_sents))
    return encoded_sents, words_per_sentence


'''
create data files to be used for training the model

input param: 
    csv_folder: folder where the CSVs with the raw data are located
    output_folder: folder where files must be created
    word_limit: truncate long sentences to these many words
    min_word_count: discard rare words which occur fewer times than this number
'''
def run_prepro(csv_folder, output_folder, word_limit, min_word_count = 5):

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