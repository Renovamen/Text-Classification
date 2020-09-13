import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


'''
initialize embedding tensor with values from the uniform distribution

input param:
    input_embedding: embedding tensor
'''
def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


'''
load pre-trained word embeddings (Glove) for words in the word map

input params:
    emb_file: path to the file containing embeddings (stored in Glove format)
    word_map: word map
    output_folder: path of the folder to store output files

return: 
    embeddings in the same order as the words in the word map, dimension of embeddings
'''
def load_embeddings(emb_file, word_map, output_folder):
    
    emb_basename = os.path.basename(emb_file)
    cache_path = os.path.join(output_folder, emb_basename + '.pth.tar')

    # no cache, load embeddings from .txt file
    if not os.path.isfile(cache_path):
        # find embedding dimension
        with open(emb_file, 'r') as f:
            emb_size = len(f.readline().split(' ')) - 1
            num_lines = len(f.readlines()) 

        vocab = set(word_map.keys())

        # create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), emb_size)
        init_embedding(embeddings)

        # read embedding file
        for line in tqdm(open(emb_file, 'r'), total = num_lines, desc = 'Loading embeddings'):
            line = line.split(' ')

            emb_word = line[0]
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            # ignore word if not in train_vocab
            if emb_word not in vocab:
                continue

            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        # create cache file so we can load it quicker the next time
        print('Saving vectors to {}'.format(cache_path))
        torch.save((embeddings, emb_size), cache_path)

    # load embeddings from cache
    else:
        print('Loading embeddings from {}'.format(cache_path))
        embeddings, emb_size = torch.load(cache_path)

    return embeddings, emb_size