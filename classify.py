import os
import json
import torch
from torch import nn
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from datasets.preprocess import get_clean_text
from datasets.info import get_label_map
from datasets.dataloader import load_data
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path to the checkpoint
checkpoint_path = '/Users/zou/Desktop/Text-Classification/checkpoints/checkpoint_fasttext.pth.tar'

# pad limits
# only makes sense when model_name = 'han'
sentence_limit_per_doc = 15
word_limit_per_sentence = 20
# only makes sense when model_name != 'han'
word_limit = 200


'''
preprocess a document into a hierarchial representation

input param:
    document (str): a document in text form
    word_map: word2ix map

return: 
    encoded_doc: pre-processed tokenized document
    sentences_per_doc: document length
    words_per_each_sentence: sentence lengths
'''
def prepro_doc(document, word_map):

    # tokenizers
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    # a list to store the document tokenized into words
    doc = list()

    # tokenize document into sentences
    sentences = list()
    for paragraph in get_clean_text(document).splitlines():
        sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

    # tokenize sentences into words
    for s in sentences[:sentence_limit_per_doc]:
        w = word_tokenizer.tokenize(s)[:word_limit_per_sentence]
        if len(w) == 0:
            continue
        doc.append(w)

    # number of sentences in the document
    sentences_per_doc = len(doc)
    sentences_per_doc = torch.LongTensor([sentences_per_doc]).to(device)  # (1)

    # number of words in each sentence
    words_per_each_sentence = list(map(lambda s: len(s), doc))
    words_per_each_sentence = torch.LongTensor(words_per_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(
            map(lambda w: word_map.get(w, word_map['<unk>']), s)
        ) + [0] * (word_limit_per_sentence - len(s)), doc)
    ) + [[0] * word_limit_per_sentence] * (sentence_limit_per_doc - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    return encoded_doc, sentences_per_doc, words_per_each_sentence


'''
preprocess a sentence

input param:
    text (str): a sentence in text form
    word_map: word2ix map

return: 
    encoded_sent: pre-processed tokenized sentence
    words_per_sentence: sentence length
'''
def prepro_sent(text, word_map):
    
    # tokenizers
    word_tokenizer = TreebankWordTokenizer()

    # tokenize sentences into words
    sentence = word_tokenizer.tokenize(text)[:word_limit]

    # number of words in sentence
    words_per_sentence = len(sentence)
    words_per_sentence = torch.LongTensor([words_per_sentence]).to(device)  # (1)

    # encode sentence with indices from the word map
    encoded_sent = list(
        map(lambda w: word_map.get(w, word_map['<unk>']), sentence)
    ) + [0] * (word_limit - len(sentence))
    encoded_sent = torch.LongTensor(encoded_sent).unsqueeze(0).to(device)

    return encoded_sent, words_per_sentence


'''
classify a text with the given model

input param:
    document: a document in text form
    model: a trained model
    model_name: model name
    dataset_name: dataset name
    word_map: word2ix map

return: 
    prediction: the predicted category with its probability
'''
def classify(document, model, model_name, dataset_name, word_map):

    _, rev_label_map = get_label_map(dataset_name)

    if model_name in ['han']:
        # preprocess document
        encoded_doc, sentences_per_doc, words_per_each_sentence = prepro_doc(document, word_map)
        # run through model
        scores, word_alphas, sentence_alphas = model(
            encoded_doc, 
            sentences_per_doc,
            words_per_each_sentence
        )  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    else:
        # preprocess sentence
        encoded_sent, words_per_sentence = prepro_sent(document, word_map)
        # run through model
        scores = model(encoded_sent, words_per_sentence)

    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim = 0)  # (n_classes)
    
    # find best prediction and its probability
    score, prediction = scores.max(dim = 0)

    prediction = 'Category: {category}, Probability: {score:.2f}%'.format(
        category = rev_label_map[prediction.item()], 
        score = score.item() * 100
    )
    return prediction

    # word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    # sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    # words_per_each_sentence = words_per_each_sentence.squeeze(0)  # (n_sentences)

    # return doc, scores, word_alphas, sentence_alphas, words_per_each_sentence


if __name__ == '__main__':
    
    text = 'How do computers work? I have a CPU I want to use. But my keyboard and motherboard do not help.\n\n You can just google how computers work. Honestly, its easy.'
    # text = 'But think about it! It\'s so cool. Physics is really all about math. what feynman said, hehe'
    # text = "I think I'm falling sick. There was some indigestion at first. But now a fever is beginning to take hold."
    # text = "I want to tell you something important. Get into the stock market and investment funds. Make some money so you can buy yourself some yogurt."
    # text = "You know what's wrong with this country? republicans and democrats. always at each other's throats\n There's no respect, no bipartisanship."

    # load model and word map
    model, model_name, _, dataset_name, word_map, _ = load_checkpoint(checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # visualize_attention(*classify(text, model, model_name, dataset_name, word_map))
    prediction = classify(text, model, model_name, dataset_name, word_map)
    print(prediction)