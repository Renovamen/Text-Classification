# HAN (Hierarchical Attention Network)

This folder contains the implemention of Hierarchical Attention Networks proposed in paper:

**Hierarchical Attention Networks for Document Classification.** *Zichao Yang, et al.* NAACL 2016. [[Paper]](https://www.aclweb.org/anthology/N16-1174.pdf)

Here are my [notes](https://renovamen.ink/2020/06/10/text-classification-papers/#hierarchical-attention-network) (Chinese). 

&nbsp;

## Overview

![HAN](../../docs/img/HAN.png)

Hierarchical Attention Network constructs a hierarchical document representation by first building representations of sentences and then aggregating those into a document representation.

It has two levels of attention mechanisms applied at the word-level and sentence-level.


- **Word Encoder.** An attentional bidirectional GRU encoding the words and form the sentence vectors.

- **Word Attention.** A one-layer MLP followed with a softmax layer for calculating importance weights over the words.
  
- **Sentence Encoder.** Also an attentional bidirectional GRU encoding the sentences and form the document vectors.
  
- **Sentence Attention.** Also a one-layer MLP followed with a softmax layer for calculating importance weights over the sentences.

- **Document Classification.** A softmax function for computing the probability distribution of all classes.

&nbsp;

## Performance


|    Dataset    | Test Accuracy (%) | Training Time per Epoch (GTX 2080 Ti) |
| :-----------: | :---------------: | :-----------------------------------: |
|    AG News    |       92.7        |                  52s                  |
|    DBpedia    |       97.9        |                  73s                  |
| Yahoo Answers |       74.5        |                 2.7m                  |