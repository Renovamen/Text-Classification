# fastText

This folder contains the implemention of fastText proposed in:

**Bag of Tricks for Efficient Text Classification.** *Armand Joulin, et al.* EACL 2017. [[Paper]](https://www.aclweb.org/anthology/E17-2068.pdf) [[Code]](https://github.com/facebookresearch/fastText)

It is a simple text classification model with the ability to give comparable performance to much complex neural network based models.


&nbsp;

## Overview

![fastText](../../docs/img/fastText.png)

1. **Embedding.** Embed **bags of ngram** to word embeddings.

2. **Hidden.** Average word embeddings into sentence embeddings.

3. **Output.** Feed sentence embeddings into a linear classifier with 10 hidden units.

4. **Classification.** Feed the output into a **hierarchical softmax** function to compute the probability distribution of all classes.


&nbsp;

## Implementation Details

Here are somethings different from the original paper:

- I used normal softmax instead of hierarchical softmax
- I didn't use ngram embeddings
- I used pre-trained GloVe embeddings for embedding words


&nbsp;

## Performance

|    Dataset    | Test Accuracy (%) | Training Time per Epoch (RTX 2080 Ti) |
| :-----------: | :---------------: | :-----------------------------------: |
|    AG News    |       91.6        |                  8s                   |
|    DBpedia    |       97.9        |                  25s                  |
| Yahoo Answers |       66.7        |                  41s                  |
