# TextCNN

This folder contains the implemention of Hierarchical Attention Networks proposed in paper:

**Convolutional Neural Networks for Sentence Classification.** *Yoon Kim.* EMNLP 2014. [[Paper]](https://www.aclweb.org/anthology/D14-1181.pdf) [[Code]](https://github.com/yoonkim/CNN_sentence)

It applies CNN to text classification problem.

&nbsp;
## Overview

![TextCNN](../../docs/img/TextCNN.png)

- Two channels of word vectors: a **static** one (frozen during backprop) and a **non-static** one (fine-tuned via backprop)

- **Convolution:** Multiple filters are used to obtain multiple features, one feature is extracted from one filter.

- **Max-overtime pooling** is used to capture the most important feature corresponding to each particular filter

- **Fully connected + dropout + softmax** for computing the probability distribution of all classes.

&nbsp;
## Performance

|    Dataset    | Test Accuracy (%) | Training Time per Epoch (GTX 2080 Ti) |
| :-----------: | :---------------: | :-----------------------------------: |
|    AG News    |       92.2        |                  24s                  |
|    DBpedia    |       98.5        |                 100s                  |
| Yahoo Answers |       72.8        |                  4m                   |