# Transformer

This folder contains the implemention of Transformer proposed in paper:

**Attention Is All You Need.** *Ashish Vaswani, et al.* NIPS 2017. [[Paper]]([Paper]) [[Code]](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

Here are my [notes](https://renovamen.ink/2020/07/17/transformer/) of Transformer (Chinese). 

Only the encoder part of Transformer is used. Some modificaitons have been made to make the Transformer work for text classification task.

&nbsp;

## Overview

![Transformer](../../docs/img/Transformer.png)

Transformer is an encoder-decoder architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. It is firstly proposed for machine translation task.

To make the Transformer work for text classification task, here I only use its encoder, and add a fc layer to compute the score for each class.


&nbsp;

## Performance


|    Dataset    | Test Accuracy (%) | Training Time per Epoch (GTX 2080 Ti) |
| :-----------: | :---------------: | :-----------------------------------: |
|    AG News    |       92.2        |                  60s                  |
|    DBpedia    |       98.6        |                  8.2m                 |
| Yahoo Answers |       72.5        |                 14.5m                 |