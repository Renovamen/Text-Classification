# Bi-LSTM + Attention

This folder contains the implemention of Attention-Based Bi-LSTM proposed in paper:

**Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.** *Peng Zhou, et al.* ACL 2016. [[Paper]](https://www.aclweb.org/anthology/P16-2034.pdf)

&nbsp;

## Overview

![HAN](../../docs/img/AttBiLSTM.png)

- It uses **element-wise sum** (instead of concatenation) to combine the forward and backward pass outputs of bidirectional LSTM.


&nbsp;

## Performance

| Dataset | Test Accuracy (%) | Training Time per Epoch (GTX 2080 Ti) |
| :-----: | :---------------: | :-----------------------------------: |
| AG News |       91.2        |                  58s                  |
| DBpedia |       98.9        |                 3.8m                  |