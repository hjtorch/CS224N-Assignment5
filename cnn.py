#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embedding_size, word_embedding_size, kernel_size=5):
        super(CNN, self).__init__()

        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.cov1d_layer = nn.Conv1d(in_channels=self.char_embedding_size,
                                     out_channels=self.word_embedding_size,
                                     kernel_size=kernel_size,
                                     bias=True)

    def forward(self, input: torch.Tensor):
        """
        :param input: (Batch_size * Sentense_lenth, char_embedding_size, max_word_length)
        :return: word embedding vectors for sentense. (Batches_size*sentense_length, word_embedding_size)
        """
        conv = self.cov1d_layer(input)
        relu = F.relu(conv)
        out = torch.max(relu, dim=2)[0]
        return out


    ### END YOUR CODE

