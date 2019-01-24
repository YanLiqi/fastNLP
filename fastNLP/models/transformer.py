# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
import fastNLP.modules.encoder as encoder
import fastNLP.modules.decoder as decoder
from fastNLP.modules.aggregator import PositionalEncoding
from tensorboardX import SummaryWriter

class Transformer(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, src_vocab_size,
                 src_max_seq_len,
                 tgt_vocab_size,
                 tgt_max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 dropout=0.5):
        super(Transformer, self).__init__()

        # no support for pre-trained embedding currently
        """
        self.embed = encoder.Embedding(embed_num, embed_dim)

        self.conv_pool = encoder.ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        """
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len
        self.src_seq_embedding = nn.Embedding(src_vocab_size + 1, model_dim, padding_idx=0)
        self.src_pos_embedding = PositionalEncoding(model_dim, src_max_seq_len)
        self.tgt_seq_embedding = nn.Embedding(tgt_vocab_size + 1, model_dim, padding_idx=0)
        self.tgt_pos_embedding = PositionalEncoding(model_dim, tgt_max_seq_len)
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)

        self.enc = encoder.TransformerEncoder(num_layers=num_layers,
                                              input_size=model_dim,
                                              output_size=model_dim,
                                              key_size=int(model_dim/num_heads),
                                              value_size=int(model_dim/num_heads),
                                              num_atte=num_heads)
        self.dec = decoder.TransformerDecoder(num_layers=num_layers,
                                              input_size=model_dim,
                                              output_size=model_dim,
                                              key_size=int(model_dim/num_heads),
                                              value_size=int(model_dim/num_heads),
                                              num_atte=num_heads)

        self.linear = encoder.Linear(model_dim, tgt_vocab_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, word_seq, translated_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        # x = self.embed(word_seq)  # [N,L] -> [N,L,C]
        # x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.src_seq_embedding(word_seq)
        x_len = [self.src_max_seq_len for seq in word_seq]
        x += self.src_pos_embedding(x_len)
        y = self.tgt_seq_embedding(translated_seq)
        y_len = [self.tgt_max_seq_len for seq in translated_seq]
        y += self.tgt_pos_embedding(y_len)

        x = self.src_dropout(x)
        y = self.tgt_dropout(y)


        x = self.enc(x)

        # y = self.embed(translated_seq)
        # y = self.dropout(y)
        x = self.dec(y, x)
        x = self.linear(x)
        # x = self.softmax(x)
        x = x.permute(0, 2, 1)
        return {'pred': x}

    def predict(self, word_seq, translated_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return predict: dict of torch.LongTensor, [batch_size, seq_len]
        """
        print("input:", word_seq)
        print("target:", translated_seq)
        output = self(word_seq, translated_seq)
        _, predict = output['pred'].max(dim=1)
        print("predict:", predict)
        return {'pred': predict}

