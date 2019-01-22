# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn

# import torch.nn.functional as F
import fastNLP.modules.encoder as encoder
import fastNLP.modules.decoder as decoder


class Transformer(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, embed_num,
                 embed_dim,
                 kernel_nums=(3, 4, 5),
                 hidden_units_num=512,
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(Transformer, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = encoder.Embedding(embed_num, embed_dim)
        """
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        """
        self.dropout = nn.Dropout(dropout)
        # self.fc = encoder.Linear(sum(kernel_nums), num_classes)

        self.enc = encoder.TransformerEncoder(6, input_size=sum(kernel_nums), output_size=hidden_units_num,key_size=10,value_size=10,num_atte=8)
        self.dec = decoder.TransformerDecoder(6, input_size=sum(kernel_nums), output_size=hidden_units_num,key_size=10,value_size=10,num_atte=8)


    def forward(self, word_seq, translated_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(word_seq)  # [N,L] -> [N,L,C]
        # x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        # x = self.fc(x)  # [N,C] -> [N, N_class]
        x = self.enc(x)

        y = self.embed(translated_seq)
        y = self.dropout(y)
        x = self.dec(y, x)
        return {'pred': x}

    def predict(self, word_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return predict: dict of torch.LongTensor, [batch_size, seq_len]
        """
        output = self(word_seq)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}
