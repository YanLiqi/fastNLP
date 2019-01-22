# -*- coding: utf-8 -*-
# /usr/bin/python3.6
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function

import numpy as np
import codecs
import regex
import sys
sys.path.append("../")
from fastNLP import Instance, DataSet, Vocabulary
from fastNLP.models import CNNText
from fastNLP.models import Transformer
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric


class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = '../corpora/train.tags.de-en.de'
    target_train = '../corpora/train.tags.de-en.en'
    source_test = '../corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = '../corpora/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 20  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.





def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= Hyperparams.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= Hyperparams.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= Hyperparams.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), Hyperparams.maxlen], np.int32)
    Y = np.zeros([len(y_list), Hyperparams.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, Hyperparams.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, Hyperparams.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(Hyperparams.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(Hyperparams.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y


def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(Hyperparams.source_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(Hyperparams.target_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets  # (1064, 150)


def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // Hyperparams.batch_size

    i = 0
    ds = DataSet()
    for x in X:
        instance = Instance(word_seq=x.tolist(), translated_seq=Y[i].tolist())
        ds.append(instance)
        i = i + 1
    ds.set_input('word_seq','translated_seq')
    ds.set_target('translated_seq')
    """
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=Hyperparams.batch_size,
                                  capacity=Hyperparams.batch_size * 64,
                                  min_after_dequeue=Hyperparams.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()
    """
    return ds




if __name__ == '__main__':
    """
    data_path = "../tutorials/sample_data/tutorial_sample_dataset.csv"
    ds = DataSet.read_csv(data_path, headers=('raw_sentence', 'label'), sep='\t') 
    
    # 将所有数字转为小写
    ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
    # label转int
    ds.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)


    def split_sent(ins):
        return ins['raw_sentence'].split()


    ds.apply(split_sent, new_field_name='words', is_input=True)

    # 分割训练集/验证集
    train_data, dev_data = ds.split(0.3)
    print("Train size: ", len(train_data))
    print("Test size: ", len(dev_data))

    
    vocab = Vocabulary(min_freq=2)
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])

    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)

    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='translated_seq',
                     is_input=True)
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='translated_seq',
                   is_input=True)

    """

    # Load vocabulary

    train_data = get_batch_data()


    print(train_data[3])
    print(len(train_data))

    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    print(len(de2idx))
    print(len(idx2de))
    print(len(en2idx))
    print(len(idx2en))
    # model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
    model = Transformer(embed_num=len(de2idx), embed_dim=Hyperparams.maxlen, padding=2, dropout=0.1, output_num=len(en2idx))



    trainer = Trainer(model=model,
                      train_data=train_data,
                      loss=CrossEntropyLoss(),
                      )
    trainer.train()
    print('Train finished!')



