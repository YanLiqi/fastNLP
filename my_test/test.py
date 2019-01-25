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
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Tester


class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = '../corpora/train.tags.de-en.de'
    target_train = '../corpora/train.tags.de-en.en'
    source_test = '../corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = '../corpora/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 32  # alias = N
    logdir = 'logdir'  # log directory

    # model
    maxlen = 5  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 50  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 128  # alias = C. In paper it's 512.
    num_blocks = 3  # number of encoder/decoder blocks. In paper it's 6.
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.1





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
        x = [de2idx.get(word, 1) for word in source_sent.split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in target_sent.split()]
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
    return X, Y, Sources, Targets


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
    return X, Y, Sources, Targets  # (1064, 150)


def get_batch_data():
    # Load train data
    X_train, Y_train, Sources_train, Targets_train = load_train_data()

    # calc total batch count
    num_batch = len(X_train) // Hyperparams.batch_size

    i = 0
    ds_train = DataSet()
    for x in X_train:
        instance = Instance(word=Sources_train[i], traslated=Targets_train[i], word_seq=x.tolist(), translated_seq=Y_train[i].tolist())
        ds_train.append(instance)
        i = i + 1
    ds_train.set_input('word_seq', 'translated_seq')
    ds_train.set_target('translated_seq')

    # Load test data
    X_test, Y_test, Sources_test, Targets_test = load_test_data()

    # calc total batch count
    num_batch = len(X_test) // Hyperparams.batch_size

    i = 0
    ds_test = DataSet()
    for x in X_test:
        instance = Instance(word=Sources_test[i], traslated=Targets_test[i], word_seq=x.tolist(), translated_seq=Y_test[i].tolist())
        ds_test.append(instance)
        i = i + 1
    ds_test.set_input('word_seq', 'translated_seq')
    ds_test.set_target('translated_seq')

    return ds_train, ds_test




if __name__ == '__main__':

    # Load vocabulary

    train_data, dev_data = get_batch_data()

    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    with open("vocab_de.txt", 'w') as f:
        f.write(str(idx2de))
    with open("vocab_en.txt", 'w') as f:
        f.write(str(idx2en))

    print("de_vocab_len:", len(de2idx))
    print("en_vocab_len:", len(en2idx))

    # train_data, dev_data = train_data.split(0.3)
    print(train_data[3])
    print(dev_data[3])
    print("train_data:", len(train_data))
    print("test_data:", len(dev_data))
    # model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
    model = Transformer(src_vocab_size=len(de2idx),
                        src_max_seq_len=Hyperparams.maxlen,
                        tgt_vocab_size=len(en2idx),
                        tgt_max_seq_len=Hyperparams.maxlen,
                        num_layers=Hyperparams.num_blocks,
                        model_dim=Hyperparams.hidden_units,
                        num_heads=Hyperparams.num_heads,
                        dropout=Hyperparams.dropout_rate)

    # print(model)
    trainer = Trainer(model=model,
                      train_data=train_data,
                      dev_data=dev_data,
                      loss=CrossEntropyLoss(),
                      batch_size=Hyperparams.batch_size,
                      save_path='./checkpoint',
                      n_epochs=Hyperparams.num_epochs,
                      metrics=AccuracyMetric()
                      )
    result = trainer.train()
    with open("log.txt", 'a+') as f:
        f.write(result + '\n')
    print(result)
    print('Train finished!')
    tester = Tester(data=dev_data, model=model, metrics=AccuracyMetric(),
                    batch_size=4)
    acc = tester.test()
    print(acc)




