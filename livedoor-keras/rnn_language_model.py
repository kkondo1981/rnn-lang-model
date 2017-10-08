# -*- coding: utf-8 -*-
"""
RNN言語モデル
"""

import tensorflow as tf

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD


def _add_embedding_layer(model, vocab_size, hidden_size,
                         use_dropout=False, keep_prob=0.5):
    """
    単語の分散表現を学習する層を追加
    """
    with tf.device("/cpu:0"):
        model.add(Embedding(vocab_size, hidden_size))
    # needs dropdout only when training
    if use_dropout:
        model.add(Dropout(1 - keep_prob))


def _add_rnn_layer(model, hidden_size, num_layers,
                   use_dropout=False, keep_prob=0.5,
                   cell_type='LSTM'):
    """
    RNN層を追加。cell_typeでLSTMとGRUを選択可能。
    """
    dropout = 1.0 - keep_prob if use_dropout else 0.0
    for i in range(num_layers):
        if cell_type == 'LSTM':
            model.add(LSTM(hidden_size, dropout=dropout, return_sequences=True))
        elif cell_type == 'GRU':
            model.add(GRU(hidden_size, dropout=dropout, return_sequences=True))
        else:
            raise ValueError("'cell_type' other than 'LSTM' or 'GRU' is not allowed.")


def _add_softmax(model, vocab_size):
    """
    softmax出力層を追加。

    Arguments:
    - vocab_size: 語彙数
    """
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(TimeDistributed(Activation('softmax')))


class RNNLanguageModel(object):
    """
    RNN言語モデル

    内部状態を有し、単語を入力すると次の単語を出力する。
    単語入力の度に内部状態が更新され、長期記憶を保持する。
    （本実装ではLSTMを使用）

    モデルを学習させる上では、計算グラフを静的に展開する際に
    再帰処理を展開する必要から、固定長（config.num_steps）の
    単語列から次の単語列を出力する際の損失最小化を図る。
    （Truncated Backpropagation）
    """

    def __init__(self, config, model_path_prefix=None):
        if model_path_prefix is None:
            # 各種定数（隠れ層次元、語彙数）
            hidden_size = config.hidden_size
            vocab_size = config.vocab_size

            # Dropout処理の設定
            use_dropout =  config.keep_prob < 1
            keep_prob = config.keep_prob

            # モデル構築
            m = Sequential()
            _add_embedding_layer(m, vocab_size, hidden_size,
                                 use_dropout, keep_prob)
            _add_rnn_layer(m, hidden_size, config.num_layers,
                           use_dropout, keep_prob, config.rnn_cell)
            _add_softmax(m, vocab_size)
        else:
            # load model structure to YAML file
            f = open(model_path_prefix + '_nn_model.yaml', 'r')
            m = model_from_yaml(f.read())
            f.close()

            # load model weights to HDF5 file
            m.load_weights(model_path_prefix + '_nn_weights.hdf5')

        optimizer = SGD(lr=config.learning_rate, clipnorm=config.max_grad_norm)
        m.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

        self._model = m
        self._optimizer = optimizer


    def save(self, prefix):
    # save model structure to YAML file
        f = open(prefix + '_nn_model.yaml', 'w')
        f.write(self._model.to_yaml())
        f.close()

        # save model weights to HDF5 file
        self._model.save_weights(prefix + '_nn_weights.hdf5')


    # accessors

    @property
    def model(self):
        return self._model


    @property
    def optimizer(self):
        return self._optimizer
