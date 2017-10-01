# -*- coding: utf-8 -*-
"""
RNN言語モデルの入力
"""

import numpy as np


def _get_xy(raw_data, num_steps, vocab_size):
    # raw_data（=IDのリスト）から1次元のndarrayを作成
    raw_data = np.array(raw_data)
    data_len = len(raw_data)

    # raw_dataをn x num_stepsの行列にreshape
    # - n: sample数
    n = (data_len - 1) // num_steps
    xy_size = n * num_steps

    x = np.reshape(raw_data[:xy_size], (n, num_steps))

    # The third dimension is necessary to use TimeDistributed
    y = np.reshape(raw_data[1:(xy_size + 1)], (n, num_steps, 1))

    return x, y


class RNNLanguageModelInput(object):
    """
    RNN言語モデルの入力作成

    - self.x: サイズ n x num_steps の入力行列
    - self.y: self.xと同サイズの出力行列
    where n = (len(data) - 1) // num_steps

    - raw_dataは長さdata_lenの単語IDのベクトル
        raw_data = [3, 4, 2, 10, 8, 20, 5, 5, 7, 8, 9, 1, 13, 6, 9]

    - raw_dataのnum_stepsごとのブロックから(x, y)を作成
      ex.)
         x, y = ([[3, 4, 2], [10, 8, 20], [5, 5, 7], [8, 9, 1]],
                 [[4, 2, 10], [8, 20, 5], [5, 7, 8], [9, 1, 13]])
    """

    def __init__(self, config, data):
        self.num_steps = num_steps = config.num_steps
        self.vocab_size = vocab_size = config.vocab_size
        self.x, self.y = _get_xy(data, num_steps, vocab_size)
