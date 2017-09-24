# -*- coding: utf-8 -*-
"""
RNN言語モデルの入力
"""

import tensorflow as tf


def _xy_producer(raw_data, batch_size, num_steps, name):
    with tf.name_scope(name, values=[raw_data, batch_size, num_steps]):
        # raw_data（=IDのリスト）から1階の整数テンソルを作成。
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        # raw_dataのワード数
        data_len = tf.size(raw_data) 

        # raw_dataをbatch_size x batch_lenの行列にreshape
        # - batch_size: 1バッチのサイズ（Mediumモデルだと20ワード）
        # - batch_len: バッチ数
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        # 1エポックの学習回数
        epoch_size = (batch_len - 1) // num_steps

        # batch_len <= num_stepsだとepoch_sizeがゼロとなるためassertionを入れる
        # control_dependenciesでepoch_sizeをassertionに依存させ、先にassertionを
        # 評価させる。
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        # 以下のコードに対応する計算グラフを構築してreturn
        # for i in range(epoch_size):
        #   x = data[:, i*num_steps:(i+1)*num_steps]
        #   y = data[:, (i*num_steps+1):((i+1)*num_steps+1)]
        #   yield x, y  # ここで(x, y)をRNNに学習させる
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        return x, y


class RNNLanguageModelInput(object):
    """
    RNN言語モデルの入力

    raw_data全体を学習単位ごとに処理する操作
    => 1学習単位（1 unit）は下記変数のペア
    - self.x: サイズ batch_size x num_steps の入力行列
    - self.y: self.xと同サイズの出力行列

    データ全体を処理するには、(self.x, self.y)をepoch_size回evalする
    - epoch_size = ((data_len // batch_size) - 1) // num_steps

    raw_dataの学習単位への分解は、以下の通り。

    - raw_dataは長さdata_lenの単語IDのベクトル
        raw_data = [3, 4, 2, 10, 8, 20, 5, 5, 7, 8, 9, 1, 13, 6, 9]

    - raw_dataをbatch_size個に分割して、data行列を作成:
      ex.)
        data = [[3, 4, 2, 10, 8, 20, 5],  # batch 1
                [5, 7, 8, 9, 1, 13, 6]]   # batch 2

    - 各バッチのnum_stepsごとのブロックから(x, y)を作成
      ex.)
       # learning unit 1
         x, y = ([[3, 4, 2],   # batch 1 for unit 1 (Input)
                  [5, 7, 8]],  # batch 2 for unit 1 (Input)
                 [[4, 2, 10],  # batch 1 for unit 1 (Output)
                  [7, 8, 9]])  # batch 2 for unit 1 (Output)

       # learning unit 2
         x, y = ([[10, 8, 20], # batch 1 for unit 2 (Input)
                  [9, 1, 13]], # batch 2 for unit 2 (Input)
                 [[8, 20, 5],  # batch 1 for unit 2 (Output)
                  [1, 13, 6]]) # batch 2 for unit 2 (Output)
    """

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.x, self.y = _xy_producer(data, batch_size, num_steps, name=name)
