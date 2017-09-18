# -*- coding: utf-8 -*-
"""
RNNの入力を生成する計算グラフを作成する機能を実装。

raw_data全体をRNNに1度に学習させる単位（入力x, 出力y）ごとに分解し、順次出力させる。

1エポックでraw_data全体を学習させるが、ここでは以下のようにミニバッチを作成。
（raw_dataは長さdata_lenの単語列、batch_size=20, num_steps=35 *Mediumモデル）

- raw_dataをbatch_size個のミニバッチに分割
  （各バッチは長さbatch_len = data_len // batch_sizeの連続する単語列）
- 各バッチをnum_stepsごとのブロックに分割（block_0, block_1, ...）
- i（=0, 1, ...）番目の学習で、各バッチにつきi番目のブロックをサンプリング
  入力：各バッチの第iブロックを行方向に連結した batch_size x num_steps 行列
  出力：各バッチの第iブロックを1つ次の単語にシフトしたものを行方向に連結した
        batch_size x num_steps 行列

具体的な処理は_xy_producer()を参照
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
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.x, self.y = _xy_producer(data, batch_size, num_steps, name=name)
