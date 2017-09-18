# -*- coding: utf-8 -*-
"""
RNNの損失関数（クロスエントロピー）を実装。

tf.contrib.seq2seq.sequence_loss()は汎用的になっているために読みにくいので、
rnn_language_modelに必要な形に単純化して再実装したもの。

See https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def rnn_loss(logits, y):
    """
    softmax用損失関数。
    logitsは対数確率（正規化前）、yは正解ラベル行列。

    期待するサイズは以下の通り：
      logits: batch_size x num_steps x vocab_size
      y: batch_size x num_steps

    出力は 長さnum_stepsの1階テンソル（バッチ方向は平均化）
    """
    with ops.name_scope("rnn_loss", values=[logits, y]):
        batch_size = array_ops.shape(logits)[0]
        sequence_length = array_ops.shape(logits)[1]
        num_classes = array_ops.shape(logits)[2]

        # sparse_softmax_cross_entropy_with_logits()に渡すために
        # axis=0, 1をflattenする。
        logits_flat = array_ops.reshape(logits, [-1, num_classes])
        y = array_ops.reshape(y, [-1])

        # sparse_softmax_cross_entropy_with_logits()で損失計算。
        # 結果は長さ(batch_size * num_steps)の1階テンソル。
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits_flat)

        # サイズ batch_size x sequence_length 行列にreshapeした上で
        # axis=0 （バッチ方向）を合計して長さsequence_lenghtの1階テンソルに変換。
        crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
        crossent = math_ops.reduce_sum(crossent, axis=[0])

        # 正規化
        crossent /= batch_size

        return crossent
