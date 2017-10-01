# -*- coding: utf-8 -*-
"""
RNN言語モデルの損失関数

tf.contrib.seq2seq.sequence_loss()でクロスエントロピーを計算する処理を、
汎用的に書かれている部分を削って単純化して書き直したもの。

tf.contribの実装は下記URLを参照:
- https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
"""

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def rnn_loss(logits, y):
    """
    クロスエントロピー計算

    Arguments:
    - logits: 事後確率ベクトル（対数確率、正規化前）
         batch_size x num_steps x vocab_size
    - y: 正解ラベル行列
         batch_size x num_steps

    Return:
    - crossent: バッチ方向を平均化した長さnum_stepsの損失ベクトル
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
        crossent /= tf.cast(batch_size, tf.float32)

        return crossent
