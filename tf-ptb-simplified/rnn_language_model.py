# -*- coding: utf-8 -*-
"""
RNNで文章を生成する計算モデルを作成する機能を実装。

連続するnum_steps個の単語列(w_t, ..., w_{t+num_steps-1})から、添字を
1つシフトした(w_{t+1}, ..., w_{t+num_steps})を順番に出力させるRNNで
文章生成をモデル化。
"""

import tensorflow as tf


import loss


def _x_distributed(x_local, vocab_size, hidden_size, use_dropout=False, keep_prob=0.5):
    """
    入力の分散表現を取得する。

    Arguments:
    - x_local: サイズ batch_size x num_steps 行列。各バッチの入力単語リストのID
    - vocab_size: 全語彙数。
      x_localの各要素は[0, vocab_size)の整数であることが期待されており、
      id >= vocab_sizeの場合には適当に丸められてしまうので注意。
    ...

    Return:
    サイズ batch_size x num_steps x hidden_size のテンソル
    - See https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    """
    with tf.device("/cpu:0"):  # GPUを節約？
        embedding = tf.get_variable(
            "embedding", [vocab_size, hidden_size], dtype=tf.float32)
        x_distributed = tf.nn.embedding_lookup(embedding, x_local)

    # needs dropdout only when training
    if use_dropout:
        x_distributed = tf.nn.dropout(x_distributed, keep_prob)

    return x_distributed


def _rnn_cell(hidden_size, num_layers, batch_size, 
              forget_bias=0.0, use_dropout=False, keep_prob=0.5):
    """
    RNNセルを作成(LSTMを使用)。
    LSTMセルの実装は右記論文依拠: https://arxiv.org/abs/1409.2329

    Arguments:
    - hidden_size: 隠れ層の次元
    - num_layers: 隠れ層の数
    - batch_size: バッチ数
    ...

    Returns:
    - cell: the created RNN cell
    - initial_state: the initial state of the cell
    """
    cell = tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=forget_bias)
    if use_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    # RNNの各層を同じcellオブジェクトで初期化するコードだが、これでOK
    # 右記URL参照: https://github.com/tensorflow/tensorflow/issues/7604
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(num_layers)], state_is_tuple=True)

    return cell, cell.zero_state(batch_size, tf.float32)


def _unroll_rnn(cell, initial_state, x, num_steps):
    """
    RNNの隠れ層の状態をinitial_stateから初めて順番にnum_steps数の
    入力を処理する演算グラフを展開する。

    なお、下記コードと意味は同じだが、意味を理解できるように展開している。
      x = tf.unstack(x, num=num_steps, axis=1)
      outputs, state = tf.contrib.rnn.static_rnn(cell, x_distributed, initial_state=self._initial_state)

    Arguments:
    - cell: RNNセル
    - initial_state: RNNセルの初期状態
    - x: 入力行列（batch_size x num_steps x hidden_size）
    - num_steps: 展開ステップ数

    Return:
    - outputs: RNNの各ステップでの出力（batch_size x hidden_size）のリスト（長さ=num_steps）
    - state: RNNの最終状態
    """
    state = initial_state
    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()  # 変数の再利用に必要
            (cell_output, state) = cell(x[:, time_step, :], state)
            outputs.append(cell_output)
    return outputs, state


def _logits(rnn_outputs, hidden_size, batch_size, num_steps, vocab_size):
    """
    RNNの出力をsoftmaxとる前の線形変換を適用し、unscaled log probabilityテンソルを計算。

    線形変換を適用するためにRNNの出力をサイズ _ x hidden_size 行列にreshapeする:
    rnn_outputs : サイズ batch_size x hidden_size 行列の長さnum_stepsのPythonリスト
      => tf.concat(rnn_outputs, 1) : サイズ batch_size x (hidden_size * num_steps) 行列
      => rnn_output = tf.reshape(..) : サイズ (batch_size * num_steps) x hidden_size 行列

    Arguments:
    - rnn_outputs: RNNセルの出力
    - hidden_size: 隠れ層の次元
    - batch_size: バッチ数
    - num_steps: RNNのステップ数
    - vocab_size: 語彙数

    Return:
    - logits: サイズ batch_size x num_steps x vocab_size のテンソル。
              各要素はunscaled lob probability
    """
    rnn_output = tf.reshape(tf.concat(rnn_outputs, 1), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.nn.xw_plus_b(rnn_output, softmax_w, softmax_b)
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
    return logits


def _learning_rate():
    """
    学習係数（learning rate）を表す変数およびその更新操作

    Returns:
    - lr: 学習係数（learning rate）
    - new_lr: 更新後の学習係数用プレースホルダ－
    - lr_update: 学習係数の更新操作
    """
    lr = tf.Variable(0.0, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    lr_update = tf.assign(lr, new_lr)
    return lr, new_lr, lr_update


def _train_op(cost, lr, max_grad_norm):
    """
    Train用の最適化操作。

    Arguments:
    - cost: 損失関数
    - lr: 学習係数
    - max_grad_norm: 勾配のL2ノルムを（こうしないと学習がうまくいかない模様）
    """
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(lr)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    return train_op


class RNNLanguageModel(object):
    """The RNN Language model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        # モデルの入出力
        x = self._input.x
        y = self._input.y

        # 各種定数（隠れ層次元、バッチ数、ステップ数、語彙数）
        hidden_size = config.hidden_size
        batch_size = config.batch_size
        num_steps = config.num_steps
        vocab_size = config.vocab_size

        # 学習時はDropout処理を使用
        use_dropout = is_training and config.keep_prob < 1
        keep_prob = config.keep_prob

        # 計算グラフを構築
        x_distributed = _x_distributed(x, vocab_size, hidden_size,
                                       use_dropout, keep_prob)
        cell, initial_state = _rnn_cell(hidden_size, config.num_layers,
                                        batch_size, forget_bias=0.0,
                                        use_dropout=use_dropout, keep_prob=keep_prob)
        outputs, final_state = _unroll_rnn(cell, initial_state, x_distributed, num_steps)
        logits = _logits(outputs, hidden_size, batch_size, num_steps, vocab_size)
        cost = tf.reduce_sum(loss.rnn_loss(logits, y))  # クロスエントロピー

        # 外部から参照する変数をセット
        self._initial_state = initial_state
        self._final_state = final_state
        self._cost = cost

        # Train操作
        if is_training:
            self._lr, self._new_lr, self._lr_update = _learning_rate()
            self._train_op = _train_op(self._cost, self._lr, config.max_grad_norm)


    def assign_lr(self, session, lr_value):
        """Updates the learning rate with 'lr_value'."""
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    # accessors

    @property
    def input(self):
        return self._input


    @property
    def initial_state(self):
        return self._initial_state


    @property
    def cost(self):
        return self._cost


    @property
    def final_state(self):
        return self._final_state


    @property
    def lr(self):
        return self._lr


    @property
    def train_op(self):
        return self._train_op
