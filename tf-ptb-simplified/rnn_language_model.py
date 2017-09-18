# -*- coding: utf-8 -*-
"""
RNNで文章を生成する計算モデルを作成する機能を実装。

連続するnum_steps個の単語列(w_t, ..., w_{t+num_steps-1})から、添字を
1つシフトした(w_{t+1}, ..., w_{t+num_steps})を順番に出力させるRNNで
文章生成をモデル化。
"""

import tensorflow as tf


import loss


class RNNLanguageModel(object):
    """The RNN Language model."""

    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # 語彙数 x 隠れ層次元 の埋め込み行列からのlookup処理
        # 埋め込み行列は"embedding"と名前付けされているので、
        # モデル間で自動的に共有されている模様。
        #
        # lookup後のテンソルのshapeは batch_size x num_steps x hidden_size
        #
        # なお、id >= 語彙数(vocab_size)の場合には語彙数未満と
        # なるように適当に丸められてしまう=他の単語と同一視される
        # ため、要注意（デフォルトではvocab_sizeで割った剰余で分類）。
        # - See https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
        #
        # なお、GPU節約するため？にlookup処理はcpuに乗せている模様。
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            x_distributed = tf.nn.embedding_lookup(embedding, input_.x)

        # 埋め込み行列のDropout処理。学習時のみ必要。
        if is_training and config.keep_prob < 1:
            x_distributed = tf.nn.dropout(x_distributed, config.keep_prob)

        # RNN構築。tf.contribにLSTMセルの実装があるのでこれを使用。Dropoutも可能。
        # LSTMセルの実装は右記論文依拠: https://arxiv.org/abs/1409.2329
        cell = tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

        # LSTMセルをnum_layers個だけ積み重ねる。
        #
        # この書き方だとMultiRNNCellの第一引数に与えるリストの各要素は同一のLSTMBlockCellを
        # 指すことになるが、思うがサンプルコードを見る限りこれでOK.
        # （下記URLを見る限り、cellは単なるcallable objectで、状態は引数として与えられる模様）
        # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/rnn/python/ops/lstm_ops.py#L379-L414
        cell = tf.contrib.rnn.MultiRNNCell(
            [cell for _ in range(config.num_layers)], state_is_tuple=True)

        # RNNの初期状態をバッチ数 x 隠れ層次元 のゼロ行列に設定
        # （各サンプルごとに、初期状態ゼロから学習させる）
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)

        # RNNの計算処理を展開して計算グラフ化。
        #
        # RNNの隠れ層の状態state = self._initial_stateから初めて
        # x_distributed[:, time_step, :] for time_step = 0, 1, .., num_steps - 1
        # を順番に入力し、状態を更新していく。
        #
        # 下記コードと意味は同じ（チュートリアル目的で露わにループを使うコードとしている模様）。
        #   x_distributed = tf.unstack(x_distributed, num=num_steps, axis=1)
        #   outputs, state = tf.contrib.rnn.static_rnn(cell, x_distributed, initial_state=self._initial_state)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()  # 変数の再利用に必要
                (cell_output, state) = cell(x_distributed[:, time_step, :], state)
                outputs.append(cell_output)

        # 分かりにくいが、線形変換を適用するためにRNNの出力をサイズ _ x hidden_size
        # 行列にしておく必要があるため、以下の順序で行列操作を行う。
        #
        # outputs : サイズ batch_size x hidden_size 行列の長さnum_stepsのPythonリスト
        # => tf.concat(outputs, 1) : サイズ batch_size x (hidden_size * num_steps) 行列
        # => output = tf.reshape(..) : サイズ (batch_size * num_steps) x hidden_size 行列
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

        # softmax層に入力する前に線形変換を適用。wもbも名前付きなのでモデル間で自動的に
        # 共有されている模様。
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # 入力と同型（ただし、axis=2は埋め込み次元数ではなく語彙数）のテンソルに変換。
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # クロスエントロピー計算。
        # 結果は長さnum_stepsの1階テンソル（バッチ方向は平均化）
        loss = loss.rnn_loss(logits, y)

        # コスト更新（コストは勾配計算の目的関数として使用）
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        # 以下は学習時専用
        if is_training:
            # 学習レート
            self._lr = tf.Variable(0.0, trainable=False)

            # 更新可能な変数
            tvars = tf.trainable_variables()

            # 勾配計算
            grads = tf.gradients(self._cost, tvars)

            # 勾配のL2ノルムを制限（こうしないと学習がうまくいかない模様）
            grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

            # 再急降下法で最適化
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            # 学習レートの更新
            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)


    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name

        # exportする演算のdictに目的関数を追加
        ops = {'{}/{}'.format(self._name, "cost"): self._cost}

        # self._is_training=Trueの場合には、learning rate関係も追加
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)

        # opsの各要素をコレクションに追加
        for name, op in ops.items():
            tf.add_to_collection(name, op)

        # RNNの初期状態をコレクションに追加
        self._initial_state_name = '{}/{}'.format(self._name, "initial")
        for state_tuple in self._initial_state:
            # c, hはLSTMStateTupleの各要素のエイリアス
            # https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/rnn/LSTMStateTuple
            tf.add_to_collection(self._initial_state_name, state_tuple.c)
            tf.add_to_collection(self._initial_state_name, state_tuple.h)

        # RNNの最終状態をコレクションに追加
        self._final_state_name = '{}/{}'.format(self._name, "final")
        for state_tuple in self._final_state:
            # c, hはLSTMStateTupleの各要素のエイリアス
            # https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/rnn/LSTMStateTuple
            tf.add_to_collection(self._final_state_name, state_tuple.c)
            tf.add_to_collection(self._final_state_name, state_tuple.h)


    def _import_state_tuples(state_tuples, name, num_replicas):
        restored = []
        for i in range(len(state_tuples) * num_replicas):
            c = tf.get_collection_ref(name)[2 * i + 0]
            h = tf.get_collection_ref(name)[2 * i + 1]
            restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        return tuple(restored)


    def import_ops(self, num_gpus):
        """Imports ops from collections."""
        # opsを復元
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
        self._cost = tf.get_collection_ref('{}/{}'.format(self._name, "cost"))[0]

        # RNNの初期状態、最終状態を復元
        num_replicas = num_gpus if self._name == "Train" else 1
        self._initial_state = _import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = _import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)


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


    @property
    def initial_state_name(self):
        return self._initial_state_name


    @property
    def final_state_name(self):
        return self._final_state_name
