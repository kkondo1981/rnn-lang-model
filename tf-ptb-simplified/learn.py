# -*- coding: utf-8 -*-
"""
RNN言語モデルの学習処理

[Usage] python ./tf-ptb-simplified/learn.py

右記URLのMediumモデルに相当: https://arxiv.org/pdf/1409.2329.pdf

AWSのGPUインスタンス（p2.xlarge）で3時間程度で学習終了(2017/9/23)。
PerplexityはTrain, Validともに80台程度を達成。

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── tf-ptb-simplified
｜   └── learn.py      : this script
｜
├── log
｜   └── tf-ptb        : log dir
｜
└── model              : model dir
```

上記以外の構成で実行する場合には、適宜LOGDIR_PATH, SAVE_PATHの値を
修正して実行すること。

"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib.tensorboard.plugins import projector

import config as conf
import raw_data
from rnn_language_model import RNNLanguageModel as Model
from rnn_language_model_input import RNNLanguageModelInput as Input


# PATHs
LOGDIR_PATH = './log/tf-ptb/'
SAVE_PATH = './model/'


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs, iters = 0.0, 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('{:.3f} perplexity: {:.3f} speed: {:.0f} wps'
                  .format(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                          iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def create_model(mode_name, config, data, initializer):
    """Creates the tensorflow model"""
    with tf.name_scope(mode_name):
        reuse = None if mode_name == 'Train' else True
        is_training = mode_name == 'Train'

        input_ = Input(config=config, data=data, name=mode_name+'Input')

        with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
            m = Model(is_training=is_training, config=config, input_=input_)

        if mode_name == 'Train':
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)
        elif mode_name == 'Valid':
            tf.summary.scalar("Validation Loss", m.cost)
        # elif mode_name == 'Test':
        #   ;

        return m


def set_embedding_visualization(name):
    """Enables visualization of the embedding matrix and metadata"""
    vocab_path = LOGDIR_PATH + 'vocab.tsv'

    print('Saving vocab file to {}.'.format(vocab_path))
    raw_data.save_vocab(vocab_path)

    with tf.variable_scope("Model", reuse=True):
        embedding_var = tf.get_variable(name)
        summary_writer = tf.summary.FileWriter(LOGDIR_PATH)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.abspath(vocab_path)
        projector.visualize_embeddings(summary_writer, config)


def main(_):
    # 入力ファイルからraw dataを読み込む
    train_data, valid_data, test_data, _ = raw_data.get_raw_data()

    # 各種設定（config: Train&Valid用, eval_config: Test用）
    config, eval_config = conf.get_config()

    # 計算グラフの構築
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # 学習用モデル構築
        m = create_model('Train', config, train_data, initializer)
        mvalid = create_model('Valid', config, valid_data, initializer)
        mtest = create_model('Test', eval_config, test_data, initializer)

        # 埋め込み行列の可視化用設定
        set_embedding_visualization('embedding')

        # 学習実行
        sv = tf.train.Supervisor(logdir=LOGDIR_PATH)
        with sv.managed_session() as session:
            for i in range(config.max_epoch):
                lr_decay = config.lr_decay \
                           ** max(i + 1 - config.decreasing_learning_rate_after, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                print('Epoch: {} Learning rate: {:.3f}'.format(i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print('Epoch: {} Train Perplexity: {:.3f}'.format(i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid)
                print('Epoch: {} Valid Perplexity: {:.3f}'.format(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print('Test Perplexity: {:.3f}'.format(test_perplexity))

            print('Saving model to {}.'.format(SAVE_PATH))
            sv.saver.save(session, SAVE_PATH + 'tf-ptb', global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
