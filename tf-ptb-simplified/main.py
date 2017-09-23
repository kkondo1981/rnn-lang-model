# -*- coding: utf-8 -*-
"""
simplified version of 'ptb_word_lm.py' medium model
"""

import time

import numpy as np
import tensorflow as tf

import util

from tensorflow.python.client import device_lib

import config as conf
import raw_data
from rnn_language_model import RNNLanguageModel as Model
from rnn_language_model_input import RNNLanguageModelInput as Input


LOGDIR_PATH = './log/tf-ptb/'
SAVE_PATH = './model/'


def run_epoch(session, model, num_gpus, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
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
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    # GPU数が1より小さい場合には終了
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    num_gpus = len(gpus)
    if num_gpus < 1:
        raise ValueError("Your machine has only %d gpus < 1" % num_gpus)

    # 入力ファイルからraw dataを読み込む
    train_data, valid_data, test_data, _ = raw_data.get_raw_data()

    # RNNの設定を取得。
    # configはTrain, Validモデル用。
    # eval_configはTestモデル用で、バッチもステップも1。
    config = conf.get_config()
    eval_config = conf.get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # 計算グラフの構築
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # Train用モデル構築
        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # Valid用モデル構築
        with tf.name_scope("Valid"):
            valid_input = Input(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        # Test用モデル構築
        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Model(is_training=False, config=eval_config, input_=test_input)

        # （多分）グラフ構造の並列化処理を加えるため、計算グラフを一度export
        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()

        # tensorflowのversionが1.1.0以上で、GPUが複数ある場合には並列化
        soft_placement = False
        if tf.__version__ >= "1.1.0" and num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)


    # 計算グラフの復元
    with tf.Graph().as_default():
        # 計算グラフをimport
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops(num_gpus)

        # Sessionを構築→学習実行
        sv = tf.train.Supervisor(logdir=LOGDIR_PATH)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):  # Mediumモデルでは39
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, num_gpus, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, num_gpus)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, num_gpus)
            print("Test Perplexity: %.3f" % test_perplexity)

            print("Saving model to %s." % SAVE_PATH)
            sv.saver.save(session, SAVE_PATH + 'tf-ptb', global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
