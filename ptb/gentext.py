# -*- coding: utf-8 -*-
"""
RNN言語モデルによる文章生成

[Usage] python ./ptb/gentext.py
- 実行前にlearn.pyでモデルを学習・保存しておくこと。

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── ptb
｜   └── gentext.py    : this script
｜
└── model              : model dir
```

上記以外の構成で実行する場合には、適宜PATHの値を修正して実行すること。

"""


import re

import numpy as np
import tensorflow as tf

import config as conf
import raw_data
from rnn_language_model import RNNLanguageModel as Model
from rnn_language_model_input import RNNLanguageModelInput as Input


# PATHs
SEED_WORDS_PATH = './ptb/seed_words.txt'
LOGDIR_PATH = './log/ptb/'
MODEL_PATH = './model/ptb-51753'


word_to_id = raw_data.get_word_to_id()
id_to_word = {v: k for k, v in word_to_id.items()}
seed_words = raw_data._file_to_word_ids(SEED_WORDS_PATH, word_to_id)


def create_model(mode_name, config, data, initializer):
    """Creates the tensorflow model"""
    with tf.name_scope(mode_name):
        reuse = None
        is_training = False

        input_ = Input(config=config, data=data, name=mode_name+'Input')

        with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
            m = Model(is_training=is_training, config=config, input_=input_)

        return m


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    a = a / (1.0 + 1e-6)  # to avoid error when sum(a[:-1]) > 1.0
    return np.argmax(np.random.multinomial(1, a, 1))


def generate_text(session, model, add_words_len, temperature=1.0):
    x = np.zeros((1, 1), dtype=np.int32)
    state = session.run(model.initial_state)
    probs = None

    output = []

    for word_id in seed_words:
        output.append(word_id)
        x[0, 0] = word_id
        feed_dict = {model.input.x: x, model.initial_state: state}
        state, probs = session.run([model.final_state, model.probs], feed_dict)
        probs = np.reshape(probs, (probs.shape[2]))

    if probs is None:
        raise RuntimeError('Runtime Error: seed_words is empty or not in dict.')

    for _ in range(add_words_len):
        word_id = sample(probs, temperature)
        output.append(word_id)

        x[0, 0] = word_id
        feed_dict = {model.input.x: x, model.initial_state: state}
        state, probs = session.run([model.final_state, model.probs], feed_dict)
        probs = np.reshape(probs, (probs.shape[2]))

    return output


def main(_):
    if len(seed_words) == 0:
        raise RuntimeError('RuntimeError: No seed words or all seed words are not in dict.')

    # 各種設定（1語ずつ処理するのでeval_configを使用）
    _, gen_config = conf.get_config()

    with open(LOGDIR_PATH + 'gentext.txt', 'w') as f:

        # 計算グラフの構築
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-gen_config.init_scale, gen_config.init_scale)

            # 文章生成用モデル構築・学習済パラメータの復元
            model = create_model('TextGen', gen_config, [2], initializer)  # word_id=2 means '<eos>'
            saver = tf.train.Saver()
            saver.restore(session, MODEL_PATH)

            # 文章生成
            for diversity in [1.0, 0.8, 0.6]:
                f.write('\n\n============================================================\n')
                f.write('** generated with diversity {:.2f} **\n'.format(diversity))
                output = generate_text(session, model, 100, diversity)
                s = ' '.join([id_to_word[word_id] for word_id in output])
                s = re.sub(r'\s+<eos>\s+', '\n', s)
                f.write(s)


if __name__ == "__main__":
    tf.app.run()
