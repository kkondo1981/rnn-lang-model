# -*- coding: utf-8 -*-
"""
RNN言語モデルによる文章生成

[Usage] python ./livedoor/gentext.py
- 実行前にlearn.pyでモデルを学習・保存しておくこと。

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── livedoor
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
LOGDIR_PATH = './log/livedoor/'
MODEL_PATH = './model/livedoor-78273'


word_to_id = raw_data.get_word_to_id()
id_to_word = {v: k for k, v in word_to_id.items()}
eos_id = word_to_id['eos']


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


def is_ascii(char):
    return ord(char) < 128


def generate_sentence(session, model, temperature=1.0, end='。'):
    x = np.zeros((1, 1), dtype=np.int32)
    state = session.run(model.initial_state)

    output = []

    x[0, 0] = eos_id
    while True:
        feed_dict = {model.input.x: x, model.initial_state: state}
        state, probs = session.run([model.final_state, model.probs], feed_dict)
        probs = np.reshape(probs, (probs.shape[2]))

        word_id = sample(probs, temperature)
        word = id_to_word[word_id]
        if word == 'eos':
            print('。', flush=True)
            break

        if len(output) > 0 and is_ascii(output[-1][-1]) and is_ascii(word[0]):
            output.append(' ')
            print(' ', end='', flush=True)

        output.append(word)
        print(word, end='', flush=True)

        x[0, 0] = word_id

    return ''.join(output) + end


def main(_):
    # 各種設定（1語ずつ処理するのでeval_configを使用）
    _, gen_config = conf.get_config()

    # 計算グラフの構築
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-gen_config.init_scale, gen_config.init_scale)

        # 文章生成用モデル構築・学習済パラメータの復元
        model = create_model('TextGen', gen_config, [eos_id], initializer)  # start with 'eos'
        saver = tf.train.Saver()
        saver.restore(session, MODEL_PATH)

        with open(LOGDIR_PATH + 'gentext.txt', 'w') as f:

            # 文章生成
            for temperature in [1.4, 1.2, 1.0, 0.8, 0.6]:
                print('\n\nGenerating the text with temperature {}'.format(temperature), flush=True)
                f.write('\n\n============================================================\n')
                f.write('** generated with temperature {:.2f} **\n'.format(temperature))
                for _ in range(10):
                    output = generate_sentence(session, model, temperature)
                    f.write(output + '\n')


if __name__ == "__main__":
    tf.app.run()
