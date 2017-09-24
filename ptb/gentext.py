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

上記以外の構成で実行する場合には、適宜LOGDIR_PATH, SAVE_PATHの値を
修正して実行すること。

"""

import numpy as np
import tensorflow as tf

import config as conf
import raw_data
import learn


# PATHs
SEED_WORDS_PATH = './ptb/seed_words.txt'
LOGDIR_PATH = './log/tf-ptb/'
SAVE_PATH = './model/'


word_to_id = raw_data.get_word_to_id()
id_to_word = {v: k for k, v in word_to_id.items()}
seed_words = raw_data._file_to_word_ids(SEED_WORDS_PATH, word_to_id)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def generate_text(session, model, add_words_len, temperature=1.0):
    x = np.zeros((1, 1), dtype=tf.int32)
    state = session.run(model.initial_state)
    probs = None

    output = []

    for word_id in seed_words:
        output.append(word_id)
        x[0, 0] = word_id
        feed_dict = {model.input_data.x: x, model.initial_state: state}
        state, probs = session.run([model.final_state, model.probs], feed_dict)

    if not probs:
        raise RuntimeError('Runtime Error: seed_words is empty or not in dict.')

    for _ in range(add_words_len):
        word_id = sample(probs[0, :], temperature)
        output.append(word_id)

        x[0, 0] = word_id
        feed_dict = {model.input_data.x: x, model.initial_state: state}
        state, probs = session.run([model.final_state, model.probs], feed_dict)

    return output


def main(_):
    if len(seed_words) == 0:
        raise RuntimeError('RuntimeError: No seed words or all seed words are not in dict.')

    # 各種設定（1語ずつ処理するのでeval_configを使用）
    _, gen_config = conf.get_config()

    with open(LOGDIR_PATH + 'gentext.txt', 'w') as f:

        # 計算グラフの構築
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

            # 文章生成用モデル構築・学習済パラメータの復元
            model = learn.create_model('TextGen', gen_config, [0], initializer)
            saver = tf.train.Saver()
            saver.restore(session, SAVE_PATH + 'tf-ptb')

            # 文章生成
            output = generate_text(session, model, 100, 1.0)
            s = ' '.join([id_to_word[word_id] for word_id in output])
            s = s.replace('<eos>', '\n')
            f.write(s)


if __name__ == "__main__":
    tf.app.run()
