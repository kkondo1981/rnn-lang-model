# -*- coding: utf-8 -*-
"""
RNN言語モデルによる尤度比較

[Usage] python ./livedoor/likelihood.py
- 実行前にlearn.pyでモデルを学習・保存しておくこと。

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── livedoor
｜   └── likelihood.py    : this script
｜
└── model              : model dir
```

上記以外の構成で実行する場合には、適宜PATHの値を修正して実行すること。

"""

import math

import MeCab
import numpy as np
import tensorflow as tf

import config as conf
import raw_data
from rnn_language_model import RNNLanguageModel as Model
from rnn_language_model_input import RNNLanguageModelInput as Input


# PATHs
LOGDIR_PATH = './log/livedoor/'
MODEL_PATH = './model/livedoor-78273'


SEED_SENTENCE = '花屋の店先に並んだ色んな花を見ていた'


# Work Around for mecab-python3 bug
# https://github.com/KosukeArima/next/issues/18
_MECAB_TOKENIZER = MeCab.Tagger("")


# https://shogo82148.github.io/blog/2015/12/20/mecab-in-python3-final/
_MECAB_TOKENIZER.parse('')


word_to_id = raw_data.get_word_to_id()
id_to_word = {v: k for k, v in word_to_id.items()}
eos_id = word_to_id['eos']

def w2id(word):
    if word not in word_to_id:
        word = 'unk'
    return word_to_id[word]


def create_model(mode_name, config, data, initializer):
    """Creates the tensorflow model"""
    with tf.name_scope(mode_name):
        reuse = None
        is_training = False

        input_ = Input(config=config, data=data, name=mode_name+'Input')

        with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
            m = Model(is_training=is_training, config=config, input_=input_)

        return m


def one_step(session, model, state, word_id):
    x = np.zeros((1, 1), dtype=np.int32)
    x[0, 0] = word_id
    feed_dict = {model.input.x: x, model.initial_state: state}
    state, probs = session.run([model.final_state, model.probs], feed_dict)
    probs = np.reshape(probs, (probs.shape[2]))
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return state, probs


def calc_loglik(session, model, words, state=None, eos=True):
    lk = 0.0
    state = session.run(model.initial_state)
    state, probs = one_step(session, model, state, eos_id)

    for word in words:
        word_id = word_to_id[word if word in word_to_id else 'unk']
        lk += math.log(probs[word_id] + 1e-10)
        state, probs = one_step(session, model, state, word_id)

    if eos:
        lk += math.log(probs[eos_id] + 1e-10)

    return lk


def count(x, y):
    return sum([1 for z in x if z == y])


def search_patterns(session, model, words):
    state = session.run(model.initial_state)
    state, probs = one_step(session, model, state, eos_id)

    cand = [words[0]]
    word_id = w2id(words[0])
    loglik = math.log(probs[word_id] + 1e-10)
    state, probs = one_step(session, model, state, word_id)

    cands = [(cand, loglik, state, probs)]
    for _ in range(len(words) - 1):
        new_cands = []
        for cand, loglik, state, probs in cands:
            for word in words:
                if count(cand, word) < count(words, word):
                    word_id = w2id(word)
                    new_cand = cand + [word]
                    new_loglik = loglik + math.log(probs[word_id] + 1e-10)
                    new_state, new_probs = one_step(session, model, state, word_id)
                    if len(new_cand) == len(words):
                        new_loglik = new_loglik + math.log(new_probs[eos_id] + 1e-10)
                        news_state, new_probs = one_step(session, model, new_state, eos_id)
                    new_cands.append((new_cand, new_loglik, new_state, new_probs))
        new_cands = sorted(new_cands, key=lambda x: -x[1])
        cands = new_cands[:10]

        print('============================================================\n', flush=True)
        for cand, loglik, _, _ in cands:
            print('LogLik: {:.3f}, {}'.format(loglik, ''.join(cand)))

    return [(''.join(cand), loglik) for cand, loglik, _, _ in cands]


def tokenize(s):
    words = []
    node = _MECAB_TOKENIZER.parseToNode(s)
    while node:
        if node.surface != '':
            words.append(node.surface)
        node = node.next
    return words


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

        # 尤度の高い単語の並び順を探索
        print('Searching likely patterns..', flush=True)
        words = tokenize(SEED_SENTENCE)
        patterns = search_patterns(session, model, words)
        if SEED_SENTENCE not in [x[0] for x in patterns]:
            patterns.append((SEED_SENTENCE, calc_loglik(session, model, words)))
            patterns = sorted(patterns, key=lambda x: -x[1])

        # ファイルへの出力
        with open(LOGDIR_PATH + 'likelihood.txt', 'w') as f:
            for pat, loglik in patterns:
                f.write('LogLik: {:.3f}, {}\n'.format(loglik, pat))


if __name__ == "__main__":
    tf.app.run()
