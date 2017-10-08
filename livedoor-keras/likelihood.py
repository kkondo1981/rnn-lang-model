# -*- coding: utf-8 -*-
"""
RNN言語モデルによる尤度比較

[Usage] python ./livedoor-keras/likelihood.py
- 実行前にlearn.pyでモデルを学習・保存しておくこと。

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── livedoor-keras
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
LOGDIR_PATH = './log/livedoor-keras/'
MODEL_PATH = './model/livedoor-keras'


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


def calc_loglik(model, num_steps, words, state=None, eos=True):
    lk = 0.0
    x = [eos_id] * num_steps
    probs = model.calc_next_word_prob(x)

    for word in words:
        word_id = w2id(word)
        lk += math.log(probs[word_id] + 1e-10)
        x = x[1:] + [word_id]
        probs = model.calc_next_word_prob(x)

    if eos:
        lk += math.log(probs[eos_id] + 1e-10)

    return lk


def count(x, y):
    return sum([1 for z in x if z == y])


def search_patterns(model, num_steps, words):
    x = [eos_id] * num_steps
    probs = model.calc_next_word_prob(x)

    word_id = w2id(words[0])
    cand = [words[0]]
    cand_by_id = x + [word_id]
    loglik = math.log(probs[word_id] + 1e-10)
    probs = model.calc_next_word_prob(cand_by_id[-num_steps:])

    cands = [(cand, cand_by_id, loglik, probs)]
    for _ in range(len(words) - 1):
        new_cands = []
        for cand, cand_by_id, loglik, probs in cands:
            for word in words:
                if count(cand, word) < count(words, word):
                    word_id = w2id(word)
                    new_cand = cand + [word]
                    new_cand_by_id = cand_by_id + [word_id]
                    new_loglik = loglik + math.log(probs[word_id] + 1e-10)
                    new_probs = model.calc_next_word_prob(new_cand_by_id[-num_steps:])
                    if len(new_cand) == len(words):
                        new_loglik = new_loglik + math.log(new_probs[eos_id] + 1e-10)
                        new_probs = None
                    new_cands.append((new_cand, new_cand_by_id, new_loglik, new_probs))
        new_cands = sorted(new_cands, key=lambda x: -x[2])
        cands = new_cands[:20]

        print('============================================================\n', flush=True)
        for cand, _, loglik, _ in cands:
            print('LogLik: {:.3f}, {}'.format(loglik, ''.join(cand)))

    return [(''.join(cand), loglik) for cand, _, loglik, _ in cands]


def tokenize(s):
    words = []
    node = _MECAB_TOKENIZER.parseToNode(s)
    while node:
        if node.surface != '':
            words.append(node.surface)
        node = node.next
    return words


if __name__ == "__main__":
    # 各種設定
    config, _ = conf.get_config()

    # モデル構築
    m = Model(config, model_path_prefix=MODEL_PATH)
    m.model.summary()

    # 尤度の高い単語の並び順を探索
    print('Searching likely patterns..', flush=True)
    words = tokenize(SEED_SENTENCE)
    patterns = search_patterns(m, config.num_steps, words)
    if SEED_SENTENCE not in [x[0] for x in patterns]:
        patterns.append((SEED_SENTENCE, calc_loglik(m, config.num_steps, words)))
        patterns = sorted(patterns, key=lambda x: -x[1])

    # ファイルへの出力
    with open(LOGDIR_PATH + 'likelihood.txt', 'w') as f:
        for pat, loglik in patterns:
            f.write('LogLik: {:.3f}, {}\n'.format(loglik, pat))
