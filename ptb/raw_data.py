# -*- coding: utf-8 -*-
"""
生入力データ

Pen Tree Bank (PTB) dataaset
Source: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

Test, Train, Valid用データのサイズはそれぞれ以下の通り。

|file name     |    行数  |    語数|
|ptb.test.txt  |   3,761  |  78,669|
|ptb.train.txt |  4,2068  | 887,521|
|ptb.valid.txt |   3,370  |  70,390|
"""

import collections

import tensorflow as tf


TEST_PATH = './data/ptb/simple-examples/data/ptb.test.txt'
TRAIN_PATH = './data/ptb/simple-examples/data/ptb.train.txt'
VALID_PATH = './data/ptb/simple-examples/data/ptb.valid.txt'


def _read_words(filename):
    """
    テキストファイルからワードリストを作成（空白文字で分割）。改行は'<eos>'に変換。

    Python標準のFile IOではなくGFileを使用しているため、Google Could Strage、HDFS
    (Hadoop Distributed File System)からも入力可能（な筈）。
    - https://stackoverflow.com/questions/42922948/why-use-tensorflow-gfile-for-file-i-o/

    なお、GFileは現時点ではS3 Filesystemには対応していない模様（今後対応予定）。
    - https://github.com/tensorflow/tensorflow/pull/11089
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """
    単語（文字列） => 単語IDのマップを作成。
    IDは出現頻度が高い順に0から振られる。
    """
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    """
    テキストファイルから単語IDのリストを作成。改行は'<eos>'に変換される。
    単語IDはword_to_idに従う。なお、word_to_idに存在しない単語は無視される。
    """
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def get_raw_data():
    """
    Returns:
    - train_data, valid_data, test_data: それぞれ単語IDのリスト
    - vocabulary: 全語彙数（語彙はTrainデータから作成）
    """
    word_to_id = _build_vocab(TRAIN_PATH)
    train_data = _file_to_word_ids(TRAIN_PATH, word_to_id)
    valid_data = _file_to_word_ids(VALID_PATH, word_to_id)
    test_data = _file_to_word_ids(TEST_PATH, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def get_word_to_id():
    """
    Returns:
    - word_to_id: 単語⇒IDのリスト
    """
    word_to_id = _builc_vocab(TRAIN_PATH)
    return word_to_id


def save_vocab(path):
    """
    辞書を1行1語形式でpathに保存
    """
    word_to_id = _build_vocab(TRAIN_PATH)
    sorted_w2id = sorted(word_to_id.items(), key=lambda x:x[1])
    words = ['{}\n'.format(x[0]) for x in sorted_w2id]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(words)
