# -*- coding: utf-8 -*-
"""
入力テキストのraw data読み込み機能を実装。
- ソース: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
Test, Train, Valid用データのサイズはそれぞれ以下の通り。

|file name     |    行数  |    語数|
|ptb.test.txt  |   3,761  |  78,669|
|ptb.train.txt |  4,2068  | 887,521|
|ptb.valid.txt |   3,370  |  70,390|
"""

import collections

import tensorflow as tf


TRAIN_PATH = './data/ptb/simple-examples/data/ptb.train.txt'
VALID_PATH = './data/ptb/simple-examples/data/ptb.valid.txt'
TEST_PATH = './data/ptb/simple-examples/data/ptb.test.txt'


def _read_words(filename):
    """
    filenameの内容を空白で分割し、ワードリストを作成。改行は'<eos>'に変換される。
    このサンプルでは入力がテキストファイルのため、Python標準のIOでも全く問題ない。
    ただし、GFileはGoogle Could StrageやHDFS (Hadoop Distributed File System)に対応しており、
    入力がそちらに置いてある場合には便利。
    * https://stackoverflow.com/questions/42922948/why-use-tensorflow-gfile-for-file-i-o/
    なお、S3 Filesystemには今後対応予定？の模様。
    * https://github.com/tensorflow/tensorflow/pull/11089
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """
    filenameに含まれる全単語から、word => idのマップを作成。
    idは頻度が高い順に0から振られる。
    """
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    """
    filenameの内容をidのリストに変換。改行は'<eos>'に変換される。
    word_to_idに存在しない単語は無視される。
    """
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def get_raw_data():
    """
    ret: train_data, valid_data, test_data, vocabulary
    - train_data, valid_data, test_dataはテキスト中の単語を
      id化した整数のリスト
    - vocabularyは（Trainデータ中の）全語彙数
    """
    word_to_id = _build_vocab(TRAIN_PATH)
    train_data = _file_to_word_ids(TRAIN_PATH, word_to_id)
    valid_data = _file_to_word_ids(VALID_PATH, word_to_id)
    test_data = _file_to_word_ids(TEST_PATH, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def save_vocab(path):
    """
    辞書を1行1語形式で保存
    """
    word_to_id = _build_vocab(TRAIN_PATH)
    sorted_w2id = sorted(word_to_id.items(), key=lambda x:x[1])
    words = ['{}\n'.format(x[0]) for x in sorted_w2id]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(words)
