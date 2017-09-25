# -*- coding: utf-8 -*-
"""
生入力データ

Livedoor News コーパス
Source: https://www.rondhuit.com/download/ldcc-20140209.tar.gz

Test, Train, Valid用データのサイズはそれぞれ以下の通り。
（Mecabでtokenizeしているため、設定次第でこの通りにならない場合あり）

|dataset     |    行数  |       語数 |
|Test        |    4,597 |     98,097 |
|Train       |   45,970 |  1,745,548 |
|Valid       |    4,597 |    101,714 |
"""


VOCAB_FILE_PATH = './data/livedoor/vocab.tsv'
TRAIN_WORDS_PATH = './data/livedoor/train_words.txt'
TEST_WORDS_PATH = './data/livedoor/test_words.txt'
VALID_WORDS_PATH = './data/livedoor/valid_words.txt'


def _buid_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = [line.strip().split('\t')[0] for line in f.readlines()]
    words = words[1:]
    return dict(zip(words, range(len(words))))


def _file_to_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split()


def _words_to_ids(words, word_to_id):
    unk_id = word_to_id['unk']
    return [word_to_id[word] if word in word_to_id else unk_id for word in words]


def _file_to_ids(path, word_to_id):
    return _words_to_ids(_file_to_words(path), word_to_id)


def get_raw_data():
    """
    Returns:
    - train_data, valid_data, test_data: それぞれ単語IDのリスト
    - vocabulary: 全語彙数（語彙はTrainデータから作成）
    """
    word_to_id = _build_vocab(VOCAB_FILE_PATH)
    train_data = _file_to_ids(TRAIN_WORDS_PATH, word_to_id)
    test_data = _file_to_ids(TEST_WORDS_PATH, word_to_id)
    valid_data = _file_to_ids(VALID_WORDS_PATH, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def get_word_to_id():
    """
    Returns:
    - word_to_id: 単語⇒IDのリスト
    """
    return = _build_vocab(VOCAB_FILE_PATH)
