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

import re
import glob
import collections
from itertools import chain

import MeCab

import config as conf


# コーパスサイズを大きくし過ぎないように、一部を使用
SRC_DIRS = ['./data/livedoor/text/topic-news/',
            './data/livedoor/text/sports-watch/',
            './data/livedoor/text/smax/',
            './data/livedoor/text/peachy/']


# Work Around for mecab-python3 bug
# https://github.com/KosukeArima/next/issues/18
_MECAB_TOKENIZER = MeCab.Tagger("")

# https://shogo82148.github.io/blog/2015/12/20/mecab-in-python3-final/
_MECAB_TOKENIZER.parse('')


def _concat_files(dirnames):
    filenames = list(chain.from_iterable([sorted(glob.glob(dirname + '*.txt')) for dirname in dirnames]))
    filenames = [filename for filename in filenames if filename.find('LICENSE.txt') == -1]

    texts = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines() if line.strip() != '']
            if len(lines) > 3:
                texts.append(''.join(lines[3:]))

    text = '\n'.join(texts)
    text = text.replace('　', ' ')
    text = re.sub(r'([^"]|^)(https?|ftp)(://[\w:;/.?%#&=+-]+)', ' urlstr ', text)
    return text


def _split_by_sentence(text, seps='。．！!？?　\n'):
    sentences = []
    while len(text) > 0:
        n = len(text)
        depth = 0
        for i in range(n):
            c = text[i]
            if (c in seps and depth == 0) or i == n - 1:
                s = text[:i].strip()
                if s != '':
                    sentences.append(s + ' eos ')
                text = text[(i+1):]
                break
            elif c in '「『（(':
                depth += 1
            elif c in '」』）)' and depth > 0:
                depth -= 1
    return sentences


def _tokenize(sentences):
    words = []

    for sentence in sentences:
        node = _MECAB_TOKENIZER.parseToNode(sentence)
        while node:
            words.append(node.surface)
            node = node.next

    return words


def _get_sentences():
    sentences = _split_by_sentence(_concat_files(SRC_DIRS))
    unit = len(sentences) // 12

    cnt = 0

    train_sentences = sentences[cnt:(cnt + unit * 10)]
    train_words = _tokenize(train_sentences)
    cnt += unit * 10

    test_sentences = sentences[cnt:(cnt + unit)]
    test_words = _tokenize(test_sentences)
    cnt += unit

    valid_sentences = sentences[cnt:(cnt + unit)]
    valid_words = _tokenize(valid_sentences)

    print('train: {} lines, {} words'.format(len(train_sentences), len(train_words)))
    print('test: {} lines, {} words'.format(len(test_sentences), len(test_words)))
    print('valid: {} lines, {} words'.format(len(valid_sentences), len(valid_words)))

    return train_words, test_words, valid_words


train_words, test_words, valid_words = _get_sentences()


def _build_vocab(words):
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    config, _ = conf.get_config()
    count_pairs = count_pairs[:(config.vocab_size - 1)]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['unknown'] = len(words)

    return word_to_id


def _words_to_ids(words, word_to_id):
    unk_id = word_to_id['unknown']
    return [word_to_id[word] if word in word_to_id else unk_id for word in words]


def get_raw_data():
    """
    Returns:
    - train_data, valid_data, test_data: それぞれ単語IDのリスト
    - vocabulary: 全語彙数（語彙はTrainデータから作成）
    """
    word_to_id = _build_vocab(train_words)
    train_data = _words_to_ids(train_words, word_to_id)
    valid_data = _words_to_ids(test_words, word_to_id)
    test_data = _words_to_ids(valid_words, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def get_word_to_id():
    """
    Returns:
    - word_to_id: 単語⇒IDのリスト
    """
    word_to_id = _build_vocab(train_words)
    return word_to_id


def save_vocab(path):
    """
    辞書を1行1語形式でpathに保存
    """
    word_to_id = _build_vocab(train_words)
    sorted_w2id = sorted(word_to_id.items(), key=lambda x:x[1])
    words = ['{}\n'.format(x[0]) for x in sorted_w2id]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(words)
