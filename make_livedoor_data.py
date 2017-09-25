# -*- coding: utf-8 -*-
"""
Livedoor News コーパスから言語モデル学習用のデータセットを作成。

Livedoor News コーパス
Source: https://www.rondhuit.com/download/ldcc-20140209.tar.gz

Test, Train, Valid用データのサイズはそれぞれ以下の通り。

|dataset     |    行数  |       語数 |
|Test        |    4,597 |     98,097 |
|Train       |   45,970 |  1,745,548 |
|Valid       |    4,597 |    101,714 |

なお、Mecabの設定次第でこの通りにならない場合あり。
当方環境では、mecab-ipadic-neologd使用。
mecab-ipadic-neologd: https://github.com/neologd/mecab-ipadic-neologd
"""

import re
import glob
import random
import collections
from itertools import chain

import MeCab


# コーパスサイズを大きくし過ぎると学習に支障をきたすので、一部を使用
SRC_DIRS = ['./data/livedoor/text/topic-news/',
            './data/livedoor/text/sports-watch/',
            './data/livedoor/text/smax/',
            './data/livedoor/text/peachy/']


# 作成したデータセット、辞書データを保存するディレクトリ
SAVE_DIR = './data/livedoor/'


# 作成する辞書サイズ
# 出現頻度が上位VOCAB_SIZE番目以下の単語は、全て'unk'に置換される。
VOCAB_SIZE = 10000


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
            if node.surface != '':
                words.append(node.surface)
            node = node.next

    return words


def _get_sentences():
    sentences = _split_by_sentence(_concat_files(SRC_DIRS))
    random.shuffle(sentences)

    cnt = 0
    unit = len(sentences) // 12

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


print('Concat and tokenize texts into (train, test, valid) words..', flush=True)
train_words, test_words, valid_words = _get_sentences()


def _build_freqs(words):
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return count_pairs


print('Build and save vocabularies ...', flush=True)
freqs = _build_freqs(train_words)
with open(SAVE_DIR + 'vocab_all.tsv', 'w', encoding='utf-8') as f:
    f.write('word\tfreq\n')
    f.write('\n'.join(['{}\t{}'.format(k, v) for k, v in freqs]))

freqs = freqs[:(VOCAB_SIZE - 1)]
freqs.append(('unk', 0))
with open(SAVE_DIR + 'vocab.tsv', 'w', encoding='utf-8') as f:
    f.write('word\tfreq\n')
    f.write('\n'.join(['{}\t{}'.format(k, v) for k, v in freqs]))


print('Save (train, test, valid) dataset into files', flush=True)
words, _ = list(zip(*freqs))
words_tuple = ([word if word in words else 'unk' for word in train_words],
               [word if word in words else 'unk' for word in test_words],
               [word if word in words else 'unk' for word in valid_words])
name_tuple = ('train', 'test', 'valid')
for words, name in zip(words_tuple, name_tuple):
    print('Save {} words into {}_words.txt ...'.format(name, name), flush=True)
    with open(SAVE_DIR + '{}_words.txt'.format(name), 'w', encoding='utf-8') as f:
        f.write(' '.join(words).replace('eos', 'eos\n'))
