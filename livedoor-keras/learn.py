# -*- coding: utf-8 -*-
"""
RNN言語モデルの学習処理

[Usage] python ./livedoor-keras/learn.py

右記URLのMediumモデルに相当: https://arxiv.org/pdf/1409.2329.pdf

実行時のディレクトリ構成は、下記想定:
```
 current dir(*)
｜
├── livedoor-keras
｜   └── learn.py      : this script
｜
├── data
｜   └── livedoor      : data dir
｜
｜
└── model              : model dir
```

上記以外の構成で実行する場合には、適宜SAVE_PATHの値を修正して実行すること。

"""

import numpy as np

import config as conf
import raw_data
from rnn_language_model import RNNLanguageModel as Model
from rnn_language_model_input import RNNLanguageModelInput as Input


# PATHs
SAVE_PATH = './model/'


def calc_perplexity(model, input_, batch_size):
    loss = model.evaluate(input_.x, input_.y, batch_size=batch_size)
    return np.exp(loss)


if __name__ == "__main__":
    # 各種設定
    config, eval_config = conf.get_config()

    # モデル構築
    m = Model(config)
    m.model.summary()

    # インプット作成
    train_data, valid_data, test_data, _ = raw_data.get_raw_data()
    train_input = Input(config, train_data)
    valid_input = Input(eval_config, valid_data)
    test_input = Input(config, test_data)

    # 学習実行
    for i in range(config.max_epoch):
        lr_decay = config.lr_decay \
                   ** max(i + 1 - config.decreasing_learning_rate_after, 0.0)
        lr = config.learning_rate * lr_decay
        print('Epoch: {} Learning rate: {:.3f}'.format(i + 1, lr))

        m.model.fit(x=train_input.x, y=train_input.y, batch_size=config.batch_size, epochs=1)

        perp = calc_perplexity(m.model, train_input, config.batch_size)
        print('Epoch: {} Train Perplexity: {:.3f}'.format(i + 1, perp))

        perp = calc_perplexity(m.model, valid_input, eval_config.batch_size)
        print('Epoch: {} Valid Perplexity: {:.3f}'.format(i + 1, perp))

    perp = calc_perplexity(m.model, test_input, config.batch_size)
    print('Test Perplexity: {:.3f}'.format(perp))

    print('Saving model to {}.'.format(SAVE_PATH))
    m.save(SAVE_PATH + 'livedoor-keras')
