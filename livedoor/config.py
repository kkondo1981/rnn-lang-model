# -*- coding: utf-8 -*-
"""
各種パスおよびRNNのハイパーパラメータを設定。

右記URLのMediumモデルに相当: https://arxiv.org/pdf/1409.2329.pdf

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
"""


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    decreasing_learning_rate_after = 6  # 6エポック後に学習レートを減衰開始
    max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_cell = 'GRU'
    # rnn_mode = "block"


def get_config():
    """Get model config."""
    config = MediumConfig()
    eval_config = MediumConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    return config, eval_config
