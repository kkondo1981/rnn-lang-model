# Livedoor
livedoorニュースコーパスから日本語言語モデルを学習。

言語モデル、学習ロジックはPTBと同じ。

## dataset
- Livedoor News コーパス: https://www.rondhuit.com/download.html
- Source: https://www.rondhuit.com/download/ldcc-20140209.tar.gz

## dependencies
以下のバージョンで検証済
- python: 3.6
- numpy: 1.13.1
- Tensorflow: tensorflow-gpu=1.2.1

## learning

```
# データセット作成
./dl-livedoor.sh

# ディレクトリ作成（初回のみ）
mkdir ./log
mkdir ./log/livedoor
# mkdir ./model  # make model dir if it does not exists

# 学習スクリプト実行
python ./livedoor/learn.py &

# 学習状況の可視化 (see http://localhost:6006/ )
tensorboard --logdir=./log/livedoor &
```

## text generation

```
python ./livedoor/gentext.py
cat ./log/livedoor/gentext.txt
```

## learning results

### 2017/9/24

#### 学習時間
EC2 p2.xlarge上で3h18m

#### hyper parameters
```
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
    # rnn_mode = "block"
```

#### perplexity
```
train: 45970 lines, 1654608 words
test: 4597 lines, 88903 words
valid: 4597 lines, 92520 words
Saving vocab file to ./log/livedoor/vocab.tsv.
Epoch: 1 Learning rate: 1.000
0.004 perplexity: 4201.983 speed: 3009 wps
0.104 perplexity: 512.686 speed: 5375 wps
0.204 perplexity: 297.511 speed: 5480 wps
0.304 perplexity: 220.549 speed: 5517 wps
0.404 perplexity: 182.368 speed: 5532 wps
0.504 perplexity: 159.010 speed: 5544 wps
0.603 perplexity: 143.087 speed: 5552 wps
0.703 perplexity: 129.238 speed: 5558 wps
0.803 perplexity: 118.941 speed: 5562 wps
0.903 perplexity: 110.913 speed: 5564 wps
Epoch: 1 Train Perplexity: 104.332
Epoch: 1 Valid Perplexity: 71.344
Epoch: 2 Learning rate: 1.000
0.004 perplexity: 66.972 speed: 5583 wps
0.104 perplexity: 59.404 speed: 5591 wps
0.204 perplexity: 55.724 speed: 5583 wps
0.304 perplexity: 53.972 speed: 5586 wps
0.404 perplexity: 53.520 speed: 5587 wps
0.504 perplexity: 53.175 speed: 5588 wps
0.603 perplexity: 52.838 speed: 5586 wps
0.703 perplexity: 51.569 speed: 5587 wps
0.803 perplexity: 50.665 speed: 5587 wps
0.903 perplexity: 49.920 speed: 5588 wps
Epoch: 2 Train Perplexity: 49.116
Epoch: 2 Valid Perplexity: 59.535
Epoch: 3 Learning rate: 1.000
0.004 perplexity: 47.848 speed: 5505 wps
0.104 perplexity: 44.757 speed: 5588 wps
0.204 perplexity: 42.255 speed: 5589 wps
0.304 perplexity: 41.264 speed: 5590 wps
0.404 perplexity: 41.271 speed: 5587 wps
0.504 perplexity: 41.443 speed: 5588 wps
0.603 perplexity: 41.533 speed: 5588 wps
0.703 perplexity: 40.830 speed: 5588 wps
0.803 perplexity: 40.412 speed: 5587 wps
0.903 perplexity: 40.019 speed: 5588 wps
Epoch: 3 Train Perplexity: 39.603
Epoch: 3 Valid Perplexity: 52.495
Epoch: 4 Learning rate: 1.000
0.004 perplexity: 41.883 speed: 5590 wps
0.104 perplexity: 38.458 speed: 5592 wps
0.204 perplexity: 36.340 speed: 5583 wps
0.304 perplexity: 35.599 speed: 5586 wps
0.404 perplexity: 35.604 speed: 5587 wps
0.504 perplexity: 35.884 speed: 5588 wps
0.603 perplexity: 36.081 speed: 5586 wps
0.703 perplexity: 35.563 speed: 5587 wps
0.803 perplexity: 35.317 speed: 5587 wps
0.903 perplexity: 35.058 speed: 5588 wps
Epoch: 4 Train Perplexity: 34.795
Epoch: 4 Valid Perplexity: 50.177
Epoch: 5 Learning rate: 1.000
0.004 perplexity: 37.933 speed: 5585 wps
0.104 perplexity: 34.704 speed: 5593 wps
0.204 perplexity: 32.952 speed: 5591 wps
0.304 perplexity: 32.311 speed: 5591 wps
0.404 perplexity: 32.372 speed: 5587 wps
0.504 perplexity: 32.632 speed: 5588 wps
0.603 perplexity: 32.863 speed: 5589 wps
0.703 perplexity: 32.458 speed: 5589 wps
0.803 perplexity: 32.288 speed: 5587 wps
0.903 perplexity: 32.081 speed: 5588 wps
Epoch: 5 Train Perplexity: 31.899
Epoch: 5 Valid Perplexity: 48.307
Epoch: 6 Learning rate: 1.000
0.004 perplexity: 35.221 speed: 5595 wps
0.104 perplexity: 32.291 speed: 5592 wps
0.204 perplexity: 30.631 speed: 5584 wps
0.304 perplexity: 30.065 speed: 5586 wps
0.404 perplexity: 30.140 speed: 5587 wps
0.504 perplexity: 30.392 speed: 5588 wps
0.603 perplexity: 30.607 speed: 5586 wps
0.703 perplexity: 30.274 speed: 5587 wps
0.803 perplexity: 30.141 speed: 5587 wps
0.903 perplexity: 29.960 speed: 5588 wps
Epoch: 6 Train Perplexity: 29.814
Epoch: 6 Valid Perplexity: 46.838
Epoch: 7 Learning rate: 0.800
0.004 perplexity: 34.510 speed: 5593 wps
0.104 perplexity: 30.123 speed: 5591 wps
0.204 perplexity: 28.444 speed: 5591 wps
0.304 perplexity: 27.828 speed: 5591 wps
0.404 perplexity: 27.788 speed: 5586 wps
0.504 perplexity: 27.905 speed: 5587 wps
0.603 perplexity: 28.068 speed: 5588 wps
0.703 perplexity: 27.723 speed: 5588 wps
0.803 perplexity: 27.599 speed: 5586 wps
0.903 perplexity: 27.402 speed: 5587 wps
Epoch: 7 Train Perplexity: 27.246
Epoch: 7 Valid Perplexity: 44.553
Epoch: 8 Learning rate: 0.640
0.004 perplexity: 33.705 speed: 5577 wps
0.104 perplexity: 27.858 speed: 5590 wps
0.204 perplexity: 26.393 speed: 5582 wps
0.304 perplexity: 25.760 speed: 5585 wps
0.404 perplexity: 25.759 speed: 5586 wps
0.504 perplexity: 25.842 speed: 5587 wps
0.603 perplexity: 25.972 speed: 5585 wps
0.703 perplexity: 25.659 speed: 5586 wps
0.803 perplexity: 25.521 speed: 5586 wps
0.903 perplexity: 25.346 speed: 5587 wps
Epoch: 8 Train Perplexity: 25.210
Epoch: 8 Valid Perplexity: 43.332
Epoch: 9 Learning rate: 0.512
0.004 perplexity: 31.164 speed: 5592 wps
0.104 perplexity: 26.131 speed: 5590 wps
0.204 perplexity: 24.675 speed: 5589 wps
0.304 perplexity: 24.100 speed: 5589 wps
0.404 perplexity: 24.148 speed: 5586 wps
0.504 perplexity: 24.194 speed: 5587 wps
0.603 perplexity: 24.329 speed: 5588 wps
0.703 perplexity: 24.026 speed: 5588 wps
0.803 perplexity: 23.907 speed: 5586 wps
0.903 perplexity: 23.717 speed: 5587 wps
Epoch: 9 Train Perplexity: 23.591
Epoch: 9 Valid Perplexity: 42.768
Epoch: 10 Learning rate: 0.410
0.004 perplexity: 28.769 speed: 5590 wps
0.104 perplexity: 24.710 speed: 5592 wps
0.204 perplexity: 23.382 speed: 5583 wps
0.304 perplexity: 22.794 speed: 5585 wps
0.404 perplexity: 22.835 speed: 5587 wps
0.504 perplexity: 22.876 speed: 5588 wps
0.603 perplexity: 23.015 speed: 5585 wps
0.703 perplexity: 22.727 speed: 5586 wps
0.803 perplexity: 22.625 speed: 5586 wps
0.903 perplexity: 22.449 speed: 5587 wps
Epoch: 10 Train Perplexity: 22.346
Epoch: 10 Valid Perplexity: 41.862
Epoch: 11 Learning rate: 0.328
0.004 perplexity: 28.539 speed: 5591 wps
0.104 perplexity: 23.586 speed: 5592 wps
0.204 perplexity: 22.353 speed: 5592 wps
0.304 perplexity: 21.793 speed: 5592 wps
0.404 perplexity: 21.842 speed: 5587 wps
0.504 perplexity: 21.889 speed: 5587 wps
0.603 perplexity: 22.024 speed: 5587 wps
0.703 perplexity: 21.737 speed: 5588 wps
0.803 perplexity: 21.646 speed: 5585 wps
0.903 perplexity: 21.467 speed: 5586 wps
Epoch: 11 Train Perplexity: 21.375
Epoch: 11 Valid Perplexity: 41.602
Epoch: 12 Learning rate: 0.262
0.004 perplexity: 24.817 speed: 5587 wps
0.104 perplexity: 22.551 speed: 5592 wps
0.204 perplexity: 21.433 speed: 5583 wps
0.304 perplexity: 20.905 speed: 5586 wps
0.404 perplexity: 20.998 speed: 5587 wps
0.504 perplexity: 21.031 speed: 5587 wps
0.603 perplexity: 21.167 speed: 5585 wps
0.703 perplexity: 20.881 speed: 5586 wps
0.803 perplexity: 20.796 speed: 5586 wps
0.903 perplexity: 20.622 speed: 5586 wps
Epoch: 12 Train Perplexity: 20.547
Epoch: 12 Valid Perplexity: 41.290
Epoch: 13 Learning rate: 0.210
0.004 perplexity: 23.947 speed: 5586 wps
0.104 perplexity: 22.017 speed: 5589 wps
0.204 perplexity: 20.885 speed: 5588 wps
0.304 perplexity: 20.319 speed: 5589 wps
0.404 perplexity: 20.396 speed: 5585 wps
0.504 perplexity: 20.420 speed: 5586 wps
0.603 perplexity: 20.546 speed: 5587 wps
0.703 perplexity: 20.276 speed: 5587 wps
0.803 perplexity: 20.204 speed: 5585 wps
0.903 perplexity: 20.023 speed: 5585 wps
Epoch: 13 Train Perplexity: 19.949
Epoch: 13 Valid Perplexity: 41.102
Epoch: 14 Learning rate: 0.168
0.004 perplexity: 20.299 speed: 5584 wps
0.104 perplexity: 21.343 speed: 5592 wps
0.204 perplexity: 20.336 speed: 5583 wps
0.304 perplexity: 19.805 speed: 5585 wps
0.404 perplexity: 19.908 speed: 5586 wps
0.504 perplexity: 19.920 speed: 5587 wps
0.603 perplexity: 20.034 speed: 5584 wps
0.703 perplexity: 19.780 speed: 5585 wps
0.803 perplexity: 19.700 speed: 5585 wps
0.903 perplexity: 19.520 speed: 5585 wps
Epoch: 14 Train Perplexity: 19.452
Epoch: 14 Valid Perplexity: 40.620
Epoch: 15 Learning rate: 0.134
0.004 perplexity: 18.720 speed: 5586 wps
0.104 perplexity: 20.938 speed: 5590 wps
0.204 perplexity: 19.849 speed: 5589 wps
0.304 perplexity: 19.357 speed: 5589 wps
0.404 perplexity: 19.478 speed: 5585 wps
0.504 perplexity: 19.498 speed: 5586 wps
0.603 perplexity: 19.591 speed: 5587 wps
0.703 perplexity: 19.343 speed: 5587 wps
0.803 perplexity: 19.283 speed: 5585 wps
0.903 perplexity: 19.108 speed: 5586 wps
Epoch: 15 Train Perplexity: 19.039
Epoch: 15 Valid Perplexity: 40.483
Epoch: 16 Learning rate: 0.107
0.004 perplexity: 20.156 speed: 5593 wps
0.104 perplexity: 20.596 speed: 5589 wps
0.204 perplexity: 19.581 speed: 5583 wps
0.304 perplexity: 19.072 speed: 5585 wps
0.404 perplexity: 19.134 speed: 5586 wps
0.504 perplexity: 19.167 speed: 5587 wps
0.603 perplexity: 19.259 speed: 5585 wps
0.703 perplexity: 19.010 speed: 5585 wps
0.803 perplexity: 18.939 speed: 5586 wps
0.903 perplexity: 18.770 speed: 5586 wps
Epoch: 16 Train Perplexity: 18.699
Epoch: 16 Valid Perplexity: 40.389
Epoch: 17 Learning rate: 0.086
0.004 perplexity: 21.172 speed: 5581 wps
0.104 perplexity: 20.329 speed: 5590 wps
0.204 perplexity: 19.289 speed: 5590 wps
0.304 perplexity: 18.809 speed: 5590 wps
0.404 perplexity: 18.878 speed: 5586 wps
0.504 perplexity: 18.920 speed: 5586 wps
0.603 perplexity: 18.986 speed: 5586 wps
0.703 perplexity: 18.763 speed: 5587 wps
0.803 perplexity: 18.682 speed: 5585 wps
0.903 perplexity: 18.521 speed: 5586 wps
Epoch: 17 Train Perplexity: 18.450
Epoch: 17 Valid Perplexity: 40.523
Epoch: 18 Learning rate: 0.069
0.004 perplexity: 22.943 speed: 5591 wps
0.104 perplexity: 20.130 speed: 5591 wps
0.204 perplexity: 19.132 speed: 5582 wps
0.304 perplexity: 18.619 speed: 5585 wps
0.404 perplexity: 18.685 speed: 5587 wps
0.504 perplexity: 18.739 speed: 5587 wps
0.603 perplexity: 18.785 speed: 5585 wps
0.703 perplexity: 18.575 speed: 5586 wps
0.803 perplexity: 18.505 speed: 5587 wps
0.903 perplexity: 18.326 speed: 5587 wps
Epoch: 18 Train Perplexity: 18.255
Epoch: 18 Valid Perplexity: 40.518
Epoch: 19 Learning rate: 0.055
0.004 perplexity: 23.715 speed: 5594 wps
0.104 perplexity: 19.966 speed: 5590 wps
0.204 perplexity: 18.965 speed: 5590 wps
0.304 perplexity: 18.419 speed: 5590 wps
0.404 perplexity: 18.509 speed: 5586 wps
0.504 perplexity: 18.579 speed: 5587 wps
0.603 perplexity: 18.602 speed: 5587 wps
0.703 perplexity: 18.404 speed: 5588 wps
0.803 perplexity: 18.332 speed: 5586 wps
0.903 perplexity: 18.150 speed: 5586 wps
Epoch: 19 Train Perplexity: 18.091
Epoch: 19 Valid Perplexity: 40.603
Epoch: 20 Learning rate: 0.044
0.004 perplexity: 23.551 speed: 5591 wps
0.104 perplexity: 19.818 speed: 5573 wps
0.204 perplexity: 18.819 speed: 5582 wps
0.304 perplexity: 18.267 speed: 5583 wps
0.404 perplexity: 18.358 speed: 5585 wps
0.504 perplexity: 18.441 speed: 5586 wps
0.603 perplexity: 18.453 speed: 5584 wps
0.703 perplexity: 18.265 speed: 5584 wps
0.803 perplexity: 18.198 speed: 5585 wps
0.903 perplexity: 18.004 speed: 5585 wps
Epoch: 20 Train Perplexity: 17.964
Epoch: 20 Valid Perplexity: 40.407
Epoch: 21 Learning rate: 0.035
0.004 perplexity: 23.537 speed: 5586 wps
0.104 perplexity: 19.661 speed: 5588 wps
0.204 perplexity: 18.692 speed: 5589 wps
0.304 perplexity: 18.145 speed: 5582 wps
0.404 perplexity: 18.244 speed: 5584 wps
0.504 perplexity: 18.346 speed: 5585 wps
0.603 perplexity: 18.351 speed: 5586 wps
0.703 perplexity: 18.170 speed: 5586 wps
0.803 perplexity: 18.105 speed: 5585 wps
0.903 perplexity: 17.906 speed: 5585 wps
Epoch: 21 Train Perplexity: 17.864
Epoch: 21 Valid Perplexity: 40.340
Epoch: 22 Learning rate: 0.028
0.004 perplexity: 24.502 speed: 5584 wps
0.104 perplexity: 19.454 speed: 5573 wps
0.204 perplexity: 18.551 speed: 5581 wps
0.304 perplexity: 18.024 speed: 5583 wps
0.404 perplexity: 18.114 speed: 5585 wps
0.504 perplexity: 18.232 speed: 5582 wps
0.603 perplexity: 18.226 speed: 5583 wps
0.703 perplexity: 18.059 speed: 5583 wps
0.803 perplexity: 17.982 speed: 5584 wps
0.903 perplexity: 17.790 speed: 5582 wps
Epoch: 22 Train Perplexity: 17.759
Epoch: 22 Valid Perplexity: 40.303
Epoch: 23 Learning rate: 0.023
0.004 perplexity: 23.370 speed: 5581 wps
0.104 perplexity: 19.402 speed: 5585 wps
0.204 perplexity: 18.476 speed: 5587 wps
0.304 perplexity: 17.948 speed: 5583 wps
0.404 perplexity: 18.043 speed: 5585 wps
0.504 perplexity: 18.171 speed: 5585 wps
0.603 perplexity: 18.167 speed: 5586 wps
0.703 perplexity: 18.008 speed: 5585 wps
0.803 perplexity: 17.926 speed: 5585 wps
0.903 perplexity: 17.736 speed: 5586 wps
Epoch: 23 Train Perplexity: 17.697
Epoch: 23 Valid Perplexity: 40.328
Epoch: 24 Learning rate: 0.018
0.004 perplexity: 24.089 speed: 5581 wps
0.104 perplexity: 19.333 speed: 5576 wps
0.204 perplexity: 18.429 speed: 5588 wps
0.304 perplexity: 17.868 speed: 5592 wps
0.404 perplexity: 17.965 speed: 5595 wps
0.504 perplexity: 18.119 speed: 5590 wps
0.603 perplexity: 18.099 speed: 5590 wps
0.703 perplexity: 17.946 speed: 5590 wps
0.803 perplexity: 17.867 speed: 5590 wps
0.903 perplexity: 17.689 speed: 5573 wps
Epoch: 24 Train Perplexity: 17.655
Epoch: 24 Valid Perplexity: 40.198
Epoch: 25 Learning rate: 0.014
0.004 perplexity: 22.338 speed: 5585 wps
0.104 perplexity: 19.206 speed: 5590 wps
0.204 perplexity: 18.349 speed: 5591 wps
0.304 perplexity: 17.816 speed: 5586 wps
0.404 perplexity: 17.914 speed: 5586 wps
0.504 perplexity: 18.062 speed: 5587 wps
0.603 perplexity: 18.048 speed: 5588 wps
0.703 perplexity: 17.896 speed: 5585 wps
0.803 perplexity: 17.811 speed: 5586 wps
0.903 perplexity: 17.627 speed: 5586 wps
Epoch: 25 Train Perplexity: 17.596
Epoch: 25 Valid Perplexity: 40.175
Epoch: 26 Learning rate: 0.012
0.004 perplexity: 20.541 speed: 5582 wps
0.104 perplexity: 19.163 speed: 5575 wps
0.204 perplexity: 18.303 speed: 5582 wps
0.304 perplexity: 17.769 speed: 5584 wps
0.404 perplexity: 17.872 speed: 5586 wps
0.504 perplexity: 18.034 speed: 5583 wps
0.603 perplexity: 18.025 speed: 5585 wps
0.703 perplexity: 17.851 speed: 5585 wps
0.803 perplexity: 17.780 speed: 5586 wps
0.903 perplexity: 17.602 speed: 5571 wps
Epoch: 26 Train Perplexity: 17.584
Epoch: 26 Valid Perplexity: 40.174
Epoch: 27 Learning rate: 0.009
0.004 perplexity: 19.105 speed: 5576 wps
0.104 perplexity: 19.040 speed: 5588 wps
0.204 perplexity: 18.219 speed: 5589 wps
0.304 perplexity: 17.689 speed: 5583 wps
0.404 perplexity: 17.817 speed: 5583 wps
0.504 perplexity: 17.973 speed: 5585 wps
0.603 perplexity: 17.965 speed: 5586 wps
0.703 perplexity: 17.806 speed: 5584 wps
0.803 perplexity: 17.733 speed: 5585 wps
0.903 perplexity: 17.556 speed: 5585 wps
Epoch: 27 Train Perplexity: 17.534
Epoch: 27 Valid Perplexity: 40.132
Epoch: 28 Learning rate: 0.007
0.004 perplexity: 18.597 speed: 5580 wps
0.104 perplexity: 19.070 speed: 5572 wps
0.204 perplexity: 18.225 speed: 5580 wps
0.304 perplexity: 17.659 speed: 5583 wps
0.404 perplexity: 17.812 speed: 5585 wps
0.504 perplexity: 17.971 speed: 5583 wps
0.603 perplexity: 17.947 speed: 5584 wps
0.703 perplexity: 17.794 speed: 5585 wps
0.803 perplexity: 17.722 speed: 5585 wps
0.903 perplexity: 17.536 speed: 5571 wps
Epoch: 28 Train Perplexity: 17.532
Epoch: 28 Valid Perplexity: 40.181
Epoch: 29 Learning rate: 0.006
0.004 perplexity: 16.618 speed: 5584 wps
0.104 perplexity: 19.005 speed: 5591 wps
0.204 perplexity: 18.163 speed: 5592 wps
0.304 perplexity: 17.641 speed: 5586 wps
0.404 perplexity: 17.792 speed: 5587 wps
0.504 perplexity: 17.958 speed: 5587 wps
0.603 perplexity: 17.924 speed: 5588 wps
0.703 perplexity: 17.782 speed: 5586 wps
0.803 perplexity: 17.708 speed: 5586 wps
0.903 perplexity: 17.515 speed: 5586 wps
Epoch: 29 Train Perplexity: 17.516
Epoch: 29 Valid Perplexity: 40.176
Epoch: 30 Learning rate: 0.005
0.004 perplexity: 16.494 speed: 5588 wps
0.104 perplexity: 19.081 speed: 5571 wps
0.204 perplexity: 18.244 speed: 5581 wps
0.304 perplexity: 17.669 speed: 5584 wps
0.404 perplexity: 17.782 speed: 5586 wps
0.504 perplexity: 17.963 speed: 5583 wps
0.603 perplexity: 17.903 speed: 5585 wps
0.703 perplexity: 17.764 speed: 5585 wps
0.803 perplexity: 17.697 speed: 5586 wps
0.903 perplexity: 17.497 speed: 5577 wps
Epoch: 30 Train Perplexity: 17.499
Epoch: 30 Valid Perplexity: 40.126
Epoch: 31 Learning rate: 0.004
0.004 perplexity: 15.853 speed: 5586 wps
0.104 perplexity: 19.100 speed: 5587 wps
0.204 perplexity: 18.220 speed: 5588 wps
0.304 perplexity: 17.630 speed: 5584 wps
0.404 perplexity: 17.761 speed: 5585 wps
0.504 perplexity: 17.945 speed: 5585 wps
0.603 perplexity: 17.876 speed: 5586 wps
0.703 perplexity: 17.739 speed: 5584 wps
0.803 perplexity: 17.681 speed: 5585 wps
0.903 perplexity: 17.474 speed: 5585 wps
Epoch: 31 Train Perplexity: 17.489
Epoch: 31 Valid Perplexity: 40.162
Epoch: 32 Learning rate: 0.003
0.004 perplexity: 15.622 speed: 5586 wps
0.104 perplexity: 19.058 speed: 5571 wps
0.204 perplexity: 18.142 speed: 5581 wps
0.304 perplexity: 17.563 speed: 5584 wps
0.404 perplexity: 17.706 speed: 5585 wps
0.504 perplexity: 17.923 speed: 5582 wps
0.603 perplexity: 17.825 speed: 5584 wps
0.703 perplexity: 17.694 speed: 5585 wps
0.803 perplexity: 17.642 speed: 5585 wps
0.903 perplexity: 17.433 speed: 5567 wps
Epoch: 32 Train Perplexity: 17.451
Epoch: 32 Valid Perplexity: 40.160
Epoch: 33 Learning rate: 0.002
0.004 perplexity: 15.957 speed: 5579 wps
0.104 perplexity: 19.124 speed: 5588 wps
0.204 perplexity: 18.239 speed: 5590 wps
0.304 perplexity: 17.610 speed: 5584 wps
0.404 perplexity: 17.751 speed: 5585 wps
0.504 perplexity: 17.963 speed: 5586 wps
0.603 perplexity: 17.832 speed: 5586 wps
0.703 perplexity: 17.706 speed: 5584 wps
0.803 perplexity: 17.664 speed: 5585 wps
0.903 perplexity: 17.435 speed: 5585 wps
Epoch: 33 Train Perplexity: 17.450
Epoch: 33 Valid Perplexity: 40.190
Epoch: 34 Learning rate: 0.002
0.004 perplexity: 16.227 speed: 5589 wps
0.104 perplexity: 19.077 speed: 5575 wps
0.204 perplexity: 18.177 speed: 5581 wps
0.304 perplexity: 17.578 speed: 5583 wps
0.404 perplexity: 17.678 speed: 5585 wps
0.504 perplexity: 17.926 speed: 5582 wps
0.603 perplexity: 17.792 speed: 5583 wps
0.703 perplexity: 17.667 speed: 5584 wps
0.803 perplexity: 17.626 speed: 5584 wps
0.903 perplexity: 17.405 speed: 5570 wps
Epoch: 34 Train Perplexity: 17.430
Epoch: 34 Valid Perplexity: 40.136
Epoch: 35 Learning rate: 0.002
0.004 perplexity: 17.400 speed: 5574 wps
0.104 perplexity: 19.043 speed: 5587 wps
0.204 perplexity: 18.148 speed: 5588 wps
0.304 perplexity: 17.599 speed: 5582 wps
0.404 perplexity: 17.709 speed: 5584 wps
0.504 perplexity: 17.936 speed: 5585 wps
0.603 perplexity: 17.792 speed: 5585 wps
0.703 perplexity: 17.666 speed: 5583 wps
0.803 perplexity: 17.631 speed: 5584 wps
0.903 perplexity: 17.410 speed: 5585 wps
Epoch: 35 Train Perplexity: 17.424
Epoch: 35 Valid Perplexity: 40.081
Epoch: 36 Learning rate: 0.001
0.004 perplexity: 18.071 speed: 5568 wps
0.104 perplexity: 19.128 speed: 5570 wps
0.204 perplexity: 18.175 speed: 5579 wps
0.304 perplexity: 17.641 speed: 5582 wps
0.404 perplexity: 17.743 speed: 5585 wps
0.504 perplexity: 17.976 speed: 5582 wps
0.603 perplexity: 17.821 speed: 5583 wps
0.703 perplexity: 17.695 speed: 5584 wps
0.803 perplexity: 17.655 speed: 5584 wps
0.903 perplexity: 17.433 speed: 5571 wps
Epoch: 36 Train Perplexity: 17.434
Epoch: 36 Valid Perplexity: 40.165
Epoch: 37 Learning rate: 0.001
0.004 perplexity: 18.554 speed: 5585 wps
0.104 perplexity: 19.163 speed: 5588 wps
0.204 perplexity: 18.156 speed: 5588 wps
0.304 perplexity: 17.654 speed: 5583 wps
0.404 perplexity: 17.751 speed: 5585 wps
0.504 perplexity: 17.972 speed: 5585 wps
0.603 perplexity: 17.805 speed: 5586 wps
0.703 perplexity: 17.686 speed: 5584 wps
0.803 perplexity: 17.631 speed: 5584 wps
0.903 perplexity: 17.412 speed: 5585 wps
Epoch: 37 Train Perplexity: 17.420
Epoch: 37 Valid Perplexity: 40.097
Epoch: 38 Learning rate: 0.001
0.004 perplexity: 18.451 speed: 5583 wps
0.104 perplexity: 19.045 speed: 5573 wps
0.204 perplexity: 18.108 speed: 5580 wps
0.304 perplexity: 17.645 speed: 5583 wps
0.404 perplexity: 17.740 speed: 5585 wps
0.504 perplexity: 17.965 speed: 5582 wps
0.603 perplexity: 17.776 speed: 5583 wps
0.703 perplexity: 17.663 speed: 5583 wps
0.803 perplexity: 17.614 speed: 5584 wps
0.903 perplexity: 17.408 speed: 5570 wps
Epoch: 38 Train Perplexity: 17.424
Epoch: 38 Valid Perplexity: 40.126
Epoch: 39 Learning rate: 0.001
0.004 perplexity: 18.541 speed: 5592 wps
0.104 perplexity: 18.924 speed: 5588 wps
0.204 perplexity: 18.023 speed: 5587 wps
0.304 perplexity: 17.669 speed: 5582 wps
0.404 perplexity: 17.754 speed: 5583 wps
0.504 perplexity: 17.992 speed: 5584 wps
0.603 perplexity: 17.788 speed: 5584 wps
0.703 perplexity: 17.685 speed: 5583 wps
0.803 perplexity: 17.646 speed: 5583 wps
0.903 perplexity: 17.411 speed: 5584 wps
Epoch: 39 Train Perplexity: 17.424
Epoch: 39 Valid Perplexity: 40.072
Test Perplexity: 46.183
Saving model to ./model/.

```
