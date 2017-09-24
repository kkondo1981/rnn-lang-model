# What's this?
RNN言語モデルの精度検証

# PTB
PTB (Pen Tree Bank）データセットから英語文章生成を学習。

tensorflowのチュートリアル(https://www.tensorflow.org/tutorials/recurrent)
に以下の修正をして単純化したもの。
- 機能ごとにファイルを分けて若干整理＋コメント追加
- 複数GPU計算、ハイパーパラメータ選択（Large, Medium, Small）機能をdrop

ロジックおよびハイパーパラメータは下記論文のMediumモデルと同じであり、
PerplexityはTrain, Validともに80前後を達成可能。
- https://arxiv.org/abs/1409.2329

実行時間は、AWSのGPUインスタンス（p2.xlarge）で3時間程度(2017/9/23)。

## dataset
- Pen Tree Bank (PTB) dataaset
- Source: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

Test, Train, Valid用データのサイズはそれぞれ以下の通り。

|file name     |    行数  |    語数|
|ptb.test.txt  |   3,761  |  78,669|
|ptb.train.txt |  4,2068  | 887,521|
|ptb.valid.txt |   3,370  |  70,390|


## dependencies
以下のバージョンで検証済
- python: 3.6
- numpy: 1.13.1
- Tensorflow: tensorflow-gpu=1.2.1

## learning

```
# データセット作成
./dl-ptb.sh

# ディレクトリ作成（初回のみ）
mkdir ./log
mkdir ./log/ptb
mkdir ./model

# 学習スクリプト実行
python ./ptb/learn.py &

# 学習状況の可視化 (see http://localhost:6006/
tensorboard --logdir=./log/ptb &
```
