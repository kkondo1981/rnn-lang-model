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
- Pen Tree Bank (PTB) dataset
- Source: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

Test, Train, Valid用データのサイズはそれぞれ以下の通り。

|データ        |    行数  |    語数|
|------------- |:--------:| ------ |
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

# 学習状況の可視化 (see http://localhost:6006/ )
tensorboard --logdir=./log/ptb &
```

## text generation

```
python ./ptb/gentext.py
cat ./log/ptb/gentext.txt
```

### generated text at 2017/9/24

最初の１文はシード文

```
============================================================
** generated with diversity 1.00 **
consumers may want to move their telephones a little closer to the tv set
he says the assault for the <unk> hollywood site is unfair
for the period of japan nasa reported a N N revenue increase
consumer cars rose N N while the u.s. had sent a library as the <unk> earthquake natural gas for the summer
cuba also signaled pleasure to <unk> <unk> that export 's in N of N the workers said
in addition further u.s. exports also <unk> in line since the seventh san francisco numbers were entered last week
our u.s. operations did n't obviously feel up how a bank proposed a domestic analyst

============================================================
** generated with diversity 0.80 **
consumers may want to move their telephones a little closer to the tv set
the most remarkable questions is mr. smith 's heart and peter <unk> a <unk> in the <unk> inventories and philosophy of the N people who <unk> the facts the <unk> of the business
there is a mystery <unk> for the <unk>
but there 's no question that the <unk> rally can become a major <unk> <unk> because it was n't the only thing
that 's an <unk> to <unk> at the hands of what i think is a united leader
the handling of those words is that the <unk> <unk> can be taken over and down

============================================================
** generated with diversity 0.60 **
consumers may want to move their telephones a little closer to the tv set
but in the past N years mr. <unk> has said he had n't seen any <unk> of the <unk> <unk>
the company 's a <unk> 's <unk> of <unk> <unk> the <unk> of the <unk> and <unk> of the <unk> <unk>
the <unk> <unk> and the <unk> of <unk> and <unk> were n't <unk>
for the first time in N they say the <unk> not to go into a <unk>
now this is the <unk> <unk> <unk> <unk> on a new <unk> <unk> <unk> on the <unk> and <unk>
the <unk> <unk> <unk> <unk> and
```
