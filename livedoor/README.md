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
# データセット作成（初回のみ）
./dl-livedoor.sh
python ./make_livedoor_data.py > ./data/livedoor/make_livedoor_data.out

# ディレクトリ作成（作成済みの場合は省略）
mkdir ./log
mkdir ./log/livedoor
# mkdir ./model

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

## likelihood test

```
python ./livedoor/likelihood.py
cat ./log/livedoor/likelihood.txt
```
