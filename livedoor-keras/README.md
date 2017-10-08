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
- mecab-python3: 0.6
- keras-gpu=2.0.5=py36_0

## learning

```
# データセット作成（初回のみ）
./dl-livedoor.sh
python ./make_livedoor_data.py > ./data/livedoor/make_livedoor_data.out

# ディレクトリ作成（作成済みの場合は省略）
mkdir ./log
mkdir ./log/livedoor-keras
mkdir ./model

# 学習スクリプト実行
python ./livedoor-keras/learn.py > ./log/livedoor-keras/train.out 2> ./log/livedoor-keras/train.err &
```

## text generation

```
python ./livedoor-keras/gentext.py
cat ./log/livedoor-keras/gentext.txt
```

## likelihood test

```
python ./livedoor-keras/likelihood.py
cat ./log/livedoor-keras/likelihood.txt
```

## 学習の様子
https://github.com/kkondo1981/rnn-lang-model/issues/5
