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
