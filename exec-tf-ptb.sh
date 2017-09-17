#!/bin/sh -x

python ./tf-ptb/ptb_word_lm.py --data_path=./data/ptb/simple-examples/data --save_path=./log/tf-ptb --model=medium
