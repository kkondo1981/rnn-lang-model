#!/bin/sh -x

rm -rf ./data/ptb
mkdir -p ./data/ptb
wget "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
tar xvzf simple-examples.tgz
mv simple-examples ./data/ptb
