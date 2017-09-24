#!/bin/sh -x

rm -rf ./data/livedoor
mkdir -p ./data/livedoor
wget "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
tar xvzf ldcc-20140209.tar.gz
mv text ./data/livedoor
rm ldcc-20140209.tar.gz
