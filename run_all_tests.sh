#!/bin/bash

dir="conf/exp/test/xstorycloze"

for file in "$dir"/*
    do
    filename=$(basename -- "$file")
    echo "Testing $filename"
    python test.py +exp=test/xstorycloze/$filename
    done
#python merge.py