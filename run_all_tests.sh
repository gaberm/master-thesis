#!/bin/bash

dir="conf/exp/test"

for file in "$dir"/*
do
    filename=$(basename -- "$file")
    echo "Testing $filename"
    python test.py +exp=test/$filename
    python merge.py
done