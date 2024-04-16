#!/bin/bash

dir="conf/exp/test"

for file in "$dir"/*
do
    filename=$(basename -- "$file")
    echo "Processing $filename"
    python test.py +exp=test/$filename
done