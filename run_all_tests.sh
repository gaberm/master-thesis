#!/bin/bash

dir="configs/exp"

for file in "$dir"/*; do
    filename=$(basename -- "$file")
    echo "Testing $filename"
    python test.py +exp=test/$filename
done
python merge.py