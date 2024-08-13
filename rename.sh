#!/bin/bash

# Directory to search
DIRECTORY="configs/exp"

# Find all files in the directory and rename them
find "$DIRECTORY" -type f -name '*multi*' | while read FILE; do
    NEW_FILE=$(echo "$FILE" | sed 's/multi/mslt/g')
    mv "$FILE" "$NEW_FILE"
done

echo "Renaming complete."