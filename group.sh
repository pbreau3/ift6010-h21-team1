#!/bin/bash

# Concatenates all files into one
if [ $# -ne 2 ]; then
    echo "Usage: ./group.sh training-folder name-of-grouped-file"
    echo "Ex: ./group.sh \"data/JSB Chorales/train/text\" musicautobot"
    exit 1
fi

train="$1"
name="$2"

# group all training files into one
mkdir -p model/group
cat "$train"/*.txt > model/group/$name