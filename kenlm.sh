#!/bin/bash
shopt -s expand_aliases

# Must change this if your compiled kenlm is somewhere else!!!
chmod +x ../kenlm/build/bin/*
alias lmplz="../kenlm/build/bin/lmplz"
alias build_binary="../kenlm/build/bin/build_binary"

# Script that creates a kenlm n-gram model
if [ $# -ne 2 ]; then
    echo "Usage: ./kenlm.sh order-of-n-gram one-grouped-training-file"
    echo "./kenlm.sh 2 model/group/musicautobot"
    exit 1
fi

order="$1"
grouped="$2"

# Train
arpa="model/kenlm/order$order.arpa"
lmplz -o "$order" -T /tmp/ --discount_fallback < $grouped > $arpa

# Build binary
binary="model/kenlm/order$order.bin"
build_binary $arpa $binary -s