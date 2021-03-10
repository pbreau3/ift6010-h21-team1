#!/bin/bash

# Requires fluidsynth

# Usage: ./midi2wav.sh midi-file wav-name
if [ $# -ne 2 ]; then
    echo Usage: ./midi2wav.sh midi-file wav-name
    exit 1
fi

# Converts a midi to a wav for listening on colab
fluidsynth -ni `cat sound_font_path.txt` $1 -F $2 -r 44100