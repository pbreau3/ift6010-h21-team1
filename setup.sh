#!/bin/bash

# Prepares the data folder and data sets

if [ ! -d data ]; then
	mkdir data
fi

cd data

# JSB Chorales data set
if [ ! -f "JSB Chorales.zip" ]; then
	wget http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.zip
fi

if [ ! -d "JSB Chorales" ]; then
    unzip "JSB Chorales.zip"
fi

# Maestro data set
## TODO

# Set up fluidsynth
echo "Sound font path is"
cat sound_font_path.txt

# Packages, libraries
pip install mido music21
apt install fluidsynth
apt install musescore