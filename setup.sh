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

# Packages, libraries
pip install mido
apt install fluidsynth

# Set up fluidsynth
echo "/usr/share/sounds/sf2/FluidR3_GM.sf2" > sound_font_path.txt