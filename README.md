# Music generation project for IFT 6010
This is a music generation machine learning project for the class of "Natural language processing".

## Getting Started
To download the data sets, use `setup.sh` for the JSB data set. A segment of the MAESTRO data set is already in the repo. The full MAESTRO data set is too large and can be found [here](https://magenta.tensorflow.org/datasets/maestro).

## Installation
`setup.sh` will install all the necessary modules. They are
* mido
* fluidsynth

## Scripts
* `setup.sh`: Handles dependencies for this project.

* `midi2wav.sh`: Converts a given midi file into `.wav` file. Takes two arguments, requires `fluidsynth` installed through `setup.sh`.

* `quantize.py`: Converts a given midi file into a quantized version.

* `preprocess.py`: Converts a quantized file into a format that machines can learn on.

* `evaluate.py`: Evaluates the performance of a given model.