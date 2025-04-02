#!/bin/bash

set -e

# Generate example scripts pointing to the Essentia models local directory and an example audio 
# file and execute them.

essentia_models_dir=/path/to/essentia-models/
audio_file=/path/to/example_track.mp3

python3 generate_example_scripts.py \
    --force \
    --metadata-base-dir ${essentia_models_dir} \
    --models-base-dir ${essentia_models_dir} \
    --audio-file ${audio_file}

TF_CPP_MIN_LOG_LEVEL=3 python3 test_scripts.py
