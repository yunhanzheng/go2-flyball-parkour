#!/usr/bin/env bash
# Reference installation. See README.md for details.
set -e

conda create -y -n parkour python=3.8
conda activate parkour

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym and
# unpack them next to this repository, then:
cd isaacgym/python && pip install -e . && cd -

cd rsl_rl     && pip install -e . && cd -
cd legged_gym && pip install -e . && pip install -r requirements.txt && cd -
