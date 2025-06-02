#!/bin/sh
REPO="unsloth/Qwen3-30B-A3B-GGUF"
FILENAME="Qwen3-30B-A3B-Q6_K.gguf"
dir="$PWD"
npm i
make
. ./source.sh
conda env create -f environment.yaml
conda activate axo
cd llama.cpp
mkdir build
cd build
./../../llama-setup.sh
echo ". $PWD/.bash_completion >> $HOME/.bash_completion"
