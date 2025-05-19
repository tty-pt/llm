#!/bin/sh
REPO="unsloth/Qwen3-30B-A3B-GGUF"
FILENAME="Qwen3-30B-A3B-Q6_K.gguf"
npm i
make
. ./source.sh
python3 -m venv .venv
./.venv/bin/pip install -r llama.cpp/requirements.txt
cd llama.cpp
mkdir build
cd build
cmake ..
make -j4
read -p "Download $REPO / $FILENAME? " y
case $y in
	y)
		llm-hug download "$REPO" "$FILENAME"
		;;
esac
