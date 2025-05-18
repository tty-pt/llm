# llm helper tools
Just some simple tooling help out experimenting with llms using llama.cpp and others.

## 🚀 Build / Install requirements
```sh
cd ~
git clone --recurse-submodules https://github.com/tty-pt/llm
cd llm
. source.sh
./init.sh
```

Use llm-chat G* and you'll start talking to the model if all went well.

> 💡 Add . $HOME/llm/source.sh to .bashrc to auto-source.

## Available tools:
### `llm-list [<filter>]`
> Lists available models in the default huggingpath download cache
### `llm-zero [<filter>]`
> Zero out models you don't need to free up space but still be able to finish a full gguf repo download
### `llm-chat <wildcard>`
> Chat with a model
### `llm-path <wildcard>`
> Get a model's path from a few characters
### `llm-hug <arguments...>`
> Evoke huggingface-cli. Example:
```sh
llm-hug download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q6_K.gguf
```
