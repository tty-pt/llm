# llm helper tools
Just some simple tooling help out experimenting with llms using llama.cpp and others.

## 🚀 Build / Install requirements
Prepare the repo:
```sh
cd ~
git clone --recurse-submodules https://github.com/tty-pt/llm
cd llm
./init.sh
```

> 💡 Add . $HOME/llm/source.sh to .bashrc
> to use the binaries from verywhere.

Dowload a model:
```sh
llm-hug download bartowski/Mistral-7B-Instruct-v0.3-GGUF Mistral-7B-Instruct-v0.3-Q2_K.gguf
```

Run the service and start chatting with the model:
```sh
llm-askd `llm-path *v0.3-Q2*`
llm-chat
```

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
### `llm-askd <model>`
> Run a daemon service to query the model

### `llm-ask <prompt>`
> Ask questions and get answers. Example:
