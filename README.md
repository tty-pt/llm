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
### `llm-askd <model>`
> Run a daemon service to query the model
### `llm-ask <prompt>`
> Ask questions and get answers. Example:
```sh
$ ./llm-askd `llm-path *Q6*`
Running from cwd
~/llm$ ./llm-ask "Hello"
Hello! How can I help you today?

~/llm$ ./llm-ask "What's the capital of Portugal?"
The capital of Portugal is Lisbon.
~/llm$ ./llm-ask "What's the capital of Israel?"
The capital of Israel is Jerusalem.
~/llm$ ./llm-ask "In bash, how do I delete all title.txt files in subdirectories of the current dir?"
To delete all title.txt files in subdirectories of the current directory, you can use the following command in the terminal:
\```
find . -type f -name "title.txt" -exec rm {} \;
\```
This command uses the `find` command to search for all files (`-type f`) in the current directory (`.`) and its subdirectories that have the name "title.txt" (`-name "title.txt"`). The `-exec rm {} \;` option then executes the `rm` command to delete each file found.

Note that this command will delete all title.txt files, regardless of their contents or importance. If you want to be more selective about which files to delete, you can modify the `find` command to include additional criteria, such as file size or modification time.
```
