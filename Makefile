npm-lib := @tty-pt/qdb @tty-pt/ndc

-include node_modules/@tty-pt/mk/include.mk
-include .config.mk

all: bin/llm-askd bin/llm-ask bin/llm-chat

LDLIBS-${CONFIG-cuda} += -lcuda -lcudart -lcublas \
	-lggml-cuda -lggml-base

LDLIBS-${CONFIG-blas} += -lblas -llapack

CFLAGS-${CONFIG-cuda} := -DCONFIG_CUDA

LDFLAGS := ${LDLIBS-y} ${LDFLAGS}
FLAGS := ${LDFLAGS} ${CFLAGS} ${CFLAGS-y} ${XCOMPILER}

bin/llm-askd: src/llm-askd.c
	${CC} src/llm-askd.c -o $@ -lllama -lqdb -lndc ${FLAGS}

bin/llm-ask: src/llm-ask.c
bin/llm-chat: src/llm-chat.c

bin/llm-ask bin/llm-chat:
	${CC} ${@:bin/%=src/%.c} -o $@ ${FLAGS}

clean:
	rm bin/llm-ask bin/llm-askd 2>/dev/null || true

.PHONY: clean
