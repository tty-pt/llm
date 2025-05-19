npm-lib := @tty-pt/qdb @tty-pt/ndc

-include node_modules/@tty-pt/mk/include.mk

all: bin/llm-askd bin/llm-ask

FLAGS := ${LDFLAGS} ${CFLAGS}

bin/llm-askd: src/llm-askd.c
	${CC} src/llm-askd.c -o $@ -lllama -lqdb -lndc ${FLAGS}

bin/llm-ask: src/llm-ask.c
	${CC} src/llm-ask.c -o $@ ${FLAGS}

clean:
	rm bin/llm-ask bin/llm-askd 2>/dev/null || true

.PHONY: clean
