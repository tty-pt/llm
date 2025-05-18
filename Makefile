libdir := /usr/local
LDFLAGS := ${libdir:%=-L%/lib} ${libdir:%=-Wl,-rpath,%/lib}
dm_daemon2: dm_daemon2.c
	${CC} dm_daemon2.c -o $@ -g -pthread -ldl /usr/local/lib/libllama.so ${LDFLAGS}
dm_daemon: dm_daemon.c
	${CC} dm_daemon.c -o $@ -pthread -ldl /usr/local/lib/libllama.so ${LDFLAGS}
