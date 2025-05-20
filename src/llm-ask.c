#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

#define PORT 4242
#define BUF_SIZE 4096
#define END_TAG "<|im"

int main(int argc, char *argv[]) {
    char prompt[BUFSIZ * 2], *p = prompt;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <prompt>\n", argv[0]);
        return 1;
    }

    p += snprintf(p, sizeof(prompt) - (p - prompt), "ask" );

    for (int i = 1; i < argc; i++) {
	    int ret = snprintf(p, sizeof(prompt) - (p - prompt),
			    " %s" , argv[i]);

	    if (ret <= 0)
		    break;

	    p += ret;
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(sock);
        return 1;
    }

    send(sock, prompt, p - prompt + 1, 0);

    char response[BUF_SIZE];
    ssize_t n;
    while ((n = recv(sock, response, sizeof(response) - 1, 0)) > 0) {
        response[n] = '\0';
	char *end = strstr(response, END_TAG);
        if (end) {
		*end++ = '\n';
		*end = '\0';
		fputs(response, stdout);
		break;
	}
        fputs(response, stdout);
    }

    close(sock);
    return 0;
}
