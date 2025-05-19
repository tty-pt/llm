#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

#define PORT 4242
#define BUF_SIZE 4096
#define END_TAG "<|im_end|>"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <prompt>\n", argv[0]);
        return 1;
    }

    const char *prompt = argv[1];

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

    char request[BUF_SIZE];
    snprintf(request, sizeof(request), "ask %s\n", prompt);
    send(sock, request, strlen(request), 0);

    char response[BUF_SIZE];
    ssize_t n;
    while ((n = recv(sock, response, sizeof(response) - 1, 0)) > 0) {
        response[n] = '\0';
	char *end = strstr(response, END_TAG);
        if (end) {
		*end = '\0';
		fputs(response, stdout);
		break;
	}
        fputs(response, stdout);
    }

    close(sock);
    return 0;
}
