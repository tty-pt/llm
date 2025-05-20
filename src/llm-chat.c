#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

#define PORT      4242
#define END_TAG   "<|im"

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
	int sock;
	struct sockaddr_in server_addr;
	char buf[BUFSIZ];
	char msg[BUFSIZ];
	char response[BUFSIZ];

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0) {
		perror("socket");
		return 1;
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_port   = htons(PORT);
	server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

	if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
		perror("connect");
		close(sock);
		return 1;
	}

	int mlen = snprintf(msg, sizeof(msg), "chat\n");
	if (send(sock, msg, mlen, 0) != mlen) {
		perror("send");
		return 1;
	}

	printf("Connected! Type your prompts (empty line to quit).\n\n");

	while (1) {
		printf("> ");

		if (!fgets(buf, sizeof(buf), stdin))
			continue;

		size_t len = strlen(buf);
		if (len == 0 || buf[0] == '\n')
			break;

		if (buf[len - 1] == '\n')
			buf[len - 1] = '\0';

		int mlen = snprintf(msg, sizeof(msg), "ask %s\n", buf);

		if ((size_t) mlen >= sizeof(msg)) {
			fprintf(stderr, "Prompt too long\n");
			continue;
		}

		if (send(sock, msg, mlen, 0) != mlen) {
			perror("send");
			break;
		}

		ssize_t cn = recv(sock, response,
				sizeof(response) - 1, 0);

		if (cn <= 0)
			goto done;

		response[cn] = '\0';
		char *end = strstr(response, END_TAG);
		if (end)
			*end = '\0';

		printf("%s\n", response);
	}

done:
	close(sock);
	printf("Session closed.\n");
	return 0;
}
