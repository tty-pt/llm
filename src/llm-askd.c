#include <llama.h>
#include <math.h>
#include <ndc.h>
#include <qdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_TOKENS 256
#define MAX_MEMORY (MAX_TOKENS * 72)
#define MAX_TOKEN_LEN 64

struct llama_model * model = NULL;
struct llama_context_params ctx_params;
struct llama_sampler * sampler = NULL;
const struct llama_vocab * vocab;

char output[MAX_TOKEN_LEN * MAX_TOKENS], *o;

llama_token tokens[MAX_TOKENS];

static llama_seq_id seq_ids[MAX_TOKENS];

const char *start = "<|im_start|>";
llama_token start_tk[6];
int start_n = 0;

const char *end = "<|im_end|>";
llama_token end_tk[6];
int end_n = 0;

const char *nl = "\n";
llama_token nl_tk[4];
int nl_n = 0;

static char fullexec[BUFSIZ * 64], *fe;

typedef struct fd_info {
	char *line;
	struct llama_context *ctx;
	int pos;
} fdi_t;

fdi_t fdis[FD_SETSIZE], general;

void quiet_logger(enum ggml_log_level level __attribute__((unused)), const char * text __attribute__((unused)), void * user_data __attribute__((unused))) {
	// Do nothing
}

static inline void
fdi_init(fdi_t *fdi) {
	fdi->ctx = llama_init_from_model(model, ctx_params);
	if (!fdi->ctx) {
		fprintf(stderr, "Failed to init context\n");
		exit(1);
	}
	llama_kv_self_clear(fdi->ctx);
	fdi->pos = 0;
	fdi->line = NULL;
}

static inline void
setup(const char * model_path) {
	llama_log_set(quiet_logger, NULL);
	llama_backend_init();

	struct llama_model_params model_params = llama_model_default_params();
	model = llama_model_load_from_file(model_path, model_params);
	if (!model) {
		fprintf(stderr, "Failed to load model\n");
		exit(1);
	}

	ctx_params = llama_context_default_params();
	ctx_params.n_ctx = MAX_MEMORY;
	ctx_params.n_batch = MAX_TOKENS;
	ctx_params.n_seq_max = 4;
	ctx_params.n_threads = sysconf(_SC_NPROCESSORS_ONLN);

	fdi_init(&general);

	struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
	sampler = llama_sampler_chain_init(chain_params);
	llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));

	llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
	llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

	vocab = llama_model_get_vocab(model);
	start_n = llama_tokenize(vocab, start, strlen(start), start_tk, strlen(start), true, true);
	end_n = llama_tokenize(vocab, end, strlen(end), end_tk, strlen(end), true, true);
	nl_n = llama_tokenize(vocab, nl, strlen(nl), nl_tk, strlen(nl), true, true);
}

static inline int
tk_mem(int pos, size_t n) {
	if (pos + n > MAX_MEMORY) {
		int excess = pos + n - MAX_MEMORY;
		pos -= excess;
	}

	pos += n;
	return pos;
}

static inline int
tokenize(int fd, const char *prompt) {
	int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, MAX_TOKENS, true, true);

	if (n_tokens < 1) {
		ndc_writef(fd, "Tokenization failed\n");
		exit(1);
	}

	return n_tokens;
}

static inline int
memorize(fdi_t *fdi, int pos, size_t len) {
	tk_mem(pos, len);

	struct llama_batch batch = llama_batch_init(len, 0, 1);
	batch.n_tokens = len;

	for (register size_t i = 0; i < len; ++i) {
		batch.token[i] = tokens[i];
		batch.pos[i] = pos + i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i] = &seq_ids[i];
		batch.seq_id[i][0] = 0;
	}
	batch.logits[len - 1] = 1;

	if (llama_decode(fdi->ctx, batch) != 0) {
		fprintf(stderr, "Decode failed\n");
		return 0;
	}

	return len;
}

static inline int
commit(int fd, fdi_t *fdi, int pos, const char *prompt) {
	int n_tokens = tokenize(fd, prompt);
	int i = memorize(fdi, pos, n_tokens);
	if (!i)
		return 0;
	pos += i;
	return pos;
}

void cmd_cb(
		int fd __attribute__((unused)),
		char *buf,
		size_t len __attribute__((unused)),
		int ofd __attribute__((unused)))
{
	fe += snprintf(fe, sizeof(fullexec) - (fe - fullexec), "%s", buf);
}

static inline int
cmd_exec(int fd, fdi_t *fdi, int pos) {
	int i;

	if (!fdi->line)
		return pos;

	char *pound = strstr(fdi->line, "$ ");
	if (!pound)
		return pos;

	char argsbuf[BUFSIZ], *space;
	int argc = 0;
	char *args[8];

	snprintf(argsbuf, sizeof(argsbuf), "%s", pound + 2);
	space = argsbuf;

	do {
		args[argc] = space;
		argc++;
		space = strchr(space, ' ');
		if (!space)
			break;
		*space = '\0';
		space++;
	} while (1);

	args[argc] = NULL;
	space = strchr(args[argc - 1], '\n');
	*space = '\0';

	fe = fullexec;
	memset(fullexec, 0, sizeof(fullexec));
	ndc_exec(fd, args, cmd_cb, NULL, 0);
	*(fe - 1) = '\0';

	if (!(i = commit(fd, fdi, pos, fullexec)))
		return 0;

	o += snprintf(o, sizeof(output) - (o - output), "%s", fullexec);
	return pos + i;
}

static inline int
inference(int fd, fdi_t *fdi, int *pos_r) {
	llama_token tok = tokens[0] = llama_sampler_sample(sampler, fdi->ctx, -1);
	int pos = *pos_r;

	llama_sampler_accept(sampler, tok);

	if (tok == llama_vocab_fim_suf(vocab) ||
			tok == llama_vocab_fim_pad(vocab) ||
			tok == llama_vocab_fim_rep(vocab) ||
			tok == llama_vocab_fim_sep(vocab) ||
			tok == llama_vocab_eos(vocab) ||
			tok == llama_vocab_eot(vocab))
	{
		return 0;
	}

	char buf[MAX_TOKEN_LEN];
	memset(buf, 0, sizeof(buf));
	llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
	char *po = o;
	o += snprintf(o, sizeof(output) - (o - output), "%s", buf);

	int ret = 1;
	tokens[0] = tok;
	int i = memorize(fdi, pos, 1);
	if (!i)
		return 0;

	pos += i;

	char *last_nl = strrchr(po, '\n');

	if (last_nl) {
		if (!(i = cmd_exec(fd, fdi, pos)))
			return 0;

		fdi->line = last_nl + 1;
		pos = i;
	}

	if (strstr(o - 6, "<|im_"))
		ret = 0;

	*pos_r = pos;
	return ret;
}

void generate(int fd, const char * prompt) {
	fdi_t *fdi = &fdis[fd];
	int *pos_r = fdi->ctx == general.ctx ? &general.pos : &fdi->pos;
	int pos = *pos_r;
	int i;
	o = output;

	if (!(i = commit(fd, fdi, pos, prompt)))
		return;
	pos += i;

	int max_gen = MAX_TOKENS;

	fdi->line = NULL;
	for (int step = 0; step < max_gen && inference(fd, fdi, &pos); ++step) ;

	snprintf(o, sizeof(output) - (o - output), "%s\n", end);

	o = output;
	while (*o == ' ')
		o++;

	*pos_r = pos;
}

void do_ASK(int fd, int argc, char *argv[]) {
	char buf[BUFSIZ * 2], *b = buf;
	b += snprintf(b, sizeof(buf) - (b - buf), "%suser\n ", start);
	for (int i = 1; i < argc; i++) {
		int ret = snprintf(b, sizeof(buf) - (b - buf), " %s", argv[i]);
		if (ret < 0) {
			ndc_writef(fd, "Buffer size exceeded\n");
			return;
		}
		b += ret;
	}
	b += snprintf(b, sizeof(buf) - (b - buf), "\n%s\n%sassistant\n ", end, start);
	generate(fd, buf);
	ndc_writef(fd, "%s\n%s\n", o, end);
}

void do_CHAT(int fd, int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
	fdi_init(&fdis[fd]);
}

struct cmd_slot cmds[] = {
	{
		.name = "ask",
		.cb = &do_ASK,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = "chat",
		.cb = &do_CHAT,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = NULL
	}
};

int
ndc_accept(int fd) {
	fdis[fd].ctx = general.ctx;
	return 0;
}

void
ndc_disconnect(int fd __attribute__((unused))) {
	fdi_t *fdi = &fdis[fd];

	if (fdi->ctx != general.ctx)
		llama_free(fdi->ctx);
}

void
usage(char *prog) {
	fprintf(stderr, "Usage: %s [-dr?] [-C PATH] [-u USER] [-k PATH] [-c PATH] [-p PORT] MODEL\n", prog);
	fprintf(stderr, "    Options:\n");
	fprintf(stderr, "        -C PATH   changes directory to PATH before starting up.\n");
	fprintf(stderr, "        -u USER   login as USER before starting up.\n");
	fprintf(stderr, "        -k PATH   specify SSL certificate 'key' file\n");
	fprintf(stderr, "        -c PATH   specify SSL certificate 'crt' file\n");
	fprintf(stderr, "        -p PORT   specify server port (defaults to 4242)\n");
	fprintf(stderr, "        -d        don't detach\n");
	fprintf(stderr, "        -r        root multiplex mode\n");
	fprintf(stderr, "        -?        display this message.\n");
}

int
main(int argc, char *argv[])
{
	struct ndc_config config = {
		.flags = NDC_DETACH,
		.port = 4242,
	};
	register char c;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) switch (c) {
		case 'd':
			config.flags &= ~NDC_DETACH;
			break;

		case 'K':
		case 'k': break;

		case 'C':
			  config.chroot = strdup(optarg);
			  break;

		case 'r':
			  config.flags |= NDC_ROOT;
			  break;

		case 'p':
			  config.port = atoi(optarg);
			  break;

		case 's':
			  config.ssl_port = atoi(optarg);
			  break;

		default:
			  usage(*argv);
			  return 1;
	}

	qdb_init();
	ndc_pre_init(&config);

	optind = 1;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) switch (c) {
		case 'K':
			ndc_certs_add(optarg);
			break;

		case 'k':
			ndc_cert_add(optarg);
			break;

		default: break;
	}

	setup(argv[argc - 1]);
	ndc_init();
	int ret = ndc_main();
	llama_sampler_free(sampler);
	llama_free(general.ctx);
	llama_model_free(model);
	llama_backend_free();
	return ret;
}
