#include <llama.h>
#include <math.h>
#include <ndc.h>
#include <qdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_TOKENS 512
#define MAX_MEMORY (MAX_TOKENS * 20)
#define MAX_TOKEN_LEN 64

struct llama_model * model = NULL;
struct llama_context_params ctx_params;
struct llama_sampler * sampler = NULL;
const struct llama_vocab * vocab;

char output[MAX_TOKEN_LEN * MAX_TOKENS], *o;

llama_token tokens[MAX_TOKENS];
int n_tokens = 0;

static llama_seq_id seq_ids[MAX_TOKENS];

const char *start = "<|im_start|>";
llama_token start_tk[6];
int start_n = 0;

const char *end = "<|im_end|>";
llama_token end_tk[6];
int end_n = 0;

typedef struct fd_info {
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
	start_n = llama_tokenize(vocab, start, strlen(start), start_tk, 4, true, true);
	end_n = llama_tokenize(vocab, end, strlen(end), end_tk, 4, true, true);
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
inference(fdi_t *fdi, int pos) {
	llama_token tok = tokens[0] = llama_sampler_sample(sampler, fdi->ctx, -1);
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
	o += snprintf(o, sizeof(output) - (o - output), "%s", buf);

	n_tokens = 1;
	pos = tk_mem(pos, 1);

	struct llama_batch b = llama_batch_init(1, 0, 1);
	b.n_tokens = 1;
	b.token[0] = tok;
	b.pos[0] = pos++;
	b.n_seq_id[0] = 1;
	b.seq_id[0] = &seq_ids[0];
	b.seq_id[0][0] = 0;
	b.logits[0] = 1;
	int ret = 1;

	if (llama_decode(fdi->ctx, b) != 0) {
		fprintf(stderr, "\nDecode failed\n");
		ret = 0;
	}

	if (strstr(o - 6, "<|im_"))
		ret = 0;

	return ret;
}

static inline int
tokenize(int fd, const char *prompt) {
	n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, MAX_TOKENS, true, true);

	if (n_tokens < 1) {
		ndc_writef(fd, "Tokenization failed\n");
		exit(1);
	}

	return n_tokens;
}

void generate(int fd, const char * prompt) {
	fdi_t *fdi = &fdis[fd];
	int *pos_r = fdi->ctx == general.ctx ? &general.pos : &fdi->pos;
	int pos = *pos_r;
	int i;
	o = output;

	tokenize(fd, prompt);
	tk_mem(pos, n_tokens);

	struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;

	for (i = 0; i < n_tokens; ++i) {
		batch.token[i] = tokens[i];
		batch.pos[i] = pos++;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i] = &seq_ids[i];
		batch.seq_id[i][0] = 0;
	}
	batch.logits[i - 1] = 1;

	if (llama_decode(fdi->ctx, batch) != 0) {
		fprintf(stderr, "Failed to decode prompt\n");
		return;
	}

	int max_gen = MAX_TOKENS;

	for (int step = 0; step < max_gen && inference(fdi, pos); ++step)
		pos++;

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
