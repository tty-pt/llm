#include <llama.h>
#include <math.h>
#include <ndc.h>
#include <qdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define END "<|im_end|>"
#define MAX_TOKENS 1024

struct llama_model * model = NULL;
struct llama_context * ctx = NULL;
struct llama_sampler * sampler = NULL;
const struct llama_vocab * vocab;

char output[64 * MAX_TOKENS], *o;
llama_token tokens[MAX_TOKENS];
int pos;

void quiet_logger(enum ggml_log_level level __attribute__((unused)), const char * text __attribute__((unused)), void * user_data __attribute__((unused))) {
    // Do nothing
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

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 64;
    ctx_params.n_seq_max = 4;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to init context\n");
        exit(1);
    }

    llama_kv_self_clear(ctx);
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));

    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    vocab = llama_model_get_vocab(model);
}

static inline int
inference(void) {
	llama_token tok = llama_sampler_sample(sampler, ctx, -1);
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

	char buf[64];
	memset(buf, 0, sizeof(buf));
	llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
	o += snprintf(o, sizeof(output) - (o - output), "%s", buf);

	if (!strcmp(o - 10, END)) {
		*(o - 10) = '\0';
		return 0;
	}

	struct llama_batch b = llama_batch_init(1, 0, 1);
	b.n_tokens = 1;
	b.token[0] = tok;
	b.pos[0] = pos++;
	b.n_seq_id[0] = 1;
	b.seq_id[0] = malloc(sizeof(llama_seq_id));
	b.seq_id[0][0] = 0;
	b.logits[0] = 1;

	if (llama_decode(ctx, b) != 0) {
		fprintf(stderr, "\nDecode failed\n");
		llama_batch_free(b);
		return 0;
	}

	llama_batch_free(b);
	return 1;
}

void generate(int fd, const char * prompt) {
    int i;
    o = output;

    llama_kv_self_clear(ctx);
    memset(tokens, 0, sizeof(llama_token) * MAX_TOKENS);

    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, MAX_TOKENS, true, true);
    if (n_tokens < 1) {
        ndc_writef(fd, "Tokenization failed (%d)\n", n_tokens);
        exit(1);
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = malloc(sizeof(llama_seq_id));
        batch.seq_id[i][0] = 0;
    }
    batch.logits[i - 1] = 1;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode prompt\n");
        llama_batch_free(batch);
	return;
    }

    llama_batch_free(batch);

    int max_gen = MAX_TOKENS;
    pos = n_tokens;

    for (int step = 0; step < max_gen && inference(); ++step)
	    ;

    ndc_writef(fd, "%s", output);
}

void do_ASK(int fd, int argc, char *argv[]) {
	char buf[BUFSIZ * 2], *b = buf;
	b += snprintf(b, sizeof(buf) - (b - buf), "<|im_start|>user\n");
	for (int i = 0; i < argc; i++) {
		int ret = snprintf(b, sizeof(buf) - (b - buf), "%s", argv[i]);
		if (ret < 0) {
			ndc_writef(fd, "Buffer size exceeded\n");
			return;
		}
		b += ret;
	}
	b += snprintf(b, sizeof(buf) - (b - buf), "<|im_end|>\n<|im_start|>assistant\n");
	generate(fd, buf);
	ndc_writef(fd, "<|im_end|>\n");
}

struct cmd_slot cmds[] = {
	{
		.name = "ask",
		.cb = &do_ASK,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = NULL
	}
};

void
ndc_update(unsigned long long dt __attribute__((unused)))
{
}

void
ndc_command(int fd __attribute__((unused)), int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
}

void
ndc_flush(int fd __attribute__((unused)), int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
}

void
ndc_vim(int fd __attribute__((unused)), int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
}

int
ndc_connect(int fd __attribute__((unused))) {
	return 0;
}

void
ndc_ws_init(int fd __attribute__((unused))) {
}

void
ndc_disconnect(int fd __attribute__((unused))) {
}

char *
ndc_auth_check(int fd __attribute__((unused))) {
	return NULL;
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

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) {
		switch (c) {
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
	}

	qdb_init();
	ndc_pre_init(&config);

	optind = 1;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) {
		switch (c) {
		case 'K':
			ndc_certs_add(optarg);
			break;

		case 'k':
			ndc_cert_add(optarg);
			break;

		default: break;
		}
	}

	setup(argv[argc - 1]);
	ndc_init();
	int ret = ndc_main();
	llama_sampler_free(sampler);
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return ret;
}
