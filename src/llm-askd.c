#include <ctype.h>
#include <llama.h>
#include <math.h>
#include <ndc.h>
#include <qdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if CONFIG_CUDA
#include <ggml-cuda.h>
#include <gguf.h>
#include <cuda_runtime.h>
#endif

#define MAX_TOKENS 256
#define MAX_MEMORY (MAX_TOKENS * 10)
#define MAX_TOKEN_LEN 64
#define DEFAULT_SEQ_MAX 4

struct llama_model * model = NULL;
struct llama_context_params ctx_params;
struct llama_sampler * sampler = NULL;
const struct llama_vocab * vocab;

llama_token tokens[MAX_TOKENS];

static llama_seq_id seq_ids[MAX_TOKENS];

const char *start = "<|im_start|>";

const char *end = "<|im_end|>";
const unsigned end_len = 10;

typedef struct fd_info {
	char line_buf[BUFSIZ * 4];
	struct llama_context *ctx;
	unsigned pos, end_pos, line_pos;
} fdi_t;

fdi_t fdis[FD_SETSIZE], general;

void quiet_logger(enum ggml_log_level level __attribute__((unused)), const char * text __attribute__((unused)), void * user_data __attribute__((unused))) {
	// Do nothing
}

static inline void
fdi_init(fdi_t *fdi) {
	struct llama_context *ctx =
		llama_init_from_model(model, ctx_params);
	fdi->ctx = ctx;
	if (!fdi->ctx)
		ndclog(LOG_ERR, "Failed to init context\n");
	llama_kv_self_clear(fdi->ctx);
	fdi->line_pos = fdi->end_pos = fdi->pos = 0;
	memset(fdi->line_buf, 0, sizeof(fdi->line_buf));
}

#if CONFIG_CUDA
static int auto_ngl(const char *path, int gpu, int n_ctx)
{
	size_t free_b, total_b;
	ggml_backend_cuda_get_device_memory(gpu, &free_b, &total_b);

	size_t reserve = (total_b <= (4ULL << 30)) ? (512ULL << 20) :
		(total_b <= (8ULL << 30)) ? (800ULL << 20) : (1024ULL << 20);
	size_t safety = 512ULL << 20;

	if (free_b <= reserve + safety)
		ndclog_err("Not enough free GPU memory to reserve/safety\n");

	size_t usable = free_b - reserve - safety;

	struct gguf_init_params ip = { .no_alloc = true };
	struct gguf_context *ctx = gguf_init_from_file(path, ip);

	if (!ctx)
		ndclog_err("Failed to load GGUF file\n");

	int n_layers = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
	int n_embd   = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
	int n_tensors = gguf_get_n_tensors(ctx);

	size_t *layer_sizes = calloc(n_layers, sizeof(*layer_sizes));
	if (!layer_sizes) {
		gguf_free(ctx);
		ndclog_err("Failed to allocate memory for layer sizes\n");
	}

	size_t global = 0;
	for (int i = 0; i < n_tensors; i++) {
		const char *name = gguf_get_tensor_name(ctx, i);
		if (!strstr(name, "blk.") && !strstr(name, "layers.") && !strstr(name, "block."))
			global += gguf_get_tensor_size(ctx, i);
	}

	usable = (usable > global) ? usable - global : 0;

	for (int i = 0; i < n_tensors; i++) {
		const char *name = gguf_get_tensor_name(ctx, i);
		const char *p = strstr(name, "blk.");
		if (!p) p = strstr(name, "layers.");
		if (!p) p = strstr(name, "block.");
		if (!p) continue;

		while (*p && !isdigit(*p)) p++;
		if (!isdigit(*p)) continue;

		int l = strtol(p, NULL, 10);
		if (l >= 0 && l < n_layers)
			layer_sizes[l] += gguf_get_tensor_size(ctx, i);
	}

	gguf_free(ctx);

	size_t kv_size = (size_t)n_ctx * n_embd * 4;
	size_t used = 0;
	int ngl = 0;

	for (int i = 0; i < n_layers; i++) {
		size_t need = layer_sizes[i] + kv_size;
		if (used + need > usable)
			break;
		used += need;
		ngl++;
	}

	free(layer_sizes);
	return ngl;
}
#endif

inline static void
setup_model(const char *model_path)
{
	struct llama_model_params model_params
		= llama_model_default_params();

#if CONFIG_CUDA
	if (llama_supports_gpu_offload()) {

		int dev = 0;
		cudaError_t err = cudaSetDevice(dev);
		if (err != cudaSuccess) {
			ndclog(LOG_ERR, "Failed to set CUDA device %d: %s\n", dev, cudaGetErrorString(err));
			ndclog(LOG_ERR, "Disabling GPU offload.\n");
			model_params.n_gpu_layers = 0;
		} else {
			model_params.n_gpu_layers = auto_ngl(model_path, dev, MAX_MEMORY);
			model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
		}
	}
#endif

	model = llama_model_load_from_file(
			model_path, model_params);

	if (!model)
		ndclog_err("Failed to load model\n");
}

inline static void
setup_context(void)
{
	ctx_params = llama_context_default_params();
	ctx_params.n_ctx = MAX_MEMORY;
	ctx_params.n_batch = MAX_TOKENS;
	ctx_params.n_seq_max = DEFAULT_SEQ_MAX;
	ctx_params.n_threads = sysconf(_SC_NPROCESSORS_ONLN) >> 2;

#if CONFIG_CUDA
	if (llama_supports_gpu_offload()) {
		ctx_params.op_offload = true;
		ctx_params.offload_kqv = true;
	}
#endif
}

inline static void
setup_sampler(void)
{
	struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();

	sampler = llama_sampler_chain_init(chain_params);
	/* llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40)); */
	/* llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1)); */
	/* llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7)); */
	llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
}

static void
setup(const char *model_path)
{
	cudaDeviceReset();
	llama_log_set(quiet_logger, NULL);
	llama_backend_init();
	setup_model(model_path);
	setup_context();
	fdi_init(&general);
	vocab = llama_model_get_vocab(model);
	setup_sampler();
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
memorize(fdi_t *fdi, size_t len) {
	unsigned pos = fdi->pos;
	int excess = pos + len - MAX_MEMORY;
	if (excess > 0)
		pos -= excess;

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
		ndclog(LOG_ERR, "Decode failed\n");
		return 0;
	}

	fdi->pos += len;
	return len;
}

static inline void
commit(int fd, fdi_t *fdi, const char *prompt) {
	int n_tokens = tokenize(fd, prompt);
	memorize(fdi, n_tokens);
	/* ndclog(LOG_INFO, "commit: '%s'\n", prompt); */
}

void cmd_cb(
	int fd,
	char *buf,
	size_t len,
	int ofd __attribute__((unused)))
{
	fdi_t *fdi = &fdis[fd];
	ndc_write(fd, buf, len);
	commit(fd, fdi, buf);
}

static inline void
cmd_exec(int fd, fdi_t *fdi) {
	if (!fdi->line_pos)
		return;

	char *pound = strstr(fdi->line_buf, "$ ");
	if (!pound)
		return;

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
	if (space)
		*space = '\0';

	ndc_exec(fd, args, cmd_cb, NULL, 0);
	commit(fd, fdi, ndc_execbuf);
}

static inline int
toktok(const llama_token *buffer, int len,
		const llama_token *needle, int needle_len)
{
	if (len < needle_len)
		return 0;

	for (int i = 0; i < needle_len; i++)
		if (buffer[len - needle_len + i] != needle[i])
			return 0;

	return needle_len;
}

static inline int
inference(int fd, fdi_t *fdi) {
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
	int ret = 1;
	tokens[0] = tok;
	memorize(fdi, 1);

	size_t buflen = strlen(buf);
	char *eoim = strchr(buf, *(end + fdi->end_pos));
	/* ndclog(LOG_ERR, "tok mem '%s' eoim '%s' clen %lu\n", buf, eoim, buflen - (eoim - buf)); */
	if (eoim && (eoim - buf) <= end_len - fdi->end_pos) {
		size_t clen = buflen - (eoim - buf);

		if (strncmp(eoim, end + fdi->end_pos, clen))
			goto end;

		fdi->end_pos += clen;

		if (fdi->end_pos < end_len)
			return 1;

		fdi->end_pos = 0;
		return 0;
	}

end:	if (fdi->end_pos)
		ndc_write(fd, (void *) end, fdi->end_pos);

	fdi->end_pos = 0;
	ndc_write(fd, buf, buflen);
	snprintf(&fdi->line_buf[fdi->line_pos], sizeof(fdi->line_buf) - fdi->line_pos, "%s", buf);
	fdi->line_pos += buflen;

	if (strrchr(buf, '\n')) {
		cmd_exec(fd, fdi);
		fdi->line_pos = 0;
	}

	return ret;
}

void generate(int fd, const char * prompt) {
	fdi_t *fdi = &fdis[fd];
	int max_gen = MAX_TOKENS;

	commit(fd, fdi, prompt);

	fdi->line_pos = 0;
	for (
			int step = 0;
			step < max_gen
			&& inference(fd, fdi);
			++step )
		;

	cmd_exec(fd, fdi);
	fdi->line_pos = 0;
}

void do_ASK(int fd, int argc, char *argv[]) {
	char buf[BUFSIZ * 2], *b = buf;
	b += snprintf(b, sizeof(buf) - (b - buf), "%suser\n", start);
	for (int i = 1; i < argc; i++) {
		int ret = snprintf(b, sizeof(buf) - (b - buf), " %s", argv[i]);
		if (ret < 0) {
			ndc_writef(fd, "Buffer size exceeded\n");
			return;
		}
		b += ret;
	}
	b += snprintf(b, sizeof(buf) - (b - buf), "%s\n%sassistant\n ", end, start);
	generate(fd, buf);
	ndc_writef(fd, "%s\n", end);
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

	struct stat st;
	char *arg_model = argv[argc - 1];
	char model_path[BUFSIZ];

	if (stat(arg_model, &st) == 0 && S_ISREG(st.st_mode)) {
		snprintf(model_path, sizeof(model_path),
				"%s", arg_model);
	} else {
		FILE *fp;
		char cmd[BUFSIZ];
		snprintf(cmd, sizeof(cmd),
				"llm-path %s", arg_model);
		fp = popen(cmd, "r");
		if (!fp || !fgets(model_path, sizeof(model_path), fp))
			ndclog_err("Couldn't resolve model\n");
		pclose(fp);

		char *nl = strchr(model_path, '\n');
		if (nl) *nl = '\0';
		arg_model = model_path;
	}

	setup(arg_model);
	llama_print_system_info();
	ndc_init();
	int ret = ndc_main();
	llama_sampler_free(sampler);
	llama_free(general.ctx);
	llama_model_free(model);
	llama_backend_free();
	return ret;
}
