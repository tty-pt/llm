#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct llama_model * model = NULL;
struct llama_context * ctx = NULL;
struct llama_sampler * sampler = NULL;

void setup(const char * model_path) {
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
}

void generate(const char * prompt) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t max_tokens = 4096;
    int i;
    llama_token *tokens = malloc(sizeof(llama_token) * max_tokens);
    memset(tokens, 0, sizeof(llama_token) * max_tokens);

    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, max_tokens, true, true);
    if (n_tokens < 1) {
        fprintf(stderr, "Tokenization failed (%d)\n", n_tokens);
        exit(1);
    }

    for (i = 0; i < n_tokens; ++i) {
        printf("Token %d = %d (%s)\n", i, tokens[i], llama_vocab_get_text(vocab, tokens[i]));
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

    int max_gen = 128;
    int pos = n_tokens;

    for (int step = 0; step < max_gen; ++step) {
	    llama_token tok = llama_sampler_sample(sampler, ctx, -1);
	    llama_sampler_accept(sampler, tok);

	    if (tok == llama_vocab_eos(vocab)) break;

	    char buf[64];
	    memset(buf, 0, sizeof(buf));
	    llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
	    printf("%s", buf);
	    fflush(stdout);

	    struct llama_batch b = llama_batch_init(1, 0, 1);
	    b.n_tokens = 1;
	    b.token[0] = tok;
	    b.pos[0] = pos++;
	    b.n_seq_id[0] = 1;
	    b.seq_id[0] = malloc(sizeof(llama_seq_id));
	    b.seq_id[0][0] = 0;
	    b.logits[0] = 1;

	    if (llama_decode(ctx, b) != 0) {
		    fprintf(stderr, "\nDecode failed at step %d\n", step);
		    llama_batch_free(b);
		    break;
	    }

	    llama_batch_free(b);
    }

    printf("\n");
    free(tokens);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }

    setup(argv[1]);
    generate(argv[2]);

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
