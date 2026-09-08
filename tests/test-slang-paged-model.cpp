#include "llama-slang-paged.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct abort_control {
    bool enabled = false;
    int calls = 0;
    int abort_after = 0;
};

bool abort_after_progress(void * data) {
    auto & control = *static_cast<abort_control *>(data);
    if (!control.enabled) {
        return false;
    }
    ++control.calls;
    return control.calls >= control.abort_after;
}

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "test-slang-paged-model: %s\n", message);
    std::exit(1);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    const int32_t needed = -llama_tokenize(vocab, text, std::strlen(text), nullptr, 0, true, true);
    if (needed <= 0) {
        fail("token count");
    }
    std::vector<llama_token> result(needed);
    const int32_t count = llama_tokenize(vocab, text, std::strlen(text), result.data(), result.size(), true, true);
    if (count != needed) {
        fail("tokenize");
    }
    return result;
}

bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos start) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens    = tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i + 1 == batch.n_tokens;
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

void compare_logits(llama_context * legacy,
                    llama_context * paged,
                    int32_t         n_vocab,
                    float           max_error_limit,
                    float           normalized_rmse_limit,
                    int             step) {
    const float * expected = llama_get_logits_ith(legacy, -1);
    const float * actual   = llama_get_logits_ith(paged, -1);
    if (!expected || !actual) {
        fail("missing logits");
    }
    float  maximum_error     = 0.0f;
    double squared_error     = 0.0;
    double squared_reference = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            fail("non-finite logit");
        }
        const float error = std::fabs(expected[i] - actual[i]);
        maximum_error     = std::max(maximum_error, error);
        squared_error += double(error) * error;
        squared_reference += double(expected[i]) * expected[i];
    }
    const auto   greedy_expected = std::max_element(expected, expected + n_vocab) - expected;
    const auto   greedy_actual   = std::max_element(actual, actual + n_vocab) - actual;
    const double rmse            = std::sqrt(squared_error / n_vocab);
    const double reference_rms   = std::sqrt(squared_reference / n_vocab);
    const double normalized_rmse = reference_rms == 0.0 ? rmse : rmse / reference_rms;
    if (maximum_error > max_error_limit || normalized_rmse > normalized_rmse_limit ||
        greedy_expected != greedy_actual) {
        std::fprintf(stderr, "step %d, max error %.9g, rmse %.9g, reference rms %.9g, greedy %td != %td\n", step,
                     maximum_error, rmse, reference_rms, greedy_expected, greedy_actual);
        fail("logit parity");
    }
}

std::vector<llama_token> run_parity_case(llama_model *                    model,
                                         const llama_context_params &     cparams,
                                         const llama_slang_paged_params & paged_params,
                                         const llama_vocab *              vocab,
                                         int32_t                          n_vocab,
                                         const char *                     text,
                                         int                              decode_steps) {
    llama_context * legacy = llama_init_from_model(model, cparams);
    llama_context * paged  = llama_init_from_model_slang_paged(model, cparams, paged_params);
    if (!legacy || !paged) {
        fail("parity context creation");
    }

    auto prompt = tokenize(vocab, text);
    if (prompt.size() <= paged_params.page_tokens) {
        fail("prompt does not cross page boundary");
    }
    if (!decode_tokens(legacy, prompt, 0) || !decode_tokens(paged, prompt, 0)) {
        fail("cold prefill");
    }
    compare_logits(legacy, paged, n_vocab, 0.0f, 0.0f, -1);

    for (int step = 0; step < decode_steps; ++step) {
        const float *            logits   = llama_get_logits_ith(legacy, -1);
        const llama_token        token    = std::max_element(logits, logits + n_vocab) - logits;
        std::vector<llama_token> next     = { token };
        const llama_pos          position = prompt.size() + step;
        if (!decode_tokens(legacy, next, position) || !decode_tokens(paged, next, position)) {
            fail("repeated decode");
        }
        compare_logits(legacy, paged, n_vocab, 0.0f, 0.0f, step);
    }

    llama_free(paged);
    llama_free(legacy);
    return prompt;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        fail("expected model path");
    }
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;
    llama_model * model        = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fail("model load");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 64;
    cparams.n_batch              = 64;
    cparams.n_ubatch             = 64;
    cparams.n_seq_max            = 1;
    cparams.type_k               = GGML_TYPE_F32;
    cparams.type_v               = GGML_TYPE_F32;
    cparams.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.offload_kqv          = false;

    llama_slang_paged_params paged_params = { 4, 16, 64u * 1024u * 1024u, 1024u * 1024u };

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    auto                prompt  = run_parity_case(model, cparams, paged_params, vocab, n_vocab,
                                                  "Once upon a time, a small fox found a bright blue stone.", 20);
    run_parity_case(model, cparams, paged_params, vocab, n_vocab,
                    "At midnight, the observatory recorded a quiet signal beyond the winter stars.", 40);

    llama_slang_paged_params limited_params = { 4, 1, 4u * 1024u * 1024u, 1024u * 1024u };
    llama_context *          limited        = llama_init_from_model_slang_paged(model, cparams, limited_params);
    if (!limited) {
        fail("limited context creation");
    }
    std::vector<llama_token> too_many(prompt.begin(), prompt.begin() + 5);
    if (decode_tokens(limited, too_many, 0)) {
        fail("admission exhaustion accepted");
    }
    std::vector<llama_token> one = { prompt.front() };
    if (!decode_tokens(limited, one, 0)) {
        fail("rollback after admission exhaustion");
    }
    llama_memory_clear(llama_get_memory(limited), true);
    if (!decode_tokens(limited, one, 0)) {
        fail("decode after clear");
    }
    llama_memory_seq_add(llama_get_memory(limited), 0, 0, -1, 1);
    if (decode_tokens(limited, one, 1)) {
        fail("unsupported mutation was not latched");
    }
    llama_memory_clear(llama_get_memory(limited), true);
    if (!decode_tokens(limited, one, 0)) {
        fail("clear after unsupported mutation");
    }
    llama_free(limited);

    llama_context * stateful = llama_init_from_model_slang_paged(model, cparams, paged_params);
    if (!stateful || !decode_tokens(stateful, one, 0)) {
        fail("state query context setup");
    }
    if (llama_state_get_size(stateful) != 0) {
        fail("unsupported state size reported success");
    }
    if (!decode_tokens(stateful, one, 1)) {
        fail("state size query poisoned decode");
    }
    llama_free(stateful);

    abort_control        abort_state;
    llama_context_params abort_params = cparams;
    abort_params.abort_callback       = abort_after_progress;
    abort_params.abort_callback_data  = &abort_state;
    llama_context * aborted           = llama_init_from_model_slang_paged(model, abort_params, paged_params);
    llama_context * abort_reference   = llama_init_from_model(model, cparams);
    if (!aborted || !abort_reference) {
        fail("abort context creation");
    }

    std::vector<llama_token> partial_prefix(prompt.begin(), prompt.begin() + 5);
    if (!decode_tokens(aborted, partial_prefix, 0) || !decode_tokens(abort_reference, partial_prefix, 0)) {
        fail("abort prefix prefill");
    }
    const float * reference_logits = llama_get_logits_ith(abort_reference, -1);
    const llama_token retry_token = std::max_element(reference_logits, reference_logits + n_vocab) - reference_logits;
    std::vector<llama_token> retry = { retry_token };
    if (!decode_tokens(abort_reference, retry, partial_prefix.size())) {
        fail("abort reference decode");
    }

    llama_set_causal_attn(aborted, false);
    abort_state.enabled = true;
    abort_state.abort_after = 2;
    if (decode_tokens(aborted, retry, partial_prefix.size())) {
        fail("graph abort accepted");
    }
    if (abort_state.calls < abort_state.abort_after) {
        fail("graph abort occurred before execution progress");
    }
    abort_state.enabled = false;
    if (!decode_tokens(aborted, retry, partial_prefix.size())) {
        fail("populated-prefix rollback after graph abort");
    }
    compare_logits(abort_reference, aborted, n_vocab, 0.0f, 0.0f, 0);
    llama_free(abort_reference);
    llama_free(aborted);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
