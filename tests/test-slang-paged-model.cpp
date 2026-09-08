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

void compare_logits_copy(llama_context *              legacy,
                         const std::vector<float> &    actual,
                         int32_t                       n_vocab,
                         const char *                  stage) {
    const float * expected = llama_get_logits_ith(legacy, -1);
    if (!expected || actual.size() != size_t(n_vocab)) {
        fail("external logits missing");
    }
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(expected[i]) || expected[i] != actual[i]) {
            std::fprintf(stderr, "%s: logit %d differs: %.9g != %.9g\n", stage, i, expected[i], actual[i]);
            fail("external logit parity");
        }
    }
}

bool decode_external(llama_context * ctx,
                     int64_t         transaction,
                     const std::vector<llama_token> & tokens,
                     llama_pos       start) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens    = tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i + 1 == batch.n_tokens;
    }
    const bool ok = llama_slang_paged_decode(ctx, transaction, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int64_t reserve_page(llama_context * ctx) {
    const int64_t result = llama_slang_paged_page_reserve(ctx);
    if (result <= 0) {
        fail("external page reserve");
    }
    return result;
}

int64_t begin_table(llama_context * ctx, int64_t request, uint32_t pages) {
    const int64_t result = llama_slang_paged_table_begin(ctx, request, 0, pages);
    if (result <= 0) {
        fail("external table begin");
    }
    return result;
}

void push_page(llama_context * ctx,
               int64_t         transaction,
               int64_t         page,
               uint32_t        valid_rows,
               uint32_t        additional_rows) {
    if (llama_slang_paged_table_push(ctx, transaction, page, valid_rows, additional_rows) != 0) {
        fail("external table push");
    }
}

void copy_committed_logits(llama_context * ctx,
                           int64_t         request,
                           std::vector<float> & destination) {
    if (llama_slang_paged_logits_copy(ctx, request, destination.data(), destination.size()) != 0) {
        fail("external committed logits copy");
    }
}

void run_external_provider_case(llama_model *                model,
                                llama_context_params         cparams,
                                const std::vector<llama_token> & prompt,
                                int32_t                      n_vocab) {
    if (prompt.size() < 7) {
        fail("external prompt too short");
    }

    abort_control abort_state;
    cparams.abort_callback      = abort_after_progress;
    cparams.abort_callback_data = &abort_state;
    llama_slang_paged_provider_params provider = {
        LLAMA_SLANG_PAGED_PROVIDER_ABI_VERSION,
        4,
        8,
        2,
        4,
        64u * 1024u * 1024u,
        1024u * 1024u,
        1024u * 1024u,
        size_t(n_vocab) * sizeof(float) * 2 * 2,
    };
    llama_context * external = llama_init_from_model_slang_paged_external(model, cparams, provider);
    llama_context * legacy_a = llama_init_from_model(model, cparams);
    llama_context * legacy_b = llama_init_from_model(model, cparams);
    if (!external || !legacy_a || !legacy_b) {
        fail("external context creation");
    }

    llama_context * foreign_context = llama_init_from_model_slang_paged_external(model, cparams, provider);
    const int64_t   foreign_request = foreign_context ? llama_slang_paged_request_open(foreign_context) : -1;
    if (!foreign_context || foreign_request <= 0 ||
        llama_slang_paged_table_begin(external, foreign_request, 0, 1) >= 0) {
        fail("foreign request handle accepted");
    }
    if (llama_slang_paged_request_close(foreign_context, foreign_request) != 0) {
        fail("foreign request cleanup");
    }
    llama_free(foreign_context);

    std::vector<llama_token> one = { prompt.front() };
    if (decode_tokens(external, one, 0)) {
        fail("direct decode accepted by external context");
    }

    const int64_t request_a = llama_slang_paged_request_open(external);
    const int64_t request_b = llama_slang_paged_request_open(external);
    if (request_a <= 0 || request_b <= 0 || llama_slang_paged_request_open(external) >= 0) {
        fail("external request admission");
    }

    int64_t transaction = begin_table(external, request_a, 1);
    if (llama_slang_paged_table_push(external, transaction, 0, 0, 1) == 0 ||
        llama_slang_paged_commit(external, transaction) == 0) {
        fail("invalid page did not poison transaction");
    }

    int64_t malformed_page = reserve_page(external);
    transaction            = begin_table(external, request_a, 1);
    push_page(external, transaction, malformed_page, 0, 1);
    llama_batch malformed{};
    malformed.n_tokens = 1;
    if (llama_slang_paged_decode(external, transaction, malformed) >= 0 ||
        llama_slang_paged_commit(external, transaction) == 0 ||
        llama_slang_paged_page_release(external, malformed_page) != 0) {
        fail("malformed batch did not fail safely");
    }

    int64_t oversized_page = reserve_page(external);
    transaction            = begin_table(external, request_a, 1);
    push_page(external, transaction, oversized_page, 0, 1);
    llama_token   oversized_token  = one.front();
    llama_pos     oversized_pos    = 0;
    int32_t       oversized_n_seq  = 1;
    llama_seq_id  oversized_seq    = 0;
    llama_seq_id * oversized_seqs  = &oversized_seq;
    int8_t        oversized_logits = 1;
    llama_batch oversized{};
    oversized.n_tokens = int32_t(cparams.n_batch + 1);
    oversized.token    = &oversized_token;
    oversized.pos      = &oversized_pos;
    oversized.n_seq_id = &oversized_n_seq;
    oversized.seq_id   = &oversized_seqs;
    oversized.logits   = &oversized_logits;
    if (llama_slang_paged_decode(external, transaction, oversized) >= 0 ||
        llama_slang_paged_commit(external, transaction) == 0 ||
        llama_slang_paged_page_release(external, oversized_page) != 0) {
        fail("oversized batch did not fail safely");
    }

    int64_t duplicate_page = reserve_page(external);
    transaction            = begin_table(external, request_a, 1);
    push_page(external, transaction, duplicate_page, 0, 1);
    if (!decode_external(external, transaction, one, 0) ||
        decode_external(external, transaction, one, 0) ||
        llama_slang_paged_commit(external, transaction) == 0 ||
        llama_slang_paged_page_release(external, duplicate_page) != 0) {
        fail("duplicate decode did not poison transaction");
    }

    std::vector<llama_token> a_prefix(prompt.begin(), prompt.begin() + 5);
    if (!decode_tokens(legacy_a, a_prefix, 0)) {
        fail("external A reference prefix");
    }
    const int64_t shared_page = reserve_page(external);
    int64_t       a_tail      = reserve_page(external);
    transaction = begin_table(external, request_a, 2);
    push_page(external, transaction, shared_page, 0, 4);
    push_page(external, transaction, a_tail, 0, 1);
    if (llama_slang_paged_commit(external, transaction) == 0) {
        fail("premature external commit accepted");
    }
    transaction = begin_table(external, request_a, 2);
    push_page(external, transaction, shared_page, 0, 4);
    push_page(external, transaction, a_tail, 0, 1);
    std::vector<float> copied_logits(n_vocab);
    if (llama_slang_paged_logits_copy(external, request_a, copied_logits.data(), copied_logits.size()) == 0) {
        fail("uncommitted logits exposed before decode");
    }
    if (!decode_external(external, transaction, a_prefix, 0)) {
        fail("external A prefix decode");
    }
    if (llama_get_logits(external) != nullptr || llama_get_logits_ith(external, -1) != nullptr) {
        fail("ambient external logits exposed");
    }
    if (llama_slang_paged_logits_copy(external, request_a, copied_logits.data(), copied_logits.size()) == 0) {
        fail("computed logits exposed before commit");
    }
    if (llama_slang_paged_commit(external, transaction) != 0) {
        fail("external A prefix commit");
    }
    copy_committed_logits(external, request_a, copied_logits);
    compare_logits_copy(legacy_a, copied_logits, n_vocab, "A prefix");
    if (llama_get_logits(external) != nullptr || llama_get_logits_ith(external, -1) != nullptr) {
        fail("ambient committed external logits exposed");
    }

    std::vector<llama_token> b_full(prompt.begin(), prompt.begin() + 6);
    std::vector<llama_token> b_suffix(prompt.begin() + 4, prompt.begin() + 6);
    if (!decode_tokens(legacy_b, b_full, 0)) {
        fail("external B reference");
    }
    int64_t b_tail = reserve_page(external);
    transaction    = begin_table(external, request_b, 2);
    push_page(external, transaction, shared_page, 4, 0);
    push_page(external, transaction, b_tail, 0, 2);
    if (!decode_external(external, transaction, b_suffix, 4) ||
        llama_slang_paged_commit_and_copy(external, transaction, copied_logits.data(), copied_logits.size()) != 0) {
        fail("external B shared-prefix decode");
    }
    compare_logits_copy(legacy_b, copied_logits, n_vocab, "B shared prefix");

    std::vector<llama_token> a_next = { prompt[5] };
    if (!decode_tokens(legacy_a, a_next, 5)) {
        fail("external A reference extension");
    }
    const int64_t old_a_tail   = a_tail;
    int64_t       a_replacement = reserve_page(external);
    if (llama_slang_paged_page_copy_tail(external, a_tail, a_replacement, 1) != 0) {
        fail("external A tail copy");
    }
    const int64_t borrowed_partial = begin_table(external, request_b, 2);
    push_page(external, borrowed_partial, shared_page, 4, 0);
    push_page(external, borrowed_partial, old_a_tail, 1, 0);

    transaction = begin_table(external, request_a, 2);
    push_page(external, transaction, shared_page, 4, 0);
    push_page(external, transaction, a_replacement, 1, 1);
    if (!decode_external(external, transaction, a_next, 5) ||
        llama_slang_paged_commit(external, transaction) != 0) {
        fail("external A interleaved extension");
    }
    a_tail = a_replacement;
    if (llama_slang_paged_page_release(external, old_a_tail) == 0 ||
        llama_slang_paged_abort(external, borrowed_partial) != 0 ||
        llama_slang_paged_page_release(external, old_a_tail) != 0) {
        fail("external shared partial-tail ownership");
    }
    copy_committed_logits(external, request_b, copied_logits);
    compare_logits_copy(legacy_b, copied_logits, n_vocab, "B retained during A tail replacement");
    copy_committed_logits(external, request_a, copied_logits);
    compare_logits_copy(legacy_a, copied_logits, n_vocab, "A interleaved");

    std::vector<llama_token> b_next = { prompt[6] };
    if (!decode_tokens(legacy_b, b_next, 6)) {
        fail("external B reference extension");
    }
    const int64_t old_b_tail   = b_tail;
    int64_t       b_replacement = reserve_page(external);
    if (llama_slang_paged_page_copy_tail(external, b_tail, b_replacement, 2) != 0) {
        fail("external B tail copy");
    }
    transaction = begin_table(external, request_b, 2);
    push_page(external, transaction, shared_page, 4, 0);
    push_page(external, transaction, b_replacement, 2, 1);
    if (!decode_external(external, transaction, b_next, 6) ||
        llama_slang_paged_commit(external, transaction) != 0) {
        fail("external B interleaved extension");
    }
    b_tail = b_replacement;
    if (llama_slang_paged_page_release(external, old_b_tail) != 0) {
        fail("external replaced B page retained ownership");
    }
    copy_committed_logits(external, request_b, copied_logits);
    compare_logits_copy(legacy_b, copied_logits, n_vocab, "B interleaved");

    std::vector<llama_token> a_retry = { prompt[6] };
    if (!decode_tokens(legacy_a, a_retry, 6)) {
        fail("external abort reference");
    }
    int64_t aborted_tail = reserve_page(external);
    if (llama_slang_paged_page_copy_tail(external, a_tail, aborted_tail, 2) != 0) {
        fail("external abort tail copy");
    }
    transaction = begin_table(external, request_a, 2);
    push_page(external, transaction, shared_page, 4, 0);
    push_page(external, transaction, aborted_tail, 2, 1);
    abort_state.enabled     = true;
    abort_state.abort_after = 2;
    if (decode_external(external, transaction, a_retry, 6)) {
        fail("external graph abort accepted");
    }
    abort_state.enabled = false;
    if (llama_slang_paged_abort(external, transaction) != 0) {
        fail("external explicit abort");
    }
    if (llama_get_logits(external) != nullptr || llama_get_logits_ith(external, -1) != nullptr ||
        llama_slang_paged_page_release(external, aborted_tail) != 0) {
        fail("external abort visibility or ownership");
    }

    int64_t retry_tail = reserve_page(external);
    if (llama_slang_paged_page_copy_tail(external, a_tail, retry_tail, 2) != 0) {
        fail("external retry tail copy");
    }
    transaction = begin_table(external, request_a, 2);
    push_page(external, transaction, shared_page, 4, 0);
    push_page(external, transaction, retry_tail, 2, 1);
    if (!decode_external(external, transaction, a_retry, 6) ||
        llama_slang_paged_commit(external, transaction) != 0) {
        fail("external retry decode");
    }
    if (llama_slang_paged_page_release(external, a_tail) != 0) {
        fail("external replaced retry source retained ownership");
    }
    a_tail = retry_tail;
    copy_committed_logits(external, request_a, copied_logits);
    compare_logits_copy(legacy_a, copied_logits, n_vocab, "A abort retry");

    const int64_t sealed_copy = reserve_page(external);
    if (llama_slang_paged_page_copy_tail(external, shared_page, sealed_copy, 4) != 0 ||
        llama_slang_paged_page_seal(external, sealed_copy) != 0 ||
        llama_slang_paged_page_seal(external, sealed_copy) == 0) {
        fail("external explicit copy and seal");
    }
    const int64_t sealed_transaction = begin_table(external, request_b, 1);
    push_page(external, sealed_transaction, sealed_copy, 4, 0);
    if (llama_slang_paged_abort(external, sealed_transaction) != 0 ||
        llama_slang_paged_page_release(external, sealed_copy) != 0) {
        fail("external attach of explicitly sealed copy");
    }

    if (llama_slang_paged_request_close(external, request_a) != 0 ||
        llama_slang_paged_request_close(external, request_b) != 0 ||
        llama_slang_paged_request_close(external, request_a) == 0) {
        fail("external request close");
    }
    // Request close removes table mappings; the external owner deliberately
    // retains sealed pages until it releases their opaque handles.
    if (llama_slang_paged_page_release(external, shared_page) != 0 ||
        llama_slang_paged_page_release(external, a_tail) != 0 ||
        llama_slang_paged_page_release(external, b_tail) != 0) {
        fail("external retained page release");
    }

    const int64_t close_request = llama_slang_paged_request_open(external);
    const int64_t close_page    = reserve_page(external);
    transaction                 = begin_table(external, close_request, 1);
    push_page(external, transaction, close_page, 0, 1);
    if (!decode_external(external, transaction, one, 0) ||
        llama_slang_paged_request_close(external, close_request) != 0 ||
        llama_slang_paged_commit(external, transaction) == 0 ||
        llama_slang_paged_page_release(external, close_page) != 0) {
        fail("computed request close");
    }

    const int64_t bounded_request = llama_slang_paged_request_open(external);
    const int64_t bounded_page    = reserve_page(external);
    transaction = llama_slang_paged_table_begin(external, bounded_request, int64_t(1) << 30, 1);
    push_page(external, transaction, bounded_page, 0, 1);
    if (decode_external(external, transaction, one, llama_pos(1) << 30) ||
        llama_slang_paged_abort(external, transaction) != 0 ||
        llama_slang_paged_request_close(external, bounded_request) != 0 ||
        llama_slang_paged_page_release(external, bounded_page) != 0) {
        fail("external absolute-position bound");
    }

    const int64_t atomic_request = llama_slang_paged_request_open(external);
    std::vector<float> sentinel(size_t(n_vocab), 123.25f);
    auto require_unchanged = [&]() {
        if (!std::all_of(sentinel.begin(), sentinel.end(), [](float value) { return value == 123.25f; })) {
            fail("failed atomic commit changed destination logits");
        }
    };
    int64_t atomic_page = reserve_page(external);
    transaction = begin_table(external, atomic_request, 1);
    push_page(external, transaction, atomic_page, 0, 1);
    if (llama_slang_paged_commit_and_copy(external, transaction, sentinel.data(), sentinel.size()) == 0 ||
        llama_slang_paged_page_release(external, atomic_page) != 0) {
        fail("premature atomic commit consumption");
    }
    require_unchanged();

    atomic_page = reserve_page(external);
    transaction = begin_table(external, atomic_request, 1);
    push_page(external, transaction, atomic_page, 0, 1);
    if (!decode_external(external, transaction, one, 0) ||
        llama_slang_paged_commit_and_copy(external, transaction, sentinel.data(), sentinel.size() - 1) == 0 ||
        llama_slang_paged_page_release(external, atomic_page) != 0) {
        fail("wrong-count atomic commit consumption");
    }
    require_unchanged();

    atomic_page = reserve_page(external);
    transaction = begin_table(external, atomic_request, 1);
    push_page(external, transaction, atomic_page, 0, 1);
    if (!decode_external(external, transaction, one, 0) ||
        llama_slang_paged_fail(external, transaction) != 0 ||
        llama_slang_paged_commit_and_copy(external, transaction, sentinel.data(), sentinel.size()) == 0 ||
        llama_slang_paged_page_release(external, atomic_page) != 0) {
        fail("poisoned atomic commit consumption");
    }
    require_unchanged();

    atomic_page = reserve_page(external);
    transaction = begin_table(external, atomic_request, 2);
    push_page(external, transaction, atomic_page, 0, 1);
    if (!decode_external(external, transaction, one, 0) ||
        llama_slang_paged_commit_and_copy(external, transaction, sentinel.data(), sentinel.size()) == 0 ||
        llama_slang_paged_page_release(external, atomic_page) != 0 ||
        llama_slang_paged_request_close(external, atomic_request) != 0) {
        fail("incomplete atomic commit consumption");
    }
    require_unchanged();

    std::vector<int64_t> reserved;
    for (;;) {
        const int64_t page = llama_slang_paged_page_reserve(external);
        if (page < 0) {
            break;
        }
        reserved.push_back(page);
    }
    if (reserved.size() != provider.page_capacity) {
        std::fprintf(stderr, "external page reclamation: reserved %zu of %u\n", reserved.size(),
                     provider.page_capacity);
        fail("external page reclamation");
    }
    for (int64_t page : reserved) {
        if (llama_slang_paged_page_release(external, page) != 0) {
            fail("external page release");
        }
    }

    llama_free(legacy_b);
    llama_free(legacy_a);
    llama_free(external);
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

    run_external_provider_case(model, cparams, prompt, n_vocab);

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
