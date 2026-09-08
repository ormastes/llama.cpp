#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental, CPU-only Slang physical-page context configuration. This API
// is separate from llama_context_params so the stable parameter ABI is not
// changed while paged execution remains capability-gated.
struct llama_slang_paged_params {
    uint32_t page_tokens;
    uint32_t page_capacity;
    size_t   byte_limit;
    size_t   descriptor_byte_limit;
};

#define LLAMA_SLANG_PAGED_PROVIDER_ABI_VERSION 1

// Versioned configuration for caller-owned request tables. Handles returned
// by this API are positive and remain valid only for the owning context.
// The caller owns every reserved page until page_release(), including pages
// retained after request_close(), transaction abort, or table replacement.
struct llama_slang_paged_provider_params {
    uint32_t abi_version;
    uint32_t page_tokens;
    uint32_t page_capacity;
    uint32_t max_requests;
    uint32_t max_pages_per_request;
    size_t   byte_limit;
    size_t   descriptor_byte_limit;
    size_t   metadata_byte_limit;
    size_t   logits_byte_limit;
};

LLAMA_API struct llama_context * llama_init_from_model_slang_paged(struct llama_model *            model,
                                                                   struct llama_context_params     params,
                                                                   struct llama_slang_paged_params paged);

// External-table mode keeps graph execution serial while allowing the caller
// to retain multiple immutable request tables. At most one transaction may
// have computed-but-uncommitted output per context. llama_decode() and the
// ambient llama_get_logits*() accessors are rejected; committed output is
// available only through llama_slang_paged_logits_copy().
//
// Status functions return 0 on success and a negative value on failure.
// A valid staging/decode failure makes its transaction abort-only. commit()
// consumes both successful and rejected transactions; abort() rolls page data
// back without releasing caller-owned handles. request_close() aborts an
// outstanding computed transaction for that request and removes table mappings.
LLAMA_API struct llama_context * llama_init_from_model_slang_paged_external(
    struct llama_model * model,
    struct llama_context_params params,
    struct llama_slang_paged_provider_params provider);

LLAMA_API int64_t llama_slang_paged_request_open(struct llama_context * ctx);
LLAMA_API int32_t llama_slang_paged_request_close(struct llama_context * ctx, int64_t request);

LLAMA_API int64_t llama_slang_paged_page_reserve(struct llama_context * ctx);
LLAMA_API int32_t llama_slang_paged_page_release(struct llama_context * ctx, int64_t page);
LLAMA_API int32_t llama_slang_paged_page_copy_tail(
    struct llama_context * ctx, int64_t source, int64_t destination, uint32_t rows);

LLAMA_API int64_t llama_slang_paged_table_begin(
    struct llama_context * ctx, int64_t request, int64_t base_position, uint32_t expected_pages);
LLAMA_API int32_t llama_slang_paged_table_push(
    struct llama_context * ctx, int64_t transaction, int64_t page, uint32_t valid_rows, uint32_t additional_rows);

LLAMA_API int32_t llama_slang_paged_decode(
    struct llama_context * ctx, int64_t transaction, struct llama_batch batch);
LLAMA_API int32_t llama_slang_paged_commit(struct llama_context * ctx, int64_t transaction);
LLAMA_API int32_t llama_slang_paged_abort(struct llama_context * ctx, int64_t transaction);

LLAMA_API int32_t llama_slang_paged_logits_copy(
    struct llama_context * ctx, int64_t request, float * destination, size_t count);

#ifdef __cplusplus
}
#endif
