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

LLAMA_API struct llama_context * llama_init_from_model_slang_paged(struct llama_model *            model,
                                                                   struct llama_context_params     params,
                                                                   struct llama_slang_paged_params paged);

#ifdef __cplusplus
}
#endif
