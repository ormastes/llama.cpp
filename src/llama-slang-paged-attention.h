#pragma once

#include "llama-slang-paged-requests.h"

#include <cstddef>
#include <cstdint>

// Allocation-free CPU scaled dot-product attention over transaction-owned
// physical pages. Q/K/V and output use [token][head][dimension] F32 layout.
class llama_slang_paged_attention {
  public:
    struct config {
        uint32_t n_layer;
        uint32_t n_head_q;
        uint32_t n_head_kv;
        uint32_t head_dim;
        float    scale;
    };

    struct input {
        const float * data;
        size_t        elements;
    };

    struct output {
        float * data;
        size_t  elements;
    };

    explicit llama_slang_paged_attention(const config & config);

    bool valid() const;
    bool execute(llama_slang_paged_requests &                   requests,
                 llama_slang_paged_requests::transaction_handle transaction,
                 uint32_t                                       layer,
                 const int64_t *                                positions,
                 uint32_t                                       n_tokens,
                 input                                          q,
                 input                                          k,
                 input                                          v,
                 output                                         result) const;

  private:
    config config_;
    bool   valid_;
};
