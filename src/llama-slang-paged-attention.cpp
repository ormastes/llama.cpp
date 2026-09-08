#include "llama-slang-paged-attention.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool checked_mul(size_t left, size_t right, size_t & result) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

}  // namespace

llama_slang_paged_attention::llama_slang_paged_attention(const config & config) :
    config_(config),
    valid_(config.n_layer != 0 && config.n_head_q != 0 && config.n_head_kv != 0 && config.head_dim != 0 &&
           config.n_head_q % config.n_head_kv == 0 && std::isfinite(config.scale) && config.scale > 0.0f) {}

bool llama_slang_paged_attention::valid() const {
    return valid_;
}

bool llama_slang_paged_attention::execute(llama_slang_paged_requests &                   requests,
                                          llama_slang_paged_requests::transaction_handle transaction,
                                          uint32_t                                       layer,
                                          const int64_t *                                positions,
                                          uint32_t                                       n_tokens,
                                          input                                          q,
                                          input                                          k,
                                          input                                          v,
                                          output                                         result) const {
    auto reject = [&]() {
        requests.fail(transaction);
        return false;
    };
    size_t q_elements  = 0;
    size_t kv_elements = 0;
    size_t q_bytes     = 0;
    size_t kv_bytes    = 0;
    if (!valid_ || layer >= config_.n_layer || positions == nullptr || n_tokens == 0 || q.data == nullptr ||
        k.data == nullptr || v.data == nullptr || result.data == nullptr ||
        !checked_mul(n_tokens, config_.n_head_q, q_elements) ||
        !checked_mul(q_elements, config_.head_dim, q_elements) ||
        !checked_mul(n_tokens, config_.n_head_kv, kv_elements) ||
        !checked_mul(kv_elements, config_.head_dim, kv_elements) || !checked_mul(q_elements, sizeof(float), q_bytes) ||
        !checked_mul(kv_elements, sizeof(float), kv_bytes) || q.elements != q_elements ||
        result.elements != q_elements || k.elements != kv_elements || v.elements != kv_elements ||
        requests.execution_count(transaction) == 0 ||
        !requests.execution_compatible(transaction, config_.n_layer, config_.n_head_kv, config_.head_dim)) {
        return reject();
    }
    (void) q_bytes;
    (void) kv_bytes;
    for (uint32_t token = 0; token < n_tokens; ++token) {
        if (positions[token] < 0 || positions[token] > std::numeric_limits<int32_t>::max() ||
            (token != 0 && positions[token] != positions[token - 1] + 1)) {
            return reject();
        }
    }

    const uint32_t n_entries      = requests.execution_count(transaction);
    uint32_t       expected_token = 0;
    for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
        llama_slang_paged_requests::execution_entry entry{};
        if (!requests.execution_entry_at(transaction, entry_index, entry)) {
            return reject();
        }
        if (entry.readable_rows != 0 && entry.position_base + entry.readable_rows - 1 >= uint64_t(positions[0])) {
            return reject();
        }
        if (entry.writable_end > entry.readable_rows) {
            if (!entry.exclusive || !requests.layer_unwritten(transaction, entry_index, layer)) {
                return reject();
            }
            for (uint32_t row = entry.readable_rows; row < entry.writable_end; ++row) {
                if (expected_token >= n_tokens || entry.position_base + row != uint64_t(positions[expected_token])) {
                    return reject();
                }
                ++expected_token;
            }
        }
    }
    if (expected_token != n_tokens) {
        return reject();
    }

    for (uint32_t token = 0; token < n_tokens; ++token) {
        bool admitted = false;
        for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
            llama_slang_paged_requests::execution_entry entry{};
            if (!requests.execution_entry_at(transaction, entry_index, entry)) {
                return reject();
            }
            const uint64_t position = uint64_t(positions[token]);
            if (position < entry.position_base || position >= entry.position_base + entry.writable_end) {
                continue;
            }
            const uint32_t row = uint32_t(position - entry.position_base);
            if (!entry.exclusive || row < entry.readable_rows) {
                return reject();
            }
            for (uint32_t head = 0; head < config_.n_head_kv; ++head) {
                float * page_k = requests.writable_row(transaction, entry_index, false, layer, row);
                float * page_v = requests.writable_row(transaction, entry_index, true, layer, row);
                if (page_k == nullptr || page_v == nullptr) {
                    return reject();
                }
                const size_t source = (size_t(token) * config_.n_head_kv + head) * config_.head_dim;
                std::copy_n(k.data + source, config_.head_dim, page_k + size_t(head) * config_.head_dim);
                std::copy_n(v.data + source, config_.head_dim, page_v + size_t(head) * config_.head_dim);
            }
            admitted = true;
            break;
        }
        if (!admitted) {
            return reject();
        }
    }
    for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
        llama_slang_paged_requests::execution_entry entry{};
        requests.execution_entry_at(transaction, entry_index, entry);
        if (entry.writable_end > entry.readable_rows && !requests.mark_layer_written(transaction, entry_index, layer)) {
            return reject();
        }
    }

    const uint32_t group_size = config_.n_head_q / config_.n_head_kv;
    for (uint32_t token = 0; token < n_tokens; ++token) {
        for (uint32_t query_head = 0; query_head < config_.n_head_q; ++query_head) {
            const uint32_t kv_head = query_head / group_size;
            const float *  query   = q.data + (size_t(token) * config_.n_head_q + query_head) * config_.head_dim;

            auto score = [&](const float * key) {
                double sum = 0.0;
                for (uint32_t dim = 0; dim < config_.head_dim; ++dim) {
                    sum += double(query[dim]) * key[size_t(kv_head) * config_.head_dim + dim];
                }
                return sum * config_.scale;
            };

            double maximum = -std::numeric_limits<double>::infinity();
            for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
                llama_slang_paged_requests::execution_entry entry{};
                if (!requests.execution_entry_at(transaction, entry_index, entry)) {
                    return reject();
                }
                for (uint32_t row = 0; row < entry.readable_rows; ++row) {
                    if (entry.position_base + row > uint64_t(positions[token])) {
                        continue;
                    }
                    const float * key = requests.readable_row(transaction, entry_index, false, layer, row);
                    if (key == nullptr) {
                        return reject();
                    }
                    maximum = std::max(maximum, score(key));
                }
            }
            for (uint32_t key_token = 0; key_token <= token; ++key_token) {
                const float * key = k.data + (size_t(key_token) * config_.n_head_kv) * config_.head_dim;
                maximum           = std::max(maximum, score(key));
            }

            double denominator = 0.0;
            for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
                llama_slang_paged_requests::execution_entry entry{};
                requests.execution_entry_at(transaction, entry_index, entry);
                for (uint32_t row = 0; row < entry.readable_rows; ++row) {
                    if (entry.position_base + row <= uint64_t(positions[token])) {
                        denominator += std::exp(
                            score(requests.readable_row(transaction, entry_index, false, layer, row)) - maximum);
                    }
                }
            }
            for (uint32_t key_token = 0; key_token <= token; ++key_token) {
                const float * key = k.data + (size_t(key_token) * config_.n_head_kv) * config_.head_dim;
                denominator += std::exp(score(key) - maximum);
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                return reject();
            }

            float * destination = result.data + (size_t(token) * config_.n_head_q + query_head) * config_.head_dim;
            std::fill_n(destination, config_.head_dim, 0.0f);
            for (uint32_t entry_index = 0; entry_index < n_entries; ++entry_index) {
                llama_slang_paged_requests::execution_entry entry{};
                requests.execution_entry_at(transaction, entry_index, entry);
                for (uint32_t row = 0; row < entry.readable_rows; ++row) {
                    if (entry.position_base + row > uint64_t(positions[token])) {
                        continue;
                    }
                    const float * key    = requests.readable_row(transaction, entry_index, false, layer, row);
                    const float * value  = requests.readable_row(transaction, entry_index, true, layer, row);
                    const double  weight = std::exp(score(key) - maximum) / denominator;
                    for (uint32_t dim = 0; dim < config_.head_dim; ++dim) {
                        destination[dim] += float(weight * value[size_t(kv_head) * config_.head_dim + dim]);
                    }
                }
            }
            for (uint32_t key_token = 0; key_token <= token; ++key_token) {
                const float * key    = k.data + (size_t(key_token) * config_.n_head_kv) * config_.head_dim;
                const float * value  = v.data + (size_t(key_token) * config_.n_head_kv + kv_head) * config_.head_dim;
                const double  weight = std::exp(score(key) - maximum) / denominator;
                for (uint32_t dim = 0; dim < config_.head_dim; ++dim) {
                    destination[dim] += float(weight * value[dim]);
                }
            }
        }
    }
    return true;
}
