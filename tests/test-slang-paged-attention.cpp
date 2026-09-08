#include "llama-slang-paged-attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

static void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

static std::vector<float> dense_oracle(const float * prefix_k,
                                       const float * prefix_v,
                                       uint32_t      prefix_tokens,
                                       const float * q,
                                       const float * current_k,
                                       const float * current_v,
                                       uint32_t      current_tokens,
                                       uint32_t      n_head_q,
                                       uint32_t      n_head_kv,
                                       uint32_t      head_dim,
                                       float         scale) {
    std::vector<float> result(size_t(current_tokens) * n_head_q * head_dim, 0.0f);
    const uint32_t     group = n_head_q / n_head_kv;
    for (uint32_t token = 0; token < current_tokens; ++token) {
        for (uint32_t q_head = 0; q_head < n_head_q; ++q_head) {
            const uint32_t kv_head = q_head / group;
            const float *  query   = q + (size_t(token) * n_head_q + q_head) * head_dim;
            auto           key_at  = [&](uint32_t key_token) {
                const float *  source = key_token < prefix_tokens ? prefix_k : current_k;
                const uint32_t index  = key_token < prefix_tokens ? key_token : key_token - prefix_tokens;
                return source + (size_t(index) * n_head_kv + kv_head) * head_dim;
            };
            auto value_at = [&](uint32_t key_token) {
                const float *  source = key_token < prefix_tokens ? prefix_v : current_v;
                const uint32_t index  = key_token < prefix_tokens ? key_token : key_token - prefix_tokens;
                return source + (size_t(index) * n_head_kv + kv_head) * head_dim;
            };
            auto score = [&](const float * key) {
                double value = 0.0;
                for (uint32_t dim = 0; dim < head_dim; ++dim) {
                    value += double(query[dim]) * key[dim];
                }
                return value * scale;
            };
            const uint32_t key_count = prefix_tokens + token + 1;
            double         maximum   = -INFINITY;
            for (uint32_t key_token = 0; key_token < key_count; ++key_token) {
                maximum = std::max(maximum, score(key_at(key_token)));
            }
            double denominator = 0.0;
            for (uint32_t key_token = 0; key_token < key_count; ++key_token) {
                denominator += std::exp(score(key_at(key_token)) - maximum);
            }
            float * output = result.data() + (size_t(token) * n_head_q + q_head) * head_dim;
            for (uint32_t key_token = 0; key_token < key_count; ++key_token) {
                const double  weight = std::exp(score(key_at(key_token)) - maximum) / denominator;
                const float * value  = value_at(key_token);
                for (uint32_t dim = 0; dim < head_dim; ++dim) {
                    output[dim] += float(weight * value[dim]);
                }
            }
        }
    }
    return result;
}

static void require_close(const float * actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::fabs(actual[i] - expected[i]) < 1e-5f);
    }
}

int main() {
    llama_slang_paged_pool::layout pool_layout = {
        2, 2, 1, 2, 4, 77, 4 * 2 * 2 * 1 * 2 * sizeof(float) * 2, 4 * 1024,
    };
    auto unique_pool = llama_slang_paged_pool::create(pool_layout);
    REQUIRE(unique_pool != nullptr);
    std::shared_ptr<llama_slang_paged_pool> pool(std::move(unique_pool));
    llama_slang_paged_requests::limits      limits   = { 2, 3, 3, 32 * 1024, 2 * 3 * sizeof(float) * 2 };
    auto                                    requests = llama_slang_paged_requests::create(pool, limits);
    REQUIRE(requests != nullptr);
    llama_slang_paged_attention attention({ 2, 2, 1, 2, 0.5f });
    REQUIRE(attention.valid());

    const auto first_request = requests->open_request(77);
    const auto first_page    = pool->reserve();
    const auto second_page   = pool->reserve();
    REQUIRE(first_request != 0 && first_page != 0 && second_page != 0);
    const auto cold = requests->begin(first_request, 5, 2);
    REQUIRE(cold != 0);
    REQUIRE(requests->push(cold, first_page, 0, 2));
    REQUIRE(requests->push(cold, second_page, 0, 1));

    const int64_t cold_positions[] = { 5, 6, 7 };
    const float   q0[]             = { 1, 0, 0, 1, 1, 1, -1, 1, 2, 1, 1, -2 };
    const float   k0[]             = { 1, 0, 0, 1, 1, 1 };
    const float   v0[]             = { 1, 2, 3, 4, 5, 6 };
    const float   q1[]             = { 0.5f, 1, 1, 0.5f, -1, 2, 2, -1, 1, 3, 3, 1 };
    const float   k1[]             = { 2, 0, 0, 2, -1, 1 };
    const float   v1[]             = { 6, 5, 4, 3, 2, 1 };
    float         cold_out0[12]{};
    float         cold_out1[12]{};
    REQUIRE(
        attention.execute(*requests, cold, 0, cold_positions, 3, { q0, 12 }, { k0, 6 }, { v0, 6 }, { cold_out0, 12 }));
    REQUIRE(
        attention.execute(*requests, cold, 1, cold_positions, 3, { q1, 12 }, { k1, 6 }, { v1, 6 }, { cold_out1, 12 }));
    require_close(cold_out0, dense_oracle(nullptr, nullptr, 0, q0, k0, v0, 3, 2, 1, 2, 0.5f));
    require_close(cold_out1, dense_oracle(nullptr, nullptr, 0, q1, k1, v1, 3, 2, 1, 2, 0.5f));
    REQUIRE(requests->finish_rows(cold, 0, 2));
    REQUIRE(requests->finish_rows(cold, 1, 1));
    const float logits[] = { 0.1f, 0.2f, 0.3f };
    REQUIRE(requests->set_logits(cold, 7, logits, 3));
    REQUIRE(requests->commit(cold));

    const auto second_request = requests->open_request(77);
    const auto cow_page       = pool->reserve();
    REQUIRE(second_request != 0 && cow_page != 0);
    REQUIRE(pool->copy_rows(second_page, cow_page, 1));
    const auto cow = requests->begin(second_request, 5, 2);
    REQUIRE(cow != 0);
    REQUIRE(requests->push(cow, first_page, 2, 0));
    REQUIRE(requests->push(cow, cow_page, 1, 1));

    const int64_t cow_position[] = { 8 };
    const float   q2[]           = { 1, -1, 2, 1 };
    const float   k2[]           = { -2, 1 };
    const float   v2[]           = { 7, 8 };
    float         cow_out0[4]{};
    REQUIRE(attention.execute(*requests, cow, 0, cow_position, 1, { q2, 4 }, { k2, 2 }, { v2, 2 }, { cow_out0, 4 }));
    require_close(cow_out0, dense_oracle(k0, v0, 3, q2, k2, v2, 1, 2, 1, 2, 0.5f));

    const float q3[] = { -1, 0.5f, 0.25f, 2 };
    const float k3[] = { 1, -2 };
    const float v3[] = { 8, 7 };
    float       cow_out1[4]{};
    REQUIRE(attention.execute(*requests, cow, 1, cow_position, 1, { q3, 4 }, { k3, 2 }, { v3, 2 }, { cow_out1, 4 }));
    require_close(cow_out1, dense_oracle(k1, v1, 3, q3, k3, v3, 1, 2, 1, 2, 0.5f));
    REQUIRE(requests->finish_rows(cow, 1, 2));
    REQUIRE(requests->set_logits(cow, 8, logits, 3));
    REQUIRE(requests->commit(cow));

    REQUIRE(pool->occupied(second_page) == 1);
    REQUIRE(pool->row(second_page, false, 0, 0)[0] == k0[4]);
    REQUIRE(pool->row(cow_page, false, 0, 1)[0] == k2[0]);
    REQUIRE(pool->mapping_references(first_page) == 2);
    REQUIRE(pool->mapping_references(second_page) == 1);
    REQUIRE(pool->mapping_references(cow_page) == 1);
    REQUIRE(!pool->release(first_page));

    REQUIRE(requests->close_request(second_request));
    REQUIRE(requests->close_request(first_request));
    REQUIRE(pool->release(first_page));
    REQUIRE(pool->release(second_page));
    REQUIRE(pool->release(cow_page));

    const float failure_q[] = { 1, 0, 0, 1, 1, 1, -1, 1 };
    const float failure_k[] = { 1, 0, 0, 1 };
    const float failure_v[] = { 2, 1, 4, 3 };
    float       failure_output[8]{};

    const auto gap_request = requests->open_request(77);
    const auto gap_page    = pool->reserve();
    const auto gap         = requests->begin(gap_request, 100, 1);
    REQUIRE(gap_request != 0 && gap_page != 0 && gap != 0);
    REQUIRE(requests->push(gap, gap_page, 0, 2));
    const int64_t gapped_positions[] = { 101, 102 };
    REQUIRE(!attention.execute(*requests, gap, 0, gapped_positions, 2, { failure_q, 8 }, { failure_k, 4 },
                               { failure_v, 4 }, { failure_output, 8 }));
    REQUIRE(requests->abort(gap));
    REQUIRE(pool->occupied(gap_page) == 0);
    REQUIRE(pool->release(gap_page));
    REQUIRE(requests->close_request(gap_request));

    const auto repeated_request = requests->open_request(77);
    const auto repeated_page    = pool->reserve();
    const auto repeated         = requests->begin(repeated_request, 100, 1);
    REQUIRE(repeated_request != 0 && repeated_page != 0 && repeated != 0);
    REQUIRE(requests->push(repeated, repeated_page, 0, 2));
    const int64_t complete_positions[] = { 100, 101 };
    REQUIRE(attention.execute(*requests, repeated, 0, complete_positions, 2, { failure_q, 8 }, { failure_k, 4 },
                              { failure_v, 4 }, { failure_output, 8 }));
    REQUIRE(!attention.execute(*requests, repeated, 0, complete_positions, 2, { failure_q, 8 }, { failure_k, 4 },
                               { failure_v, 4 }, { failure_output, 8 }));
    REQUIRE(requests->abort(repeated));
    REQUIRE(pool->occupied(repeated_page) == 0);
    REQUIRE(pool->release(repeated_page));
    REQUIRE(requests->close_request(repeated_request));

    const auto oversized_request = requests->open_request(77);
    const auto oversized_page    = pool->reserve();
    const auto oversized         = requests->begin(oversized_request, 100, 1);
    REQUIRE(oversized_request != 0 && oversized_page != 0 && oversized != 0);
    REQUIRE(requests->push(oversized, oversized_page, 0, 2));
    llama_slang_paged_attention oversized_attention({ 2, UINT32_MAX, 1, 2, 0.5f });
    REQUIRE(oversized_attention.valid());
    REQUIRE(!oversized_attention.execute(*requests, oversized, 0, complete_positions, UINT32_MAX, { failure_q, 0 },
                                         { failure_k, 0 }, { failure_v, 0 }, { failure_output, 0 }));
    REQUIRE(requests->abort(oversized));
    REQUIRE(pool->exclusive_transaction(oversized_page) == 0);
    REQUIRE(pool->release(oversized_page));
    REQUIRE(requests->close_request(oversized_request));
    return 0;
}
