#include "llama-memory-slang-paged.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

int main() {
    llama_slang_paged_pool::layout layout = {
        2, 4, 2, 3, 2, 7, 2 * 2 * 4 * 2 * 3 * sizeof(float) * 2, 2 * 1024,
    };
    auto pool = llama_slang_paged_pool::create(layout);
    REQUIRE(pool != nullptr);
    REQUIRE(pool->page_bytes() == 2 * 4 * 2 * 3 * 2 * sizeof(float));

    const auto first  = pool->reserve();
    const auto second = pool->reserve();
    REQUIRE(first != 0 && second != 0 && first != second);
    REQUIRE(pool->reserve() == 0);
    REQUIRE(pool->allocated_bytes() == layout.byte_limit);

    for (uint32_t layer = 0; layer < layout.n_layer; ++layer) {
        for (uint32_t row = 0; row < 3; ++row) {
            float * k = pool->writable_row(first, false, layer, row);
            float * v = pool->writable_row(first, true, layer, row);
            REQUIRE(k != nullptr && v != nullptr);
            for (uint32_t i = 0; i < layout.n_head_kv * layout.head_dim; ++i) {
                k[i] = float(1000 * layer + 100 * row + i);
                v[i] = -k[i];
            }
        }
    }
    REQUIRE(pool->set_occupied(first, 3));
    REQUIRE(pool->writable_row(first, false, 0, 2) == nullptr);
    REQUIRE(pool->seal(first));
    REQUIRE(pool->copy_rows(first, second, 3));

    for (uint32_t layer = 0; layer < layout.n_layer; ++layer) {
        for (uint32_t row = 0; row < 3; ++row) {
            const float * source_k = pool->row(first, false, layer, row);
            const float * copied_k = pool->row(second, false, layer, row);
            const float * copied_v = pool->row(second, true, layer, row);
            REQUIRE(source_k != nullptr && copied_k != nullptr && copied_v != nullptr);
            for (uint32_t i = 0; i < layout.n_head_kv * layout.head_dim; ++i) {
                REQUIRE(copied_k[i] == source_k[i]);
                REQUIRE(copied_v[i] == -source_k[i]);
            }
        }
    }

    REQUIRE(pool->release(first));
    REQUIRE(!pool->release(first));
    const auto replacement = pool->reserve();
    REQUIRE(replacement != 0 && replacement != first);
    REQUIRE(pool->row(first, false, 0, 0) == nullptr);
    REQUIRE(pool->release(second));
    REQUIRE(pool->release(replacement));
    REQUIRE(pool->allocated_bytes() == 0);

    auto other_pool = llama_slang_paged_pool::create(layout);
    REQUIRE(other_pool != nullptr);
    const auto owned = pool->reserve();
    const auto other = other_pool->reserve();
    REQUIRE(owned != 0 && other != 0);
    REQUIRE(!pool->release(other));
    REQUIRE(other_pool->writable_row(other, false, 0, 0) != nullptr);
    REQUIRE(pool->release(owned));
    other_pool.reset();
    auto recreated_pool = llama_slang_paged_pool::create(layout);
    REQUIRE(recreated_pool != nullptr);
    const auto recreated = recreated_pool->reserve();
    REQUIRE(recreated != 0 && recreated != other);
    REQUIRE(recreated_pool->release(recreated));

    auto ownership_pool = llama_slang_paged_pool::create(layout);
    REQUIRE(ownership_pool != nullptr);
    const auto mapped = ownership_pool->reserve();
    REQUIRE(mapped != 0);
    REQUIRE(ownership_pool->map_for_request(mapped, 41));
    REQUIRE(ownership_pool->mapping_references(mapped) == 1);
    REQUIRE(!ownership_pool->release(mapped));
    REQUIRE(!ownership_pool->claim_exclusive(mapped, 99, 42));
    REQUIRE(!ownership_pool->claim_exclusive(mapped, 99, 41));
    REQUIRE(ownership_pool->unmap_for_request(mapped, 41));
    REQUIRE(ownership_pool->claim_exclusive(mapped, 99, 41));
    REQUIRE(ownership_pool->exclusive_transaction(mapped) == 99);
    REQUIRE(ownership_pool->writable_row(mapped, false, 0, 0) == nullptr);
    REQUIRE(ownership_pool->writable_row_exclusive(mapped, 99, false, 0, 0) != nullptr);
    REQUIRE(!ownership_pool->abort_exclusive(mapped, 98));
    REQUIRE(ownership_pool->abort_exclusive(mapped, 99));
    REQUIRE(ownership_pool->mapping_references(mapped) == 0);
    REQUIRE(ownership_pool->claim_exclusive(mapped, 100, 41));
    REQUIRE(ownership_pool->set_occupied_exclusive(mapped, 100, 1));
    REQUIRE(!ownership_pool->publish_exclusive(mapped, 100, 42));
    REQUIRE(ownership_pool->publish_exclusive(mapped, 100, 41));
    REQUIRE(ownership_pool->unmap_for_request(mapped, 41));

    REQUIRE(ownership_pool->claim_exclusive(mapped, 102, 41));
    float * aborted_k = ownership_pool->writable_row_exclusive(mapped, 102, false, 0, 1);
    float * aborted_v = ownership_pool->writable_row_exclusive(mapped, 102, true, 0, 1);
    REQUIRE(aborted_k != nullptr && aborted_v != nullptr);
    aborted_k[0] = 17.0f;
    aborted_v[0] = -17.0f;
    REQUIRE(ownership_pool->set_occupied_exclusive(mapped, 102, 2));
    REQUIRE(ownership_pool->seal_exclusive(mapped, 102));
    REQUIRE(ownership_pool->abort_exclusive(mapped, 102));
    REQUIRE(ownership_pool->state(mapped) == llama_slang_paged_pool::page_state::writable);
    REQUIRE(ownership_pool->occupied(mapped) == 1);
    aborted_k = ownership_pool->writable_row(mapped, false, 0, 1);
    aborted_v = ownership_pool->writable_row(mapped, true, 0, 1);
    REQUIRE(aborted_k != nullptr && aborted_v != nullptr);
    REQUIRE(aborted_k[0] == 0.0f && aborted_v[0] == 0.0f);
    REQUIRE(ownership_pool->seal(mapped));
    REQUIRE(ownership_pool->map_for_request(mapped, 41));
    REQUIRE(ownership_pool->map_for_request(mapped, 42));
    REQUIRE(ownership_pool->borrow_sealed(mapped));
    REQUIRE(ownership_pool->borrow_sealed(mapped));
    REQUIRE(ownership_pool->mapping_references(mapped) == 2);
    REQUIRE(ownership_pool->read_borrows(mapped) == 2);
    REQUIRE(!ownership_pool->release(mapped));
    REQUIRE(ownership_pool->unmap_for_request(mapped, 41));
    REQUIRE(ownership_pool->unmap_for_request(mapped, 42));
    REQUIRE(ownership_pool->return_sealed(mapped));
    REQUIRE(ownership_pool->return_sealed(mapped));
    REQUIRE(ownership_pool->release(mapped));

    const auto sealed_by_transaction = ownership_pool->reserve();
    REQUIRE(sealed_by_transaction != 0);
    REQUIRE(ownership_pool->claim_exclusive(sealed_by_transaction, 101, 43));
    REQUIRE(ownership_pool->writable_row_exclusive(sealed_by_transaction, 101, true, 0, 0) != nullptr);
    REQUIRE(ownership_pool->set_occupied_exclusive(sealed_by_transaction, 101, 1));
    REQUIRE(ownership_pool->seal_exclusive(sealed_by_transaction, 101));
    const auto copy_destination = ownership_pool->reserve();
    REQUIRE(copy_destination != 0);
    REQUIRE(!ownership_pool->copy_rows(sealed_by_transaction, copy_destination, 1));
    REQUIRE(ownership_pool->publish_exclusive(sealed_by_transaction, 101, 43));
    REQUIRE(ownership_pool->copy_rows(sealed_by_transaction, copy_destination, 1));
    REQUIRE(ownership_pool->state(sealed_by_transaction) == llama_slang_paged_pool::page_state::sealed);
    REQUIRE(ownership_pool->unmap_for_request(sealed_by_transaction, 43));
    REQUIRE(ownership_pool->release(sealed_by_transaction));
    REQUIRE(ownership_pool->release(copy_destination));

    auto invalid    = layout;
    invalid.n_layer = 0;
    REQUIRE(llama_slang_paged_pool::create(invalid) == nullptr);
    invalid             = layout;
    invalid.n_layer     = UINT32_MAX;
    invalid.page_tokens = UINT32_MAX;
    REQUIRE(llama_slang_paged_pool::create(invalid) == nullptr);
    invalid                       = layout;
    invalid.descriptor_byte_limit = 1;
    REQUIRE(llama_slang_paged_pool::create(invalid) == nullptr);

    llama_slang_paged_pool::test_set_next_identity(UINT64_MAX - 1);
    auto exhausted_pool = llama_slang_paged_pool::create(layout);
    REQUIRE(exhausted_pool != nullptr);
    const auto final_identity = exhausted_pool->reserve();
    REQUIRE(final_identity == UINT64_MAX - 1);
    REQUIRE(exhausted_pool->reserve() == 0);
    REQUIRE(exhausted_pool->release(final_identity));
    REQUIRE(exhausted_pool->reserve() == 0);
    return 0;
}
