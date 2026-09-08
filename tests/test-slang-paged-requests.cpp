#include "llama-slang-paged-requests.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

static void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

static void mark_all_layers(llama_slang_paged_requests &                   requests,
                            llama_slang_paged_requests::transaction_handle transaction,
                            uint32_t                                       entry,
                            uint32_t                                       layers) {
    for (uint32_t layer = 0; layer < layers; ++layer) {
        REQUIRE(requests.mark_layer_written(transaction, entry, layer));
    }
}

int main() {
    llama_slang_paged_pool::layout pool_layout = {
        2, 2, 1, 2, 8, 7, 8 * 2 * 2 * 1 * 2 * sizeof(float) * 2, 8 * 1024,
    };
    auto unique_pool = llama_slang_paged_pool::create(pool_layout);
    REQUIRE(unique_pool != nullptr);
    std::shared_ptr<llama_slang_paged_pool> pool(std::move(unique_pool));

    llama_slang_paged_requests::limits request_limits = {
        3, 4, 3, 64 * 1024, 3 * 3 * sizeof(float) * 2,
    };
    auto requests = llama_slang_paged_requests::create(pool, request_limits);
    REQUIRE(requests != nullptr);

    const auto first_request  = requests->open_request(7);
    const auto second_request = requests->open_request(7);
    const auto third_request  = requests->open_request(7);
    REQUIRE(first_request != 0 && second_request != 0 && third_request != 0);
    REQUIRE(first_request != second_request && second_request != third_request);
    REQUIRE(requests->open_request(7) == 0);
    REQUIRE(requests->open_request(8) == 0);

    const auto closed_page = pool->reserve();
    REQUIRE(closed_page != 0);
    const auto closed_transaction = requests->begin(third_request, 20, 1);
    REQUIRE(closed_transaction != 0);
    REQUIRE(requests->push(closed_transaction, closed_page, 0, 1));
    REQUIRE(requests->writable_row(closed_transaction, 0, false, 0, 0) != nullptr);
    mark_all_layers(*requests, closed_transaction, 0, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(closed_transaction, 0, 1));
    REQUIRE(requests->close_request(third_request));
    REQUIRE(!requests->abort(closed_transaction));
    REQUIRE(pool->occupied(closed_page) == 0);
    REQUIRE(pool->release(closed_page));

    const auto first_page  = pool->reserve();
    const auto second_page = pool->reserve();
    REQUIRE(first_page != 0 && second_page != 0);
    const auto cold = requests->begin(first_request, 10, 2);
    REQUIRE(cold != 0);
    REQUIRE(requests->begin(first_request, 10, 2) == 0);
    REQUIRE(requests->push(cold, first_page, 0, 2));
    REQUIRE(requests->writable_row(cold, 0, false, 0, 0) != nullptr);
    REQUIRE(requests->writable_row(cold, 0, true, 1, 1) != nullptr);
    mark_all_layers(*requests, cold, 0, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(cold, 0, 2));
    REQUIRE(requests->push(cold, second_page, 0, 1));
    for (uint32_t layer = 0; layer < pool_layout.n_layer; ++layer) {
        float * key   = requests->writable_row(cold, 1, false, layer, 0);
        float * value = requests->writable_row(cold, 1, true, layer, 0);
        REQUIRE(key != nullptr && value != nullptr);
        key[0]   = float(10 + layer);
        value[0] = float(-10 - int(layer));
    }
    mark_all_layers(*requests, cold, 1, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(cold, 1, 1));
    const float cold_logits[] = { 1.0f, 2.0f, 3.0f };
    REQUIRE(requests->set_logits(cold, 12, cold_logits, 3));
    REQUIRE(requests->commit(cold));
    REQUIRE(!requests->abort(cold));
    REQUIRE(requests->published_count(first_request) == 2);
    REQUIRE(requests->cursor(first_request) == 13);
    REQUIRE(requests->logits_valid(first_request));
    REQUIRE(requests->logits(first_request)[2] == 3.0f);
    REQUIRE(pool->state(first_page) == llama_slang_paged_pool::page_state::sealed);
    REQUIRE(pool->mapping_references(first_page) == 1);
    REQUIRE(!pool->release(first_page));

    const auto overflow_page = pool->reserve();
    REQUIRE(overflow_page != 0);
    const auto overflow = requests->begin(second_request, UINT64_MAX, 1);
    REQUIRE(overflow != 0);
    REQUIRE(!requests->push(overflow, overflow_page, 0, 2));
    REQUIRE(pool->exclusive_transaction(overflow_page) == 0);
    REQUIRE(requests->abort(overflow));
    REQUIRE(pool->release(overflow_page));

    const auto wrong_position = requests->begin(second_request, 0, 1);
    REQUIRE(wrong_position != 0);
    REQUIRE(!requests->push(wrong_position, first_page, 2, 0));
    REQUIRE(requests->abort(wrong_position));
    REQUIRE(pool->read_borrows(first_page) == 0);

    const auto shared = requests->begin(second_request, 10, 2);
    REQUIRE(shared != 0);
    REQUIRE(requests->push(shared, first_page, 2, 0));
    REQUIRE(requests->push(shared, second_page, 1, 0));
    REQUIRE(pool->read_borrows(first_page) == 1);
    REQUIRE(requests->set_logits(shared, 12, cold_logits, 3));
    REQUIRE(requests->commit(shared));
    REQUIRE(pool->mapping_references(first_page) == 2);
    REQUIRE(pool->mapping_references(second_page) == 2);

    const auto duplicate = requests->begin(second_request, 10, 2);
    REQUIRE(duplicate != 0);
    REQUIRE(requests->push(duplicate, first_page, 2, 0));
    REQUIRE(!requests->push(duplicate, first_page, 2, 0));
    REQUIRE(!requests->logits_valid(second_request));
    REQUIRE(requests->abort(duplicate));
    REQUIRE(pool->read_borrows(first_page) == 0);

    const auto cow_page = pool->reserve();
    REQUIRE(cow_page != 0);
    REQUIRE(pool->copy_rows(second_page, cow_page, 1));
    for (uint32_t layer = 0; layer < pool_layout.n_layer; ++layer) {
        const float * source_key = pool->row(second_page, false, layer, 0);
        const float * copied_key = pool->row(cow_page, false, layer, 0);
        REQUIRE(source_key != nullptr && copied_key != nullptr && copied_key[0] == source_key[0]);
    }
    const auto cow = requests->begin(second_request, 10, 2);
    REQUIRE(cow != 0);
    REQUIRE(requests->push(cow, first_page, 2, 0));
    REQUIRE(requests->push(cow, cow_page, 1, 1));
    for (uint32_t layer = 0; layer < pool_layout.n_layer; ++layer) {
        float * appended = requests->writable_row(cow, 1, false, layer, 1);
        REQUIRE(appended != nullptr);
        appended[0] = float(20 + layer);
    }
    mark_all_layers(*requests, cow, 1, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(cow, 1, 2));
    const float cow_logits[] = { 4.0f, 5.0f, 6.0f };
    REQUIRE(requests->set_logits(cow, 13, cow_logits, 3));
    REQUIRE(requests->commit(cow));
    REQUIRE(pool->mapping_references(first_page) == 2);
    REQUIRE(pool->mapping_references(second_page) == 1);
    REQUIRE(pool->mapping_references(cow_page) == 1);
    REQUIRE(requests->logits(second_request)[0] == 4.0f);
    REQUIRE(pool->occupied(second_page) == 1);
    for (uint32_t layer = 0; layer < pool_layout.n_layer; ++layer) {
        REQUIRE(pool->row(cow_page, false, layer, 0)[0] == pool->row(second_page, false, layer, 0)[0]);
    }

    const auto aborted_page = pool->reserve();
    REQUIRE(aborted_page != 0);
    const auto doomed = requests->begin(second_request, 10, 2);
    REQUIRE(doomed != 0);
    REQUIRE(requests->push(doomed, first_page, 2, 0));
    REQUIRE(requests->push(doomed, aborted_page, 0, 1));
    float * abandoned = requests->writable_row(doomed, 1, false, 0, 0);
    REQUIRE(abandoned != nullptr);
    abandoned[0] = 99.0f;
    mark_all_layers(*requests, doomed, 1, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(doomed, 1, 1));
    REQUIRE(requests->set_logits(doomed, 12, cow_logits, 3));
    REQUIRE(requests->fail(doomed));
    REQUIRE(requests->abort(doomed));
    REQUIRE(pool->occupied(aborted_page) == 0);
    REQUIRE(pool->state(aborted_page) == llama_slang_paged_pool::page_state::writable);
    REQUIRE(pool->writable_row(aborted_page, false, 0, 0)[0] == 0.0f);
    REQUIRE(pool->read_borrows(first_page) == 0);
    REQUIRE(pool->release(aborted_page));

    const auto bad_logits = requests->begin(second_request, 10, 2);
    REQUIRE(bad_logits != 0);
    REQUIRE(requests->push(bad_logits, first_page, 2, 0));
    REQUIRE(requests->push(bad_logits, cow_page, 2, 0));
    REQUIRE(requests->set_logits(bad_logits, 11, cow_logits, 3));
    REQUIRE(!requests->commit(bad_logits));
    REQUIRE(!requests->abort(bad_logits));
    REQUIRE(pool->read_borrows(first_page) == 0);

    REQUIRE(!requests->close_request(cold));
    REQUIRE(!requests->abort(first_request));

    const auto cancelled_page = pool->reserve();
    REQUIRE(cancelled_page != 0);
    const auto cancelled = requests->begin(second_request, 10, 2);
    REQUIRE(cancelled != 0);
    REQUIRE(requests->push(cancelled, first_page, 2, 0));
    REQUIRE(requests->push(cancelled, cancelled_page, 0, 1));
    REQUIRE(requests->writable_row(cancelled, 1, false, 0, 0) != nullptr);
    mark_all_layers(*requests, cancelled, 1, pool_layout.n_layer);
    REQUIRE(requests->finish_rows(cancelled, 1, 1));
    REQUIRE(requests->cancel_request(second_request));
    REQUIRE(pool->read_borrows(first_page) == 0);
    REQUIRE(pool->occupied(cancelled_page) == 0);
    REQUIRE(pool->release(cancelled_page));
    REQUIRE(requests->close_request(first_request));
    REQUIRE(requests->begin(first_request, 0, 1) == 0);
    REQUIRE(pool->mapping_references(first_page) == 0);
    REQUIRE(pool->mapping_references(second_page) == 0);
    REQUIRE(pool->mapping_references(cow_page) == 0);
    REQUIRE(pool->release(first_page));
    REQUIRE(pool->release(second_page));
    REQUIRE(pool->release(cow_page));

    auto invalid_limits                = request_limits;
    invalid_limits.metadata_byte_limit = 1;
    REQUIRE(llama_slang_paged_requests::create(pool, invalid_limits) == nullptr);
    invalid_limits                   = request_limits;
    invalid_limits.logits_byte_limit = 1;
    REQUIRE(llama_slang_paged_requests::create(pool, invalid_limits) == nullptr);

    std::weak_ptr<llama_slang_paged_pool> pool_lifetime = pool;
    pool.reset();
    REQUIRE(!pool_lifetime.expired());
    requests.reset();
    REQUIRE(pool_lifetime.expired());
    return 0;
}
