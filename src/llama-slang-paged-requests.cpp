#include "llama-slang-paged-requests.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace {

std::atomic<uint64_t> g_next_request_identity{ 1 };

uint64_t next_identity() {
    constexpr uint64_t limit   = uint64_t(std::numeric_limits<int64_t>::max());
    uint64_t           current = g_next_request_identity.load(std::memory_order_relaxed);
    while (current <= limit) {
        if (g_next_request_identity.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
            return current;
        }
    }
    return 0;
}

bool checked_mul(size_t left, size_t right, size_t & result) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_add(size_t left, size_t right, size_t & result) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

}  // namespace

std::unique_ptr<llama_slang_paged_requests> llama_slang_paged_requests::create(
    std::shared_ptr<llama_slang_paged_pool> pool,
    const limits &                          config) {
    if (pool == nullptr || config.max_requests == 0 || config.max_pages_per_request == 0 ||
        config.vocabulary_width == 0 || config.metadata_byte_limit == 0 || config.logits_byte_limit == 0) {
        return nullptr;
    }

    size_t slots             = 0;
    size_t metadata_bytes    = 0;
    size_t request_bytes     = 0;
    size_t transaction_bytes = 0;
    size_t published_bytes   = 0;
    size_t staged_bytes      = 0;
    size_t progress_bytes    = 0;
    if (!checked_mul(config.max_requests, config.max_pages_per_request, slots) ||
        !checked_mul(config.max_requests, sizeof(request_record), request_bytes) ||
        !checked_mul(config.max_requests, sizeof(transaction_record), transaction_bytes) ||
        !checked_mul(slots, sizeof(published_entry), published_bytes) ||
        !checked_mul(slots, sizeof(staged_entry), staged_bytes) ||
        !checked_mul(slots, pool->config().n_layer, progress_bytes) ||
        !checked_add(request_bytes, transaction_bytes, metadata_bytes) ||
        !checked_add(metadata_bytes, published_bytes, metadata_bytes) ||
        !checked_add(metadata_bytes, staged_bytes, metadata_bytes) ||
        !checked_add(metadata_bytes, progress_bytes, metadata_bytes) || metadata_bytes > config.metadata_byte_limit) {
        return nullptr;
    }

    size_t logits_elements = 0;
    size_t logits_bytes    = 0;
    if (!checked_mul(config.max_requests, config.vocabulary_width, logits_elements) ||
        !checked_mul(logits_elements, sizeof(float), logits_bytes) || !checked_mul(logits_bytes, 2, logits_bytes) ||
        logits_bytes > config.logits_byte_limit) {
        return nullptr;
    }

    try {
        return std::unique_ptr<llama_slang_paged_requests>(new llama_slang_paged_requests(std::move(pool), config));
    } catch (const std::bad_alloc &) {
        return nullptr;
    } catch (const std::length_error &) {
        return nullptr;
    }
}

llama_slang_paged_requests::llama_slang_paged_requests(std::shared_ptr<llama_slang_paged_pool> pool,
                                                       const limits &                          config) :
    pool_(std::move(pool)),
    config_(config),
    requests_(config.max_requests),
    transactions_(config.max_requests),
    published_(size_t(config.max_requests) * config.max_pages_per_request),
    staged_(size_t(config.max_requests) * config.max_pages_per_request),
    published_logits_(size_t(config.max_requests) * config.vocabulary_width),
    staged_logits_(size_t(config.max_requests) * config.vocabulary_width),
    layer_progress_(size_t(config.max_requests) * config.max_pages_per_request * pool_->config().n_layer) {}

llama_slang_paged_requests::~llama_slang_paged_requests() {
    for (size_t i = 0; i < requests_.size(); ++i) {
        if (requests_[i].active) {
            cancel_request(requests_[i].identity);
        }
    }
}

llama_slang_paged_requests::request_record * llama_slang_paged_requests::find_request(request_handle request) {
    if (request == 0) {
        return nullptr;
    }
    for (auto & record : requests_) {
        if (record.active && record.identity == request) {
            return &record;
        }
    }
    return nullptr;
}

const llama_slang_paged_requests::request_record * llama_slang_paged_requests::find_request(
    request_handle request) const {
    return const_cast<llama_slang_paged_requests *>(this)->find_request(request);
}

llama_slang_paged_requests::transaction_record * llama_slang_paged_requests::find_transaction(
    transaction_handle transaction) {
    if (transaction == 0) {
        return nullptr;
    }
    for (auto & record : transactions_) {
        if (record.active && record.identity == transaction) {
            return &record;
        }
    }
    return nullptr;
}

const llama_slang_paged_requests::transaction_record * llama_slang_paged_requests::find_transaction(
    transaction_handle transaction) const {
    return const_cast<llama_slang_paged_requests *>(this)->find_transaction(transaction);
}

size_t llama_slang_paged_requests::request_index(const request_record & request) const {
    return size_t(&request - requests_.data());
}

llama_slang_paged_requests::published_entry * llama_slang_paged_requests::published_slice(size_t index) {
    return published_.data() + index * config_.max_pages_per_request;
}

const llama_slang_paged_requests::published_entry * llama_slang_paged_requests::published_slice(size_t index) const {
    return published_.data() + index * config_.max_pages_per_request;
}

llama_slang_paged_requests::staged_entry * llama_slang_paged_requests::staged_slice(size_t index) {
    return staged_.data() + index * config_.max_pages_per_request;
}

float * llama_slang_paged_requests::published_logits_slice(size_t index) {
    return published_logits_.data() + index * config_.vocabulary_width;
}

const float * llama_slang_paged_requests::published_logits_slice(size_t index) const {
    return published_logits_.data() + index * config_.vocabulary_width;
}

float * llama_slang_paged_requests::staged_logits_slice(size_t index) {
    return staged_logits_.data() + index * config_.vocabulary_width;
}

uint8_t * llama_slang_paged_requests::layer_progress_slice(size_t index, uint32_t entry) {
    return layer_progress_.data() + (index * config_.max_pages_per_request + entry) * pool_->config().n_layer;
}

const uint8_t * llama_slang_paged_requests::layer_progress_slice(size_t index, uint32_t entry) const {
    return layer_progress_.data() + (index * config_.max_pages_per_request + entry) * pool_->config().n_layer;
}

llama_slang_paged_requests::request_handle llama_slang_paged_requests::open_request(uint64_t execution_namespace) {
    if (execution_namespace == 0 || execution_namespace != pool_->config().execution_namespace) {
        return 0;
    }
    for (size_t i = 0; i < requests_.size(); ++i) {
        if (!requests_[i].active) {
            const request_handle identity = next_identity();
            if (identity == 0) {
                return 0;
            }
            requests_[i]     = { identity, execution_namespace, 0, 0, true, false };
            transactions_[i] = {};
            std::fill_n(published_slice(i), config_.max_pages_per_request, published_entry{});
            std::fill_n(staged_slice(i), config_.max_pages_per_request, staged_entry{});
            std::fill_n(published_logits_slice(i), config_.vocabulary_width, 0.0f);
            std::fill_n(staged_logits_slice(i), config_.vocabulary_width, 0.0f);
            return identity;
        }
    }
    return 0;
}

bool llama_slang_paged_requests::close_request(request_handle request) {
    request_record * record = find_request(request);
    if (record == nullptr) {
        return false;
    }
    const size_t index = request_index(*record);
    if (transactions_[index].active && !abort_record(index)) {
        std::abort();
    }
    const published_entry * table = published_slice(index);
    for (uint32_t i = 0; i < record->published_count; ++i) {
        if (!pool_->unmap_for_request(table[i].page, request)) {
            std::abort();
        }
    }
    *record = {};
    std::fill_n(published_slice(index), config_.max_pages_per_request, published_entry{});
    std::fill_n(published_logits_slice(index), config_.vocabulary_width, 0.0f);
    return true;
}

bool llama_slang_paged_requests::cancel_request(request_handle request) {
    return close_request(request);
}

llama_slang_paged_requests::transaction_handle llama_slang_paged_requests::begin(request_handle request,
                                                                                 uint64_t       table_base,
                                                                                 uint32_t       expected_pages) {
    request_record * record = find_request(request);
    if (record == nullptr || expected_pages == 0 || expected_pages > config_.max_pages_per_request) {
        return 0;
    }
    const size_t index = request_index(*record);
    if (transactions_[index].active) {
        return 0;
    }
    const uint64_t page_tokens = pool_->config().page_tokens;
    if (uint64_t(expected_pages - 1) > (std::numeric_limits<uint64_t>::max() - table_base) / page_tokens) {
        return 0;
    }
    const transaction_handle identity = next_identity();
    if (identity == 0) {
        return 0;
    }
    transactions_[index] = { identity, request, table_base, expected_pages, 0, 0, true, false, false };
    std::fill_n(staged_slice(index), config_.max_pages_per_request, staged_entry{});
    std::fill_n(staged_logits_slice(index), config_.vocabulary_width, 0.0f);
    std::fill_n(layer_progress_slice(index, 0), size_t(config_.max_pages_per_request) * pool_->config().n_layer, 0);
    return identity;
}

bool llama_slang_paged_requests::fail_valid(transaction_record & transaction) {
    transaction.failed       = true;
    request_record * request = find_request(transaction.request);
    if (request != nullptr) {
        request->logits_valid = false;
    }
    return false;
}

bool llama_slang_paged_requests::push(transaction_handle transaction,
                                      page_handle        page,
                                      uint32_t           valid_rows,
                                      uint32_t           writable_capacity) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return false;
    }
    request_record * request = find_request(record->request);
    if (request == nullptr) {
        return fail_valid(*record);
    }
    const size_t index = request_index(*request);
    if (record->staged_count >= record->expected_pages) {
        return fail_valid(*record);
    }
    staged_entry * entries = staged_slice(index);
    for (uint32_t i = 0; i < record->staged_count; ++i) {
        if (entries[i].page == page) {
            return fail_valid(*record);
        }
    }
    const auto     state             = pool_->state(page);
    const uint32_t page_tokens       = pool_->config().page_tokens;
    const uint64_t expected_position = record->table_base + uint64_t(record->staged_count) * page_tokens;
    if (page == 0 || valid_rows != pool_->occupied(page) || valid_rows > page_tokens ||
        writable_capacity > page_tokens - valid_rows ||
        uint64_t(valid_rows) + writable_capacity > std::numeric_limits<uint64_t>::max() - expected_position) {
        return fail_valid(*record);
    }
    bool exclusive = false;
    if (state == llama_slang_paged_pool::page_state::sealed) {
        if (valid_rows == 0 || writable_capacity != 0 || !pool_->position_bound(page) ||
            pool_->position_base(page) != expected_position || !pool_->borrow_sealed(page)) {
            return fail_valid(*record);
        }
    } else if (state == llama_slang_paged_pool::page_state::writable) {
        if (!pool_->claim_exclusive(page, transaction, record->request)) {
            return fail_valid(*record);
        }
        if (!pool_->bind_position_exclusive(page, transaction, expected_position)) {
            if (!pool_->abort_exclusive(page, transaction)) {
                std::abort();
            }
            return fail_valid(*record);
        }
        exclusive = true;
    } else {
        return fail_valid(*record);
    }
    entries[record->staged_count++] = { page, valid_rows, valid_rows, valid_rows + writable_capacity, exclusive };
    record->logits_valid            = false;
    return true;
}

float * llama_slang_paged_requests::writable_row(transaction_handle transaction,
                                                 uint32_t           entry,
                                                 bool               value,
                                                 uint32_t           layer,
                                                 uint32_t           row) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return nullptr;
    }
    request_record * request = find_request(record->request);
    if (request == nullptr || entry >= record->staged_count) {
        fail_valid(*record);
        return nullptr;
    }
    staged_entry & staged = staged_slice(request_index(*request))[entry];
    if (!staged.exclusive || row < staged.valid_rows || row >= staged.writable_end) {
        fail_valid(*record);
        return nullptr;
    }
    float * result = pool_->writable_row_exclusive(staged.page, transaction, value, layer, row);
    if (result == nullptr) {
        fail_valid(*record);
        return nullptr;
    }
    record->logits_valid = false;
    return result;
}

bool llama_slang_paged_requests::finish_rows(transaction_handle transaction, uint32_t entry, uint32_t new_valid) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return false;
    }
    request_record * request = find_request(record->request);
    if (request == nullptr || entry >= record->staged_count) {
        return fail_valid(*record);
    }
    staged_entry & staged = staged_slice(request_index(*request))[entry];
    if (new_valid > staged.initial_rows) {
        const uint8_t * progress = layer_progress_slice(request_index(*request), entry);
        for (uint32_t layer = 0; layer < pool_->config().n_layer; ++layer) {
            if (progress[layer] == 0) {
                return fail_valid(*record);
            }
        }
    }
    if (!staged.exclusive || new_valid < staged.valid_rows || new_valid > staged.writable_end ||
        !pool_->set_occupied_exclusive(staged.page, transaction, new_valid)) {
        return fail_valid(*record);
    }
    staged.valid_rows    = new_valid;
    record->logits_valid = false;
    return true;
}

bool llama_slang_paged_requests::mark_layer_written(transaction_handle transaction, uint32_t entry, uint32_t layer) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return false;
    }
    request_record * request = find_request(record->request);
    if (request == nullptr || entry >= record->staged_count || layer >= pool_->config().n_layer) {
        return fail_valid(*record);
    }
    const staged_entry & staged   = staged_slice(request_index(*request))[entry];
    uint8_t *            progress = layer_progress_slice(request_index(*request), entry);
    if (!staged.exclusive || staged.writable_end <= staged.initial_rows || progress[layer] != 0) {
        return fail_valid(*record);
    }
    progress[layer] = 1;
    return true;
}

bool llama_slang_paged_requests::set_logits(transaction_handle transaction,
                                            uint64_t           position,
                                            const float *      logits,
                                            uint32_t           count) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return false;
    }
    request_record * request = find_request(record->request);
    if (request == nullptr || logits == nullptr || count != config_.vocabulary_width) {
        return fail_valid(*record);
    }
    std::memcpy(staged_logits_slice(request_index(*request)), logits, size_t(count) * sizeof(float));
    record->logits_position = position;
    record->logits_valid    = true;
    return true;
}

bool llama_slang_paged_requests::fail(transaction_handle transaction) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed) {
        return false;
    }
    fail_valid(*record);
    return true;
}

bool llama_slang_paged_requests::commit(transaction_handle transaction) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr) {
        return false;
    }
    const size_t transaction_index = size_t(record - transactions_.data());
    auto         reject            = [&]() {
        if (!record->failed) {
            fail_valid(*record);
        }
        if (!abort_record(transaction_index)) {
            std::abort();
        }
        return false;
    };
    if (record->failed || record->staged_count != record->expected_pages) {
        return reject();
    }
    request_record * request = find_request(record->request);
    if (request == nullptr) {
        return reject();
    }
    const size_t   index       = request_index(*request);
    staged_entry * staged      = staged_slice(index);
    const uint32_t page_tokens = pool_->config().page_tokens;
    for (uint32_t i = 0; i < record->staged_count; ++i) {
        const uint64_t expected_position = record->table_base + uint64_t(i) * page_tokens;
        if (staged[i].valid_rows == 0 || staged[i].valid_rows > page_tokens || !pool_->position_bound(staged[i].page) ||
            pool_->position_base(staged[i].page) != expected_position ||
            (i + 1 < record->staged_count && staged[i].valid_rows != page_tokens)) {
            return reject();
        }
        if (staged[i].exclusive) {
            if (pool_->exclusive_transaction(staged[i].page) != transaction ||
                pool_->state(staged[i].page) != llama_slang_paged_pool::page_state::writable) {
                return reject();
            }
        } else if (pool_->state(staged[i].page) != llama_slang_paged_pool::page_state::sealed ||
                   pool_->occupied(staged[i].page) != staged[i].valid_rows ||
                   pool_->read_borrows(staged[i].page) == 0) {
            return reject();
        }
    }
    const uint64_t last_base = record->table_base + uint64_t(record->staged_count - 1) * page_tokens;
    if (staged[record->staged_count - 1].valid_rows > std::numeric_limits<uint64_t>::max() - last_base) {
        return reject();
    }
    const uint64_t new_cursor = last_base + staged[record->staged_count - 1].valid_rows;
    if (!record->logits_valid || new_cursor == 0 || record->logits_position != new_cursor - 1) {
        return reject();
    }

    const published_entry * old_table      = published_slice(index);
    auto                    appears_in_new = [&](page_handle page) {
        for (uint32_t i = 0; i < record->staged_count; ++i) {
            if (staged[i].page == page) {
                return true;
            }
        }
        return false;
    };
    auto appeared_before = [&](uint32_t entry) {
        for (uint32_t i = 0; i < request->published_count; ++i) {
            if (old_table[i].page == staged[entry].page) {
                return true;
            }
        }
        return false;
    };

    for (uint32_t i = 0; i < record->staged_count; ++i) {
        if (!staged[i].exclusive && !appeared_before(i) &&
            pool_->mapping_references(staged[i].page) == std::numeric_limits<uint32_t>::max()) {
            return reject();
        }
    }

    // Publication begins here. All storage and reference transitions were
    // prevalidated; under the serialized-owner contract none can now fail.
    for (uint32_t i = 0; i < record->staged_count; ++i) {
        bool ok = true;
        if (staged[i].exclusive) {
            ok = pool_->seal_exclusive(staged[i].page, transaction) &&
                 pool_->publish_exclusive(staged[i].page, transaction, record->request);
        } else if (appeared_before(i)) {
            ok = pool_->return_sealed(staged[i].page);
        } else {
            ok = pool_->map_for_request(staged[i].page, record->request) && pool_->return_sealed(staged[i].page);
        }
        if (!ok) {
            std::abort();
        }
    }

    for (uint32_t i = 0; i < request->published_count; ++i) {
        if (!appears_in_new(old_table[i].page) && !pool_->unmap_for_request(old_table[i].page, record->request)) {
            std::abort();
        }
    }
    published_entry * new_table = published_slice(index);
    for (uint32_t i = 0; i < record->staged_count; ++i) {
        new_table[i] = { staged[i].page, staged[i].valid_rows, record->table_base + uint64_t(i) * page_tokens };
    }
    for (uint32_t i = record->staged_count; i < config_.max_pages_per_request; ++i) {
        new_table[i] = {};
    }
    std::copy_n(staged_logits_slice(index), config_.vocabulary_width, published_logits_slice(index));
    request->published_count = record->staged_count;
    request->cursor          = new_cursor;
    request->logits_valid    = true;
    clear_transaction(index);
    return true;
}

void llama_slang_paged_requests::clear_transaction(size_t index) {
    transactions_[index] = {};
    std::fill_n(staged_slice(index), config_.max_pages_per_request, staged_entry{});
    std::fill_n(staged_logits_slice(index), config_.vocabulary_width, 0.0f);
    std::fill_n(layer_progress_slice(index, 0), size_t(config_.max_pages_per_request) * pool_->config().n_layer, 0);
}

bool llama_slang_paged_requests::abort_record(size_t index) {
    transaction_record & record = transactions_[index];
    if (!record.active) {
        return false;
    }
    staged_entry * staged = staged_slice(index);
    for (uint32_t i = 0; i < record.staged_count; ++i) {
        const bool ok = staged[i].exclusive ? pool_->abort_exclusive(staged[i].page, record.identity) :
                                              pool_->return_sealed(staged[i].page);
        if (!ok) {
            return false;
        }
    }
    clear_transaction(index);
    return true;
}

bool llama_slang_paged_requests::abort(transaction_handle transaction) {
    transaction_record * record = find_transaction(transaction);
    if (record == nullptr) {
        return false;
    }
    request_record * request = find_request(record->request);
    return request != nullptr && abort_record(request_index(*request));
}

uint32_t llama_slang_paged_requests::published_count(request_handle request) const {
    const request_record * record = find_request(request);
    return record == nullptr ? 0 : record->published_count;
}

const llama_slang_paged_requests::published_entry * llama_slang_paged_requests::published_table(
    request_handle request) const {
    const request_record * record = find_request(request);
    return record == nullptr ? nullptr : published_slice(request_index(*record));
}

uint64_t llama_slang_paged_requests::cursor(request_handle request) const {
    const request_record * record = find_request(request);
    return record == nullptr ? 0 : record->cursor;
}

const float * llama_slang_paged_requests::logits(request_handle request) const {
    const request_record * record = find_request(request);
    return record == nullptr || !record->logits_valid ? nullptr : published_logits_slice(request_index(*record));
}

bool llama_slang_paged_requests::logits_valid(request_handle request) const {
    const request_record * record = find_request(request);
    return record != nullptr && record->logits_valid;
}

uint32_t llama_slang_paged_requests::execution_count(transaction_handle transaction) const {
    const transaction_record * record = find_transaction(transaction);
    return record == nullptr || record->failed ? 0 : record->staged_count;
}

bool llama_slang_paged_requests::execution_compatible(transaction_handle transaction,
                                                      uint32_t           n_layer,
                                                      uint32_t           n_head_kv,
                                                      uint32_t           head_dim) const {
    const transaction_record * record = find_transaction(transaction);
    const auto &               config = pool_->config();
    return record != nullptr && !record->failed && config.n_layer == n_layer && config.n_head_kv == n_head_kv &&
           config.head_dim == head_dim;
}

bool llama_slang_paged_requests::execution_entry_at(transaction_handle transaction,
                                                    uint32_t           index,
                                                    execution_entry &  result) const {
    const transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed || index >= record->staged_count) {
        return false;
    }
    const request_record * request = find_request(record->request);
    if (request == nullptr) {
        return false;
    }
    const staged_entry & entry = staged_[request_index(*request) * config_.max_pages_per_request + index];
    result = { entry.page, entry.valid_rows, entry.writable_end,
               record->table_base + uint64_t(index) * pool_->config().page_tokens, entry.exclusive };
    return true;
}

bool llama_slang_paged_requests::layer_unwritten(transaction_handle transaction, uint32_t entry, uint32_t layer) const {
    const transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed || entry >= record->staged_count || layer >= pool_->config().n_layer) {
        return false;
    }
    const request_record * request = find_request(record->request);
    return request != nullptr && layer_progress_slice(request_index(*request), entry)[layer] == 0;
}

const float * llama_slang_paged_requests::readable_row(transaction_handle transaction,
                                                       uint32_t           entry_index,
                                                       bool               value,
                                                       uint32_t           layer,
                                                       uint32_t           row) const {
    const transaction_record * record = find_transaction(transaction);
    if (record == nullptr || record->failed || entry_index >= record->staged_count) {
        return nullptr;
    }
    const request_record * request = find_request(record->request);
    if (request == nullptr) {
        return nullptr;
    }
    const staged_entry & entry = staged_[request_index(*request) * config_.max_pages_per_request + entry_index];
    if (row >= entry.valid_rows) {
        return nullptr;
    }
    return entry.exclusive ? pool_->row_exclusive(entry.page, transaction, value, layer, row) :
                             pool_->row(entry.page, value, layer, row);
}
