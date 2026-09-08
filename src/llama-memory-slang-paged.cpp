#include "llama-memory-slang-paged.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace {

std::atomic<uint64_t> g_next_page_identity{ 1 };

uint64_t next_page_identity() {
    uint64_t current = g_next_page_identity.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<uint64_t>::max()) {
        if (g_next_page_identity.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
            return current;
        }
    }
    return 0;
}

bool checked_mul(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

}  // namespace

std::unique_ptr<llama_slang_paged_pool> llama_slang_paged_pool::create(const layout & config) {
    if (config.n_layer == 0 || config.page_tokens == 0 || config.n_head_kv == 0 || config.head_dim == 0 ||
        config.page_capacity == 0 || config.execution_namespace == 0 || config.byte_limit == 0 ||
        config.descriptor_byte_limit == 0) {
        return nullptr;
    }

    size_t elements = 0;
    if (!checked_mul(config.n_layer, config.page_tokens, elements) ||
        !checked_mul(elements, config.n_head_kv, elements) || !checked_mul(elements, config.head_dim, elements) ||
        !checked_mul(elements, 2, elements)) {
        return nullptr;
    }
    size_t bytes = 0;
    if (!checked_mul(elements, sizeof(float), bytes) || bytes > config.byte_limit) {
        return nullptr;
    }

    size_t descriptor_bytes = 0;
    if (!checked_mul(config.page_capacity, sizeof(page), descriptor_bytes) ||
        descriptor_bytes > config.descriptor_byte_limit) {
        return nullptr;
    }

    try {
        return std::unique_ptr<llama_slang_paged_pool>(new llama_slang_paged_pool(config, bytes));
    } catch (const std::bad_alloc &) {
        return nullptr;
    } catch (const std::length_error &) {
        return nullptr;
    }
}

llama_slang_paged_pool::llama_slang_paged_pool(const layout & config, size_t page_bytes) :
    config_(config),
    row_elements_(size_t(config.n_head_kv) * config.head_dim),
    layer_elements_(size_t(config.page_tokens) * row_elements_),
    plane_elements_(size_t(config.n_layer) * layer_elements_),
    page_bytes_(page_bytes),
    pages_(config.page_capacity) {}

llama_slang_paged_pool::~llama_slang_paged_pool() {
    for (auto & page : pages_) {
        if (page.data != nullptr) {
            ::operator delete(page.data, std::align_val_t(64));
        }
    }
}

llama_slang_paged_pool::page * llama_slang_paged_pool::find(page_handle handle) {
    if (handle == 0) {
        return nullptr;
    }
    for (auto & page : pages_) {
        if (page.identity == handle && page.state != page_state::free) {
            return &page;
        }
    }
    return nullptr;
}

const llama_slang_paged_pool::page * llama_slang_paged_pool::find(page_handle handle) const {
    return const_cast<llama_slang_paged_pool *>(this)->find(handle);
}

llama_slang_paged_pool::page_handle llama_slang_paged_pool::reserve() {
    if (allocated_bytes_ > config_.byte_limit - page_bytes_) {
        return 0;
    }
    page * available = nullptr;
    for (auto & candidate : pages_) {
        if (candidate.state == page_state::free) {
            available = &candidate;
            break;
        }
    }
    if (available == nullptr) {
        return 0;
    }
    try {
        available->data = static_cast<float *>(::operator new(page_bytes_, std::align_val_t(64)));
    } catch (const std::bad_alloc &) {
        return 0;
    }
    const page_handle identity = next_page_identity();
    if (identity == 0) {
        ::operator delete(available->data, std::align_val_t(64));
        available->data = nullptr;
        return 0;
    }
    std::memset(available->data, 0, page_bytes_);
    available->occupied                         = 0;
    available->mapping_references               = 0;
    available->read_borrows                     = 0;
    available->writable_owner                   = 0;
    available->exclusive_transaction            = 0;
    available->exclusive_request                = 0;
    available->exclusive_initial_occupied       = 0;
    available->position_base                    = 0;
    available->exclusive_initial_position       = 0;
    available->position_bound                   = false;
    available->exclusive_initial_position_bound = false;
    available->identity                         = identity;
    available->state                            = page_state::writable;
    allocated_bytes_ += page_bytes_;
    return identity;
}

bool llama_slang_paged_pool::release(page_handle handle) {
    page * page = find(handle);
    if (page == nullptr || page->mapping_references != 0 || page->read_borrows != 0 ||
        page->exclusive_transaction != 0) {
        return false;
    }
    ::operator delete(page->data, std::align_val_t(64));
    page->data                             = nullptr;
    page->identity                         = 0;
    page->occupied                         = 0;
    page->mapping_references               = 0;
    page->read_borrows                     = 0;
    page->writable_owner                   = 0;
    page->exclusive_transaction            = 0;
    page->exclusive_request                = 0;
    page->exclusive_initial_occupied       = 0;
    page->position_base                    = 0;
    page->exclusive_initial_position       = 0;
    page->position_bound                   = false;
    page->exclusive_initial_position_bound = false;
    page->state                            = page_state::free;
    allocated_bytes_ -= page_bytes_;
    return true;
}

bool llama_slang_paged_pool::set_occupied(page_handle handle, uint32_t rows) {
    page * page = find(handle);
    if (page == nullptr || page->state != page_state::writable || page->mapping_references != 0 ||
        page->exclusive_transaction != 0 || rows < page->occupied || rows > config_.page_tokens) {
        return false;
    }
    page->occupied = rows;
    return true;
}

bool llama_slang_paged_pool::seal(page_handle handle) {
    page * page = find(handle);
    if (page == nullptr || page->state != page_state::writable || page->mapping_references != 0 ||
        page->exclusive_transaction != 0 || page->occupied == 0) {
        return false;
    }
    page->state = page_state::sealed;
    return true;
}

size_t llama_slang_paged_pool::row_offset(bool value, uint32_t layer, uint32_t row) const {
    return (value ? plane_elements_ : 0) + size_t(layer) * layer_elements_ + size_t(row) * row_elements_;
}

float * llama_slang_paged_pool::writable_row(page_handle handle, bool value, uint32_t layer, uint32_t row) {
    page * page = find(handle);
    if (page == nullptr || page->state != page_state::writable || page->mapping_references != 0 ||
        page->exclusive_transaction != 0 || layer >= config_.n_layer || row >= config_.page_tokens ||
        row < page->occupied) {
        return nullptr;
    }
    return page->data + row_offset(value, layer, row);
}

const float * llama_slang_paged_pool::row(page_handle handle, bool value, uint32_t layer, uint32_t row) const {
    const page * page = find(handle);
    if (page == nullptr || page->exclusive_transaction != 0 || layer >= config_.n_layer || row >= page->occupied) {
        return nullptr;
    }
    return page->data + row_offset(value, layer, row);
}

const float * llama_slang_paged_pool::row_exclusive(page_handle handle,
                                                    uint64_t    transaction,
                                                    bool        value,
                                                    uint32_t    layer,
                                                    uint32_t    row) const {
    const page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction || layer >= config_.n_layer ||
        row >= page->occupied) {
        return nullptr;
    }
    return page->data + row_offset(value, layer, row);
}

bool llama_slang_paged_pool::copy_rows(page_handle source, page_handle destination, uint32_t rows) {
    const page * src = find(source);
    page *       dst = find(destination);
    if (src == nullptr || dst == nullptr || src == dst || src->state != page_state::sealed ||
        src->exclusive_transaction != 0 || dst->state != page_state::writable || dst->mapping_references != 0 ||
        dst->exclusive_transaction != 0 || dst->occupied != 0 || rows == 0 || rows > src->occupied ||
        rows > config_.page_tokens) {
        return false;
    }
    const size_t bytes = size_t(rows) * row_elements_ * sizeof(float);
    for (uint32_t layer = 0; layer < config_.n_layer; ++layer) {
        std::memcpy(dst->data + row_offset(false, layer, 0), src->data + row_offset(false, layer, 0), bytes);
        std::memcpy(dst->data + row_offset(true, layer, 0), src->data + row_offset(true, layer, 0), bytes);
    }
    dst->occupied       = rows;
    dst->position_base  = src->position_base;
    dst->position_bound = src->position_bound;
    return true;
}

bool llama_slang_paged_pool::bind_position_exclusive(page_handle handle, uint64_t transaction, uint64_t position_base) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction) {
        return false;
    }
    if (page->position_bound) {
        return page->position_base == position_base;
    }
    page->position_base  = position_base;
    page->position_bound = true;
    return true;
}

bool llama_slang_paged_pool::map_for_request(page_handle handle, uint64_t request) {
    page * page = find(handle);
    if (page == nullptr || request == 0 || page->exclusive_transaction != 0 ||
        page->mapping_references == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (page->state == page_state::writable) {
        if (page->mapping_references != 0 || page->writable_owner != 0) {
            return false;
        }
        page->writable_owner = request;
    }
    ++page->mapping_references;
    return true;
}

bool llama_slang_paged_pool::unmap_for_request(page_handle handle, uint64_t request) {
    page * page = find(handle);
    if (page == nullptr || request == 0 || page->exclusive_transaction != 0 || page->mapping_references == 0) {
        return false;
    }
    if (page->state == page_state::writable) {
        if (page->mapping_references != 1 || page->writable_owner != request) {
            return false;
        }
        page->writable_owner = 0;
    }
    --page->mapping_references;
    return true;
}

bool llama_slang_paged_pool::borrow_sealed(page_handle handle) {
    page * page = find(handle);
    if (page == nullptr || page->state != page_state::sealed || page->exclusive_transaction != 0 ||
        page->read_borrows == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    ++page->read_borrows;
    return true;
}

bool llama_slang_paged_pool::return_sealed(page_handle handle) {
    page * page = find(handle);
    if (page == nullptr || page->state != page_state::sealed || page->read_borrows == 0) {
        return false;
    }
    --page->read_borrows;
    return true;
}

bool llama_slang_paged_pool::claim_exclusive(page_handle handle, uint64_t transaction, uint64_t request) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || request == 0 || page->state != page_state::writable ||
        page->exclusive_transaction != 0 || page->read_borrows != 0 || page->mapping_references != 0 ||
        page->writable_owner != 0) {
        return false;
    }
    page->exclusive_transaction            = transaction;
    page->exclusive_request                = request;
    page->exclusive_initial_occupied       = page->occupied;
    page->exclusive_initial_position       = page->position_base;
    page->exclusive_initial_position_bound = page->position_bound;
    return true;
}

bool llama_slang_paged_pool::abort_exclusive(page_handle handle, uint64_t transaction) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction) {
        return false;
    }
    const size_t clear_rows  = config_.page_tokens - page->exclusive_initial_occupied;
    const size_t clear_bytes = clear_rows * row_elements_ * sizeof(float);
    for (uint32_t layer = 0; layer < config_.n_layer; ++layer) {
        std::memset(page->data + row_offset(false, layer, page->exclusive_initial_occupied), 0, clear_bytes);
        std::memset(page->data + row_offset(true, layer, page->exclusive_initial_occupied), 0, clear_bytes);
    }
    page->occupied                         = page->exclusive_initial_occupied;
    page->state                            = page_state::writable;
    page->position_base                    = page->exclusive_initial_position;
    page->position_bound                   = page->exclusive_initial_position_bound;
    page->exclusive_transaction            = 0;
    page->exclusive_request                = 0;
    page->exclusive_initial_occupied       = 0;
    page->exclusive_initial_position       = 0;
    page->exclusive_initial_position_bound = false;
    return true;
}

bool llama_slang_paged_pool::publish_exclusive(page_handle handle, uint64_t transaction, uint64_t request) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || request == 0 || page->exclusive_transaction != transaction ||
        page->exclusive_request != request || page->mapping_references != 0 || page->writable_owner != 0) {
        return false;
    }
    page->mapping_references               = 1;
    page->writable_owner                   = page->state == page_state::writable ? request : 0;
    page->exclusive_transaction            = 0;
    page->exclusive_request                = 0;
    page->exclusive_initial_occupied       = 0;
    page->exclusive_initial_position       = 0;
    page->exclusive_initial_position_bound = false;
    return true;
}

bool llama_slang_paged_pool::set_occupied_exclusive(page_handle handle, uint64_t transaction, uint32_t rows) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction ||
        page->state != page_state::writable || rows < page->occupied || rows > config_.page_tokens) {
        return false;
    }
    page->occupied = rows;
    return true;
}

bool llama_slang_paged_pool::seal_exclusive(page_handle handle, uint64_t transaction) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction ||
        page->state != page_state::writable || page->occupied == 0) {
        return false;
    }
    page->state = page_state::sealed;
    return true;
}

float * llama_slang_paged_pool::writable_row_exclusive(page_handle handle,
                                                       uint64_t    transaction,
                                                       bool        value,
                                                       uint32_t    layer,
                                                       uint32_t    row) {
    page * page = find(handle);
    if (page == nullptr || transaction == 0 || page->exclusive_transaction != transaction ||
        page->state != page_state::writable || layer >= config_.n_layer || row >= config_.page_tokens ||
        row < page->occupied) {
        return nullptr;
    }
    return page->data + row_offset(value, layer, row);
}

llama_slang_paged_pool::page_state llama_slang_paged_pool::state(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr ? page_state::free : page->state;
}

uint32_t llama_slang_paged_pool::occupied(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr ? 0 : page->occupied;
}

uint32_t llama_slang_paged_pool::mapping_references(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr ? 0 : page->mapping_references;
}

uint32_t llama_slang_paged_pool::read_borrows(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr ? 0 : page->read_borrows;
}

uint64_t llama_slang_paged_pool::exclusive_transaction(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr ? 0 : page->exclusive_transaction;
}

bool llama_slang_paged_pool::position_bound(page_handle handle) const {
    const page * page = find(handle);
    return page != nullptr && page->position_bound;
}

uint64_t llama_slang_paged_pool::position_base(page_handle handle) const {
    const page * page = find(handle);
    return page == nullptr || !page->position_bound ? 0 : page->position_base;
}

size_t llama_slang_paged_pool::page_bytes() const {
    return page_bytes_;
}

size_t llama_slang_paged_pool::allocated_bytes() const {
    return allocated_bytes_;
}

const llama_slang_paged_pool::layout & llama_slang_paged_pool::config() const {
    return config_;
}

#ifdef LLAMA_SLANG_PAGED_TESTING
void llama_slang_paged_pool::test_set_next_identity(uint64_t next) {
    g_next_page_identity.store(next, std::memory_order_relaxed);
}
#endif
