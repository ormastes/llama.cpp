#include "llama-memory-slang-paged-cache.h"

#include "ggml-cpu.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

uint64_t make_namespace(const llama_model & model, const llama_slang_paged_params & paged) {
    uint64_t value = 1469598103934665603ULL;
    auto     mix   = [&](uint64_t part) {
        value ^= part;
        value *= 1099511628211ULL;
    };
    mix(model.hparams.n_layer());
    mix(model.hparams.n_head());
    mix(model.hparams.n_head_kv());
    mix(model.hparams.n_embd_head_k());
    mix(paged.page_tokens);
    return value == 0 ? 1 : value;
}

[[noreturn]] void reject_state_serialization() {
    throw std::runtime_error("Slang paged cache state serialization is unsupported");
}

}  // namespace

llama_memory_slang_paged_context::llama_memory_slang_paged_context(llama_memory_status status) : status_(status) {}

llama_memory_slang_paged_context::llama_memory_slang_paged_context(
    llama_memory_slang_paged_cache *               owner,
    llama_ubatch                                   ubatch,
    llama_slang_paged_requests::transaction_handle transaction,
    bool                                           sizing_only) :
    status_(LLAMA_MEMORY_STATUS_SUCCESS),
    owner_(owner),
    ubatch_(std::move(ubatch)),
    transaction_(transaction),
    sizing_only_(sizing_only) {
    if (!sizing_only_) {
        positions_.reserve(ubatch_.n_tokens);
        for (uint32_t i = 0; i < ubatch_.n_tokens; ++i) {
            positions_.push_back(ubatch_.pos[i]);
        }
        const uint64_t end    = uint64_t(positions_.back()) + 1;
        const uint64_t padded = std::max<uint64_t>(256, (end + 255) & ~uint64_t(255));
        scratch_.resize(size_t(padded) * 2);
    }
}

llama_memory_slang_paged_context::~llama_memory_slang_paged_context() {
    if (!completed_ && transaction_ != 0) {
        owner_->abort(transaction_);
    }
}

bool llama_memory_slang_paged_context::next() {
    return false;
}

bool llama_memory_slang_paged_context::apply() {
    if (status_ != LLAMA_MEMORY_STATUS_SUCCESS || applied_) {
        return false;
    }
    applied_ = true;
    return true;
}

const llama_ubatch & llama_memory_slang_paged_context::get_ubatch() const {
    return ubatch_;
}

llama_memory_status llama_memory_slang_paged_context::get_status() const {
    return status_;
}

bool llama_memory_slang_paged_context::requires_finalize() const {
    return !sizing_only_;
}

bool llama_memory_slang_paged_context::finalize(const float * logits, uint32_t count, llama_pos position) {
    if (completed_ || sizing_only_ || graph_failed_ || !applied_ ||
        !owner_->finish(transaction_, logits, count, position)) {
        abort();
        return false;
    }
    completed_   = true;
    transaction_ = 0;
    return true;
}

void llama_memory_slang_paged_context::abort() {
    if (!completed_ && transaction_ != 0) {
        owner_->abort(transaction_);
        transaction_ = 0;
    }
    completed_ = true;
}

llama_slang_paged_requests & llama_memory_slang_paged_context::requests() const {
    return owner_->requests();
}

llama_slang_paged_requests::transaction_handle llama_memory_slang_paged_context::transaction() const {
    return transaction_;
}

llama_slang_paged_attention & llama_memory_slang_paged_context::attention() const {
    return owner_->attention();
}

bool llama_memory_slang_paged_context::latch_graph_failure() {
    graph_failed_ = true;
    return false;
}

bool llama_memory_slang_paged_context::graph_failed() const {
    return graph_failed_;
}

const int64_t * llama_memory_slang_paged_context::positions() const {
    return positions_.data();
}

float * llama_memory_slang_paged_context::scratch() {
    return scratch_.data();
}

size_t llama_memory_slang_paged_context::scratch_elements() const {
    return scratch_.size();
}

std::unique_ptr<llama_memory_slang_paged_cache> llama_memory_slang_paged_cache::create(
    const llama_model &              model,
    const llama_memory_params &      memory,
    const llama_cparams &            cparams,
    const llama_slang_paged_params & paged) {
    const int32_t vocabulary_width = model.vocab.n_tokens();
    if (vocabulary_width <= 0 || size_t(vocabulary_width) > std::numeric_limits<size_t>::max() / sizeof(float) / 2) {
        return nullptr;
    }
    return create_impl(model, memory, cparams, paged, 1, paged.page_capacity,
                       paged.descriptor_byte_limit, size_t(vocabulary_width) * sizeof(float) * 2, false);
}

std::unique_ptr<llama_memory_slang_paged_cache> llama_memory_slang_paged_cache::create_external(
        const llama_model & model,
        const llama_memory_params & memory,
        const llama_cparams & cparams,
        const llama_slang_paged_provider_params & provider) {
    if (provider.abi_version != LLAMA_SLANG_PAGED_PROVIDER_ABI_VERSION || provider.max_requests == 0 ||
        provider.max_pages_per_request == 0 || provider.logits_byte_limit == 0) {
        return nullptr;
    }
    const llama_slang_paged_params paged = {
        provider.page_tokens, provider.page_capacity, provider.byte_limit, provider.descriptor_byte_limit,
    };
    return create_impl(model, memory, cparams, paged, provider.max_requests, provider.max_pages_per_request,
                       provider.metadata_byte_limit, provider.logits_byte_limit, true);
}

std::unique_ptr<llama_memory_slang_paged_cache> llama_memory_slang_paged_cache::create_impl(
    const llama_model &              model,
    const llama_memory_params &      memory,
    const llama_cparams &            cparams,
    const llama_slang_paged_params & paged,
    uint32_t                         max_requests,
    uint32_t                         max_pages_per_request,
    size_t                           metadata_byte_limit,
    size_t                           logits_byte_limit,
    bool                             external_mode) {
#if !defined(LLAMA_SLANG_PAGED_CPU_DIRECT)
    GGML_UNUSED(model);
    GGML_UNUSED(memory);
    GGML_UNUSED(cparams);
    GGML_UNUSED(paged);
    GGML_UNUSED(max_requests);
    GGML_UNUSED(max_pages_per_request);
    GGML_UNUSED(metadata_byte_limit);
    GGML_UNUSED(logits_byte_limit);
    GGML_UNUSED(external_mode);
    return nullptr;
#else
    const auto & hp           = model.hparams;
    bool         cpu_resident = ggml_backend_dev_type(model.dev_output()) == GGML_BACKEND_DEVICE_TYPE_CPU;
    for (uint32_t il = 0; il < hp.n_layer() && cpu_resident; ++il) {
        cpu_resident = ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    if (model.arch != LLM_ARCH_LLAMA || memory.type_k != GGML_TYPE_F32 || memory.type_v != GGML_TYPE_F32 ||
        cparams.flash_attn || cparams.offload_kqv || !cparams.causal_attn || cparams.embeddings ||
        cparams.n_seq_max != 1 || memory.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || hp.is_swa_any() ||
        hp.n_layer() == 0 || hp.n_head() == 0 || hp.n_head_kv() == 0 || hp.n_head() % hp.n_head_kv() != 0 ||
        hp.n_embd_head_k() == 0 || hp.n_embd_head_k() != hp.n_embd_head_v() || hp.is_n_embd_k_gqa_variable() ||
        hp.is_n_embd_v_gqa_variable() || paged.page_tokens == 0 || paged.page_capacity == 0 || paged.byte_limit == 0 ||
        paged.descriptor_byte_limit == 0 || model.vocab.n_tokens() <= 0 || !cpu_resident) {
        return nullptr;
    }
    const uint64_t                 execution_namespace = make_namespace(model, paged);
    llama_slang_paged_pool::layout layout              = {
        hp.n_layer(),        paged.page_tokens,   hp.n_head_kv(),   hp.n_embd_head_k(),
        paged.page_capacity, execution_namespace, paged.byte_limit, paged.descriptor_byte_limit,
    };
    auto unique_pool = llama_slang_paged_pool::create(layout);
    if (!unique_pool) {
        return nullptr;
    }
    std::shared_ptr<llama_slang_paged_pool> pool(std::move(unique_pool));
    const uint32_t                          vocabulary_width = uint32_t(model.vocab.n_tokens());
    llama_slang_paged_requests::limits      limits           = {
        max_requests,
        max_pages_per_request,
        vocabulary_width,
        metadata_byte_limit,
        logits_byte_limit,
    };
    auto                        requests = llama_slang_paged_requests::create(pool, limits);
    llama_slang_paged_attention attention({
        hp.n_layer(),
        hp.n_head(),
        hp.n_head_kv(),
        hp.n_embd_head_k(),
        hp.f_attention_scale == 0.0f ? 1.0f / std::sqrt(float(hp.n_embd_head_k())) : hp.f_attention_scale,
    });
    if (!requests || !attention.valid()) {
        return nullptr;
    }
    auto result = std::unique_ptr<llama_memory_slang_paged_cache>(new llama_memory_slang_paged_cache(
        std::move(pool), std::move(requests), attention, vocabulary_width, paged.page_tokens, execution_namespace,
        cparams.n_ctx, external_mode));
    return external_mode || result->reset_request() ? std::move(result) : nullptr;
#endif
}

llama_memory_slang_paged_cache::llama_memory_slang_paged_cache(std::shared_ptr<llama_slang_paged_pool>     pool,
                                                               std::unique_ptr<llama_slang_paged_requests> requests,
                                                               llama_slang_paged_attention                 attention,
                                                               uint32_t vocabulary_width,
                                                               uint32_t page_tokens,
                                                               uint64_t execution_namespace,
                                                               uint32_t max_context_tokens,
                                                               bool external_mode) :
    pool_(std::move(pool)),
    requests_(std::move(requests)),
    attention_(attention),
    vocabulary_width_(vocabulary_width),
    page_tokens_(page_tokens),
    execution_namespace_(execution_namespace),
    max_context_tokens_(max_context_tokens),
    external_mode_(external_mode) {}

llama_memory_slang_paged_cache::~llama_memory_slang_paged_cache() {
    if (request_ != 0) {
        requests_->discard_request(request_);
    }
}

bool llama_memory_slang_paged_cache::reset_request() {
    if (request_ != 0 && !requests_->discard_request(request_)) {
        return false;
    }
    request_ = requests_->open_request(execution_namespace_);
    return request_ != 0;
}

llama_memory_context_ptr llama_memory_slang_paged_cache::init_batch(llama_batch_allocr & balloc,
                                                                    uint32_t             n_ubatch,
                                                                    bool                 embd_all) {
    auto reject = [&]() {
        if (external_mode_ && bound_transaction_ != 0) {
            fail_external(bound_transaction_);
        }
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    };
    if (failed_ || embd_all || balloc.get_n_tokens() == 0 || balloc.get_n_tokens() > n_ubatch ||
        (external_mode_ && bound_transaction_ == 0)) {
        return reject();
    }
    balloc.split_reset();
    llama_ubatch ubatch  = balloc.split_simple(n_ubatch);
    llama_ubatch extra   = balloc.split_simple(n_ubatch);
    uint32_t     outputs = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        outputs += ubatch.output[i] != 0;
    }
    const uint64_t active_request = external_mode_ ? requests_->transaction_request(bound_transaction_) : request_;
    const uint64_t cursor = external_mode_ ? requests_->transaction_append_position(bound_transaction_) :
                                             requests_->cursor(active_request);
    const bool position_overflow = ubatch.n_tokens > std::numeric_limits<uint64_t>::max() - cursor;
    if (ubatch.n_tokens != balloc.get_n_tokens() || extra.n_tokens != 0 || ubatch.n_seqs_unq != 1 ||
        ubatch.seq_id_unq[0] != 0 || ubatch.n_pos != 1 || outputs != 1 || !ubatch.output[ubatch.n_tokens - 1] ||
        cursor == std::numeric_limits<uint64_t>::max() || position_overflow ||
        ubatch.pos[0] < 0 || uint64_t(ubatch.pos[0]) != cursor) {
        return reject();
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != 0 || ubatch.pos[i] < 0 ||
            uint64_t(ubatch.pos[i]) != cursor + i) {
            return reject();
        }
    }
    if (external_mode_) {
        const uint64_t end = requests_->transaction_end_position(bound_transaction_);
        if (cursor == std::numeric_limits<uint64_t>::max() || end == std::numeric_limits<uint64_t>::max() ||
            end != cursor + ubatch.n_tokens || end > max_context_tokens_) {
            return reject();
        }
        try {
            return std::make_unique<llama_memory_slang_paged_context>(
                this, std::move(ubatch), bound_transaction_, false);
        } catch (const std::bad_alloc &) {
            return reject();
        } catch (const std::length_error &) {
            return reject();
        }
    }
    const uint64_t end = cursor + ubatch.n_tokens;
    if (end < cursor || end > uint64_t(std::numeric_limits<int32_t>::max()) + 1) {
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    const uint32_t expected_pages = uint32_t((end + page_tokens_ - 1) / page_tokens_);
    auto           transaction    = requests_->begin(request_, 0, expected_pages);
    if (transaction == 0) {
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    const auto *   published       = requests_->published_table(request_);
    const uint32_t published_count = requests_->published_count(request_);
    uint64_t       remaining       = ubatch.n_tokens;
    bool           ok              = true;
    for (uint32_t i = 0; i < expected_pages && ok; ++i) {
        const uint32_t old_rows    = i < published_count ? published[i].valid_rows : 0;
        const uint32_t append_rows = uint32_t(std::min<uint64_t>(remaining, page_tokens_ - old_rows));
        if (append_rows == 0) {
            ok = requests_->push(transaction, published[i].page, old_rows, 0);
        } else {
            auto page = pool_->reserve();
            if (page == 0 || (old_rows != 0 && !pool_->copy_rows(published[i].page, page, old_rows)) ||
                !requests_->push(transaction, page, old_rows, append_rows)) {
                if (page != 0 && pool_->state(page) == llama_slang_paged_pool::page_state::writable &&
                    pool_->exclusive_transaction(page) == 0) {
                    pool_->release(page);
                }
                ok = false;
            }
            remaining -= append_rows;
        }
    }
    if (!ok || remaining != 0) {
        requests_->discard(transaction);
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    try {
        return std::make_unique<llama_memory_slang_paged_context>(this, std::move(ubatch), transaction, false);
    } catch (const std::bad_alloc &) {
        requests_->discard(transaction);
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    } catch (const std::length_error &) {
        requests_->discard(transaction);
        return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
}

llama_memory_context_ptr llama_memory_slang_paged_cache::init_full() {
    return std::make_unique<llama_memory_slang_paged_context>(this, llama_ubatch{}, 0, true);
}

llama_memory_context_ptr llama_memory_slang_paged_cache::init_update(llama_context *, bool) {
    return std::make_unique<llama_memory_slang_paged_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_slang_paged_cache::get_can_shift() const {
    return false;
}

void llama_memory_slang_paged_cache::clear(bool) {
    if (external_mode_) {
        unsupported("clear");
        return;
    }
    failed_ = !reset_request();
    if (failed_) {
        LLAMA_LOG_ERROR("%s: paged cache clear failed\n", __func__);
    }
}

bool llama_memory_slang_paged_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (external_mode_) {
        return false;
    }
    return seq_id == 0 && p0 == 0 && p1 == -1 && reset_request();
}

void llama_memory_slang_paged_cache::unsupported(const char * operation) const {
    failed_ = true;
    LLAMA_LOG_ERROR("%s: Slang paged cache does not support %s\n", __func__, operation);
}

void llama_memory_slang_paged_cache::seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) {
    unsupported("seq_cp");
}

void llama_memory_slang_paged_cache::seq_keep(llama_seq_id seq_id) {
    if (seq_id != 0) {
        unsupported("seq_keep");
    }
}

void llama_memory_slang_paged_cache::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    unsupported("seq_add");
}

void llama_memory_slang_paged_cache::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    unsupported("seq_div");
}

llama_pos llama_memory_slang_paged_cache::seq_pos_min(llama_seq_id seq_id) const {
    return seq_id == 0 && requests_->cursor(request_) != 0 ? 0 : -1;
}

llama_pos llama_memory_slang_paged_cache::seq_pos_max(llama_seq_id seq_id) const {
    const uint64_t cursor = requests_->cursor(request_);
    return seq_id == 0 && cursor != 0 ? llama_pos(cursor - 1) : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_slang_paged_cache::memory_breakdown() const {
#if !defined(LLAMA_SLANG_PAGED_CPU_DIRECT)
    return {};
#else
    return {
        { ggml_backend_cpu_buffer_type(), pool_->allocated_bytes() }
    };
#endif
}

[[noreturn]] void llama_memory_slang_paged_cache::state_write(
        llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {
    reject_state_serialization();
}

[[noreturn]] void llama_memory_slang_paged_cache::state_read(
        llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
    reject_state_serialization();
}

llama_slang_paged_requests & llama_memory_slang_paged_cache::requests() {
    return *requests_;
}

llama_slang_paged_attention & llama_memory_slang_paged_cache::attention() {
    return attention_;
}

bool llama_memory_slang_paged_cache::external_mode() const {
    return external_mode_;
}

int64_t llama_memory_slang_paged_cache::request_open() {
    if (!external_mode_) {
        return -1;
    }
    const uint64_t request = requests_->open_request(execution_namespace_);
    return request == 0 || request > uint64_t(INT64_MAX) ? -1 : int64_t(request);
}

bool llama_memory_slang_paged_cache::request_close(uint64_t request) {
    if (!external_mode_ || request == 0 || bound_transaction_ != 0) {
        return false;
    }
    if (computed_transaction_ != 0 && requests_->transaction_request(computed_transaction_) == request) {
        if (!requests_->abort(computed_transaction_)) {
            return false;
        }
        computed_transaction_ = 0;
    }
    return requests_->close_request(request);
}

int64_t llama_memory_slang_paged_cache::page_reserve() {
    if (!external_mode_) {
        return -1;
    }
    const uint64_t page = pool_->reserve();
    return page == 0 || page > uint64_t(INT64_MAX) ? -1 : int64_t(page);
}

bool llama_memory_slang_paged_cache::page_release(uint64_t page) {
    return external_mode_ && page != 0 && pool_->release(page);
}

bool llama_memory_slang_paged_cache::page_seal(uint64_t page) {
    return external_mode_ && page != 0 && pool_->seal(page);
}

bool llama_memory_slang_paged_cache::page_copy_tail(uint64_t source, uint64_t destination, uint32_t rows) {
    return external_mode_ && source != 0 && destination != 0 && pool_->copy_rows(source, destination, rows);
}

int64_t llama_memory_slang_paged_cache::table_begin(
        uint64_t request, uint64_t base_position, uint32_t expected_pages) {
    if (!external_mode_ || request == 0 || bound_transaction_ != 0 || computed_transaction_ != 0) {
        return -1;
    }
    const uint64_t transaction = requests_->begin(request, base_position, expected_pages);
    return transaction == 0 || transaction > uint64_t(INT64_MAX) ? -1 : int64_t(transaction);
}

bool llama_memory_slang_paged_cache::table_push(
        uint64_t transaction, uint64_t page, uint32_t valid_rows, uint32_t additional_rows) {
    if (!external_mode_ || transaction == 0 || bound_transaction_ != 0 || computed_transaction_ != 0) {
        return false;
    }
    if (page == 0 || !requests_->push(transaction, page, valid_rows, additional_rows)) {
        fail_external(transaction);
        return false;
    }
    return true;
}

bool llama_memory_slang_paged_cache::bind_transaction(uint64_t transaction) {
    if (!external_mode_ || transaction == 0 || bound_transaction_ != 0 || computed_transaction_ != 0 ||
        requests_->transaction_request(transaction) == 0 || requests_->transaction_failed(transaction)) {
        return false;
    }
    bound_transaction_ = transaction;
    return true;
}

void llama_memory_slang_paged_cache::unbind_transaction(uint64_t transaction) {
    if (bound_transaction_ == transaction) {
        bound_transaction_ = 0;
    }
}

bool llama_memory_slang_paged_cache::fail_external(uint64_t transaction) {
    if (!external_mode_ || transaction == 0 || requests_->transaction_request(transaction) == 0) {
        return false;
    }
    return requests_->transaction_failed(transaction) || requests_->fail(transaction);
}

bool llama_memory_slang_paged_cache::commit_external(uint64_t transaction) {
    if (!external_mode_ || bound_transaction_ != 0 || transaction == 0 ||
        requests_->transaction_request(transaction) == 0) {
        return false;
    }
    const bool computed = computed_transaction_ == transaction && !requests_->transaction_failed(transaction);
    bool       ok       = false;
    if (computed) {
        ok = requests_->commit(transaction, false);
    } else {
        fail_external(transaction);
        requests_->abort(transaction);
    }
    if (computed_transaction_ == transaction) {
        computed_transaction_ = 0;
    }
    return ok;
}

bool llama_memory_slang_paged_cache::commit_external_and_copy(
        uint64_t transaction, float * destination, size_t count) {
    if (!external_mode_ || transaction == 0 || requests_->transaction_request(transaction) == 0) {
        return false;
    }
    if (destination == nullptr || count != vocabulary_width_ || count > UINT32_MAX ||
        computed_transaction_ != transaction || requests_->transaction_failed(transaction)) {
        fail_external(transaction);
        (void)commit_external(transaction);
        return false;
    }
    const uint64_t request = requests_->transaction_request(transaction);
    if (!commit_external(transaction)) {
        return false;
    }
    const float * logits = requests_->logits(request);
    if (logits == nullptr) {
        std::abort();
    }
    std::memcpy(destination, logits, count * sizeof(float));
    return true;
}

bool llama_memory_slang_paged_cache::abort_external(uint64_t transaction) {
    if (!external_mode_ || transaction == 0 || bound_transaction_ == transaction) {
        return false;
    }
    const bool ok = requests_->abort(transaction);
    if (computed_transaction_ == transaction) {
        computed_transaction_ = 0;
    }
    return ok;
}

bool llama_memory_slang_paged_cache::logits_copy(uint64_t request, float * destination, size_t count) const {
    if (!external_mode_ || request == 0 || destination == nullptr || count != vocabulary_width_ ||
        !requests_->logits_valid(request)) {
        return false;
    }
    const float * logits = requests_->logits(request);
    if (logits == nullptr) {
        return false;
    }
    std::memcpy(destination, logits, count * sizeof(float));
    return true;
}

size_t llama_memory_slang_paged_cache::page_bytes() const {
    return external_mode_ ? pool_->page_bytes() : 0;
}

bool llama_memory_slang_paged_cache::finish(llama_slang_paged_requests::transaction_handle transaction,
                                            const float *                                  logits,
                                            uint32_t                                       count,
                                            llama_pos                                      position) {
    const uint32_t entries = requests_->execution_count(transaction);
    for (uint32_t i = 0; i < entries; ++i) {
        llama_slang_paged_requests::execution_entry entry{};
        if (!requests_->execution_entry_at(transaction, i, entry) ||
            (entry.writable_end > entry.readable_rows && !requests_->finish_rows(transaction, i, entry.writable_end))) {
            if (external_mode_) {
                fail_external(transaction);
            } else {
                requests_->discard(transaction);
            }
            return false;
        }
    }
    if (logits == nullptr || count != vocabulary_width_ || position < 0 ||
        !requests_->set_logits(transaction, uint64_t(position), logits, count)) {
        return false;
    }
    if (external_mode_) {
        computed_transaction_ = transaction;
        return true;
    }
    return requests_->commit(transaction, true);
}

void llama_memory_slang_paged_cache::abort(llama_slang_paged_requests::transaction_handle transaction) {
    if (transaction != 0) {
        requests_->fail(transaction);
        if (!external_mode_) {
            requests_->discard(transaction);
        }
    }
}
