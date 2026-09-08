#pragma once

#include "llama-memory-slang-paged.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Bounded request-table transactions over Slang's physical KV page pool.
// This owner is scheduler-serialized; it does not provide internal locking.
class llama_slang_paged_requests {
  public:
    using request_handle     = uint64_t;
    using transaction_handle = uint64_t;
    using page_handle        = llama_slang_paged_pool::page_handle;

    struct limits {
        uint32_t max_requests;
        uint32_t max_pages_per_request;
        uint32_t vocabulary_width;
        size_t   metadata_byte_limit;
        size_t   logits_byte_limit;
    };

    struct published_entry {
        page_handle page;
        uint32_t    valid_rows;
        uint64_t    position_base;
    };

    struct execution_entry {
        page_handle page;
        uint32_t    readable_rows;
        uint32_t    writable_end;
        uint64_t    position_base;
        bool        exclusive;
    };

    static std::unique_ptr<llama_slang_paged_requests> create(std::shared_ptr<llama_slang_paged_pool> pool,
                                                              const limits &                          config);

    ~llama_slang_paged_requests();
    llama_slang_paged_requests(const llama_slang_paged_requests &)             = delete;
    llama_slang_paged_requests & operator=(const llama_slang_paged_requests &) = delete;

    request_handle open_request(uint64_t execution_namespace);
    bool           close_request(request_handle request);
    bool           cancel_request(request_handle request);

    transaction_handle begin(request_handle request, uint64_t table_base, uint32_t expected_pages);
    bool    push(transaction_handle transaction, page_handle page, uint32_t valid_rows, uint32_t writable_capacity);
    float * writable_row(transaction_handle transaction, uint32_t entry, bool value, uint32_t layer, uint32_t row);
    bool    mark_layer_written(transaction_handle transaction, uint32_t entry, uint32_t layer);
    bool    finish_rows(transaction_handle transaction, uint32_t entry, uint32_t new_valid);
    bool    set_logits(transaction_handle transaction, uint64_t position, const float * logits, uint32_t count);
    bool    fail(transaction_handle transaction);
    bool    commit(transaction_handle transaction);
    bool    abort(transaction_handle transaction);

    uint32_t                published_count(request_handle request) const;
    const published_entry * published_table(request_handle request) const;
    uint64_t                cursor(request_handle request) const;
    const float *           logits(request_handle request) const;
    bool                    logits_valid(request_handle request) const;
    uint32_t                execution_count(transaction_handle transaction) const;
    bool                    execution_compatible(transaction_handle transaction,
                                                 uint32_t           n_layer,
                                                 uint32_t           n_head_kv,
                                                 uint32_t           head_dim) const;
    bool          execution_entry_at(transaction_handle transaction, uint32_t index, execution_entry & result) const;
    bool          layer_unwritten(transaction_handle transaction, uint32_t entry, uint32_t layer) const;
    const float * readable_row(transaction_handle transaction,
                               uint32_t           entry,
                               bool               value,
                               uint32_t           layer,
                               uint32_t           row) const;

  private:
    struct staged_entry {
        page_handle page         = 0;
        uint32_t    initial_rows = 0;
        uint32_t    valid_rows   = 0;
        uint32_t    writable_end = 0;
        bool        exclusive    = false;
    };

    struct request_record {
        request_handle identity            = 0;
        uint64_t       execution_namespace = 0;
        uint32_t       published_count     = 0;
        uint64_t       cursor              = 0;
        bool           active              = false;
        bool           logits_valid        = false;
    };

    struct transaction_record {
        transaction_handle identity        = 0;
        request_handle     request         = 0;
        uint64_t           table_base      = 0;
        uint32_t           expected_pages  = 0;
        uint32_t           staged_count    = 0;
        uint64_t           logits_position = 0;
        bool               active          = false;
        bool               failed          = false;
        bool               logits_valid    = false;
    };

    llama_slang_paged_requests(std::shared_ptr<llama_slang_paged_pool> pool, const limits & config);

    request_record *           find_request(request_handle request);
    const request_record *     find_request(request_handle request) const;
    transaction_record *       find_transaction(transaction_handle transaction);
    const transaction_record * find_transaction(transaction_handle transaction) const;
    size_t                     request_index(const request_record & request) const;
    published_entry *          published_slice(size_t request_index);
    const published_entry *    published_slice(size_t request_index) const;
    staged_entry *             staged_slice(size_t request_index);
    float *                    published_logits_slice(size_t request_index);
    const float *              published_logits_slice(size_t request_index) const;
    float *                    staged_logits_slice(size_t request_index);
    uint8_t *                  layer_progress_slice(size_t request_index, uint32_t entry);
    const uint8_t *            layer_progress_slice(size_t request_index, uint32_t entry) const;
    bool                       fail_valid(transaction_record & transaction);
    void                       clear_transaction(size_t index);
    bool                       abort_record(size_t index);

    std::shared_ptr<llama_slang_paged_pool> pool_;
    limits                                  config_;
    std::vector<request_record>             requests_;
    std::vector<transaction_record>         transactions_;
    std::vector<published_entry>            published_;
    std::vector<staged_entry>               staged_;
    std::vector<float>                      published_logits_;
    std::vector<float>                      staged_logits_;
    std::vector<uint8_t>                    layer_progress_;
};
