#pragma once

#include "llama-memory.h"
#include "llama-slang-paged-attention.h"
#include "llama-slang-paged.h"

#include <memory>
#include <vector>

struct llama_cparams;
struct llama_model;
class llama_batch_allocr;

class llama_memory_slang_paged_cache;

class llama_memory_slang_paged_context final : public llama_memory_context_i {
  public:
    llama_memory_slang_paged_context(llama_memory_status status);
    llama_memory_slang_paged_context(llama_memory_slang_paged_cache *               owner,
                                     llama_ubatch                                   ubatch,
                                     llama_slang_paged_requests::transaction_handle transaction,
                                     bool                                           sizing_only);
    ~llama_memory_slang_paged_context() override;

    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;
    llama_memory_status  get_status() const override;
    bool                 requires_finalize() const override;
    bool                 finalize(const float * logits, uint32_t count, llama_pos position) override;
    void                 abort() override;

    llama_slang_paged_requests &                   requests() const;
    llama_slang_paged_requests::transaction_handle transaction() const;
    llama_slang_paged_attention &                  attention() const;
    bool                                           latch_graph_failure();
    bool                                           graph_failed() const;
    const int64_t *                                positions() const;
    float *                                        scratch();
    size_t                                         scratch_elements() const;

  private:
    llama_memory_status                            status_ = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    llama_memory_slang_paged_cache *               owner_  = nullptr;
    llama_ubatch                                   ubatch_{};
    llama_slang_paged_requests::transaction_handle transaction_  = 0;
    bool                                           sizing_only_  = false;
    bool                                           applied_      = false;
    bool                                           completed_    = false;
    bool                                           graph_failed_ = false;
    std::vector<int64_t>                           positions_;
    std::vector<float>                             scratch_;
};

class llama_memory_slang_paged_cache final : public llama_memory_i {
  public:
    static std::unique_ptr<llama_memory_slang_paged_cache> create(const llama_model &              model,
                                                                  const llama_memory_params &      memory,
                                                                  const llama_cparams &            cparams,
                                                                  const llama_slang_paged_params & paged);

    ~llama_memory_slang_paged_cache() override;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context *, bool) override;
    bool                     get_can_shift() const override;
    void                     clear(bool data) override;
    bool                     seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void                     seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override;
    void                     seq_keep(llama_seq_id) override;
    void                     seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override;
    void                     seq_div(llama_seq_id, llama_pos, llama_pos, int) override;
    llama_pos                seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos                seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    [[noreturn]] void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override;
    [[noreturn]] void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override;

    llama_slang_paged_requests &  requests();
    llama_slang_paged_attention & attention();
    bool                          finish(llama_slang_paged_requests::transaction_handle transaction,
                                         const float *                                  logits,
                                         uint32_t                                       count,
                                         llama_pos                                      position);
    void                          abort(llama_slang_paged_requests::transaction_handle transaction);

  private:
    llama_memory_slang_paged_cache(std::shared_ptr<llama_slang_paged_pool>     pool,
                                   std::unique_ptr<llama_slang_paged_requests> requests,
                                   llama_slang_paged_attention                 attention,
                                   uint32_t                                    vocabulary_width,
                                   uint32_t                                    page_tokens,
                                   uint64_t                                    execution_namespace);
    bool reset_request();
    void unsupported(const char * operation) const;

    std::shared_ptr<llama_slang_paged_pool>     pool_;
    std::unique_ptr<llama_slang_paged_requests> requests_;
    llama_slang_paged_attention                 attention_;
    llama_slang_paged_requests::request_handle  request_             = 0;
    uint32_t                                    vocabulary_width_    = 0;
    uint32_t                                    page_tokens_         = 0;
    uint64_t                                    execution_namespace_ = 0;
    mutable bool                                failed_              = false;
};
