#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Slang's CPU physical KV storage substrate. This pool owns real, aligned F32
// K/V tensors. It deliberately does not implement llama_memory_i yet and is
// therefore unable to advertise or select paged execution by itself.
class llama_slang_paged_pool {
  public:
    using page_handle = uint64_t;

    enum class page_state : uint8_t {
        free,
        writable,
        sealed,
    };

    struct layout {
        uint32_t n_layer;
        uint32_t page_tokens;
        uint32_t n_head_kv;
        uint32_t head_dim;
        uint32_t page_capacity;
        uint64_t execution_namespace;
        size_t   byte_limit;
        size_t   descriptor_byte_limit;
    };

    static std::unique_ptr<llama_slang_paged_pool> create(const layout & config);

    ~llama_slang_paged_pool();
    llama_slang_paged_pool(const llama_slang_paged_pool &)             = delete;
    llama_slang_paged_pool & operator=(const llama_slang_paged_pool &) = delete;

    page_handle reserve();
    bool        release(page_handle handle);
    bool        set_occupied(page_handle handle, uint32_t rows);
    bool        seal(page_handle handle);
    bool        copy_rows(page_handle source, page_handle destination, uint32_t rows);
    bool        bind_position_exclusive(page_handle handle, uint64_t transaction, uint64_t position_base);

    // The future request registry owns exact sealed-page membership; this pool
    // enforces aggregate references and names the sole writable-page owner.
    bool map_for_request(page_handle handle, uint64_t request);
    bool unmap_for_request(page_handle handle, uint64_t request);
    bool borrow_sealed(page_handle handle);
    bool return_sealed(page_handle handle);
    // Exclusive staging is accepted only on an unpublished writable page.
    // Published pages require a distinct copy-on-write destination.
    bool claim_exclusive(page_handle handle, uint64_t transaction, uint64_t request);
    bool abort_exclusive(page_handle handle, uint64_t transaction);
    bool publish_exclusive(page_handle handle, uint64_t transaction, uint64_t request);
    bool set_occupied_exclusive(page_handle handle, uint64_t transaction, uint32_t rows);
    bool seal_exclusive(page_handle handle, uint64_t transaction);

    float * writable_row(page_handle handle, bool value, uint32_t layer, uint32_t row);
    float * writable_row_exclusive(page_handle handle, uint64_t transaction, bool value, uint32_t layer, uint32_t row);
    const float * row(page_handle handle, bool value, uint32_t layer, uint32_t row) const;

    page_state     state(page_handle handle) const;
    uint32_t       occupied(page_handle handle) const;
    uint32_t       mapping_references(page_handle handle) const;
    uint32_t       read_borrows(page_handle handle) const;
    uint64_t       exclusive_transaction(page_handle handle) const;
    bool           position_bound(page_handle handle) const;
    uint64_t       position_base(page_handle handle) const;
    size_t         page_bytes() const;
    size_t         allocated_bytes() const;
    const layout & config() const;

#ifdef LLAMA_SLANG_PAGED_TESTING
    // Compiled only into the dedicated storage test, never into libllama.
    static void test_set_next_identity(uint64_t next);
#endif

  private:
    struct page {
        float *     data                             = nullptr;
        page_handle identity                         = 0;
        uint32_t    occupied                         = 0;
        uint32_t    mapping_references               = 0;
        uint32_t    read_borrows                     = 0;
        uint64_t    writable_owner                   = 0;
        uint64_t    exclusive_transaction            = 0;
        uint64_t    exclusive_request                = 0;
        uint32_t    exclusive_initial_occupied       = 0;
        uint64_t    position_base                    = 0;
        uint64_t    exclusive_initial_position       = 0;
        bool        position_bound                   = false;
        bool        exclusive_initial_position_bound = false;
        page_state  state                            = page_state::free;
    };

    explicit llama_slang_paged_pool(const layout & config, size_t page_bytes);

    page *       find(page_handle handle);
    const page * find(page_handle handle) const;
    size_t       row_offset(bool value, uint32_t layer, uint32_t row) const;

    layout            config_;
    size_t            row_elements_;
    size_t            layer_elements_;
    size_t            plane_elements_;
    size_t            page_bytes_;
    size_t            allocated_bytes_ = 0;
    std::vector<page> pages_;
};
