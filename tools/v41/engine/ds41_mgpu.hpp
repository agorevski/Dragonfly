#pragma once

#include "ds41_moe.hpp"

namespace strata::v41 {

// Experimental capacity-only expert runner, deliberately separate from ds41_generate.
// Build with -DSTRATA_DS41_MGPU=ON (default OFF); CUDA execution additionally
// requires -DSTRATA_DS41_CUDA=ON and the ds41_mgpu_cuda target.
// Uniform standard Q2_K or Q3_K, all experts resident, >=2 CUDA devices. No CPU/file
// misses, mixed formats, prompt batches, device-pointer API, graphs or concurrent calls.
// Dense/attention/KV ownership is NOT handled here: this is not a full-model loader.
struct Ds41MultiGpuConfig {
    std::vector<int> devices;                 // CUDA-visible ordinals, not physical GPU IDs
    std::vector<size_t> device_bytes;         // hard cap per device: weights AND explicit workspace
    std::vector<size_t> layer_owner;          // device-vector index per layer; empty = contiguous balance
    size_t free_reserve_bytes = size_t(256) << 20; // retain this much currently free VRAM on EACH device
};

struct Ds41MultiGpuPlan {
    std::vector<size_t> layer_owner;
    std::vector<size_t> owned_layers;
    std::vector<size_t> cache_bytes;
    size_t expert_bytes = 0;
    size_t expert_stride = 0;
    size_t workspace_bytes = 0;               // per device; excludes CUDA driver/context overhead
    size_t pinned_bytes = 0;                  // per device; ONE blob + activation/parts/metadata
};

// Pure CPU validation/planning. visible_devices is supplied explicitly: no CUDA calls,
// model reads, host pinning or allocations proportional to model weights.
bool ds41_mgpu_plan(const Ds41MoeGeom & geom, const Ds41MultiGpuConfig & config,
                    int visible_devices, Ds41MultiGpuPlan & plan, std::string & error);

class Ds41MultiGpuMoe {
public:
    Ds41MultiGpuMoe();
    ~Ds41MultiGpuMoe();
    Ds41MultiGpuMoe(const Ds41MultiGpuMoe &) = delete;
    Ds41MultiGpuMoe & operator=(const Ds41MultiGpuMoe &) = delete;

    // Blocking transactional initialization: loads only this runner's experts, one
    // pinned blob at a time. Source need not outlive init. Failure leaves it closed.
    // Device context is restored on every path. Allocation errors/OOM return false.
    // Reused native grouped kernels retain their upstream fatal launch-error checks;
    // stream completion errors return false and invalidate this runner.
    bool init(const Ds41MoeGeom & geom, Ds41BlobSource & source,
              const Ds41MultiGpuConfig & config, std::string & error);
    // Blocking single-token decode. ALL pointers are host pointers. Completed owner
    // stream before return; caller may consume output on any device afterwards.
    // Host staging deliberately needs no peer access, NVLink, default-stream
    // ordering or cross-device events (including PCIe NODE device pairs).
    bool run(int64_t layer, const int32_t * ids, const float * weights,
             const float * x, float * out, std::string & error);
    void close() noexcept;
    bool ready() const;
    const Ds41MultiGpuPlan * plan() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::v41
