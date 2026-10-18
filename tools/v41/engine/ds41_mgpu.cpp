#include "ds41_mgpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if defined(DS41_MGPU_CUDA)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#endif

namespace strata::v41 {
namespace {
struct Metadata {
    unsigned long long ptr[8] = {};
    int32_t start[9] = {};
    int32_t groups = 0;
    int32_t dst[8] = {};
    int32_t tok[8] = {};
};
size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
size_t scratch_bytes(const Ds41MoeGeom & g) {
    return 3 * align256(size_t(g.top_k * g.n_ff) * sizeof(float)) +
           align256(size_t(g.top_k * (g.n_ff / 32)) * 36);
}
}

bool ds41_mgpu_plan(const Ds41MoeGeom & g, const Ds41MultiGpuConfig & c,
                    int visible_devices, Ds41MultiGpuPlan & result, std::string & err) {
    result = {};
    err.clear();
    auto fail = [&](const char * message) { err = message; return false; };
    // Bounds also make all byte arithmetic safe on supported 64-bit hosts.
    if (sizeof(size_t) < 8 || g.n_layers < 2 || g.n_layers > 256 ||
        g.n_experts < 1 || g.n_experts > 1024 || g.top_k < 1 || g.top_k > 8 ||
        g.top_k > g.n_experts || g.n_embd < 256 || g.n_embd > 32768 ||
        g.n_ff < 256 || g.n_ff > 32768 || g.n_embd % 256 || g.n_ff % 256 ||
        std::isnan(g.swiglu_limit) || g.swiglu_limit <= 0)
        return fail("multi-GPU geometry exceeds supported bounds (256-aligned dimensions, top-k <= 8)");
    if (!g.gate_types.empty() || !g.up_types.empty() || !g.down_types.empty() ||
        !g.eclass.empty() || !g.cls_gu.empty() || !g.cls_d.empty() ||
        (g.gu_type != 10 && g.gu_type != 11) || g.d_type != g.gu_type)
        return fail("multi-GPU requires uniform standard Q2_K or Q3_K for gate/up/down");
    if (c.devices.size() < 2 || c.devices.size() > size_t(g.n_layers) ||
        c.device_bytes.size() != c.devices.size())
        return fail("multi-GPU needs >=2 devices, at least one layer each, and an explicit byte cap per device");
    auto devices = c.devices;
    std::sort(devices.begin(), devices.end());
    if (devices.front() < 0 || devices.back() >= visible_devices ||
        std::adjacent_find(devices.begin(), devices.end()) != devices.end())
        return fail("multi-GPU devices must be distinct valid CUDA-visible ordinals");
    Ds41MultiGpuPlan p;
    p.layer_owner = c.layer_owner;
    if (p.layer_owner.empty()) {
        p.layer_owner.resize(size_t(g.n_layers));
        for (size_t l = 0; l < p.layer_owner.size(); ++l)
            p.layer_owner[l] = l * c.devices.size() / p.layer_owner.size();
    }
    if (p.layer_owner.size() != size_t(g.n_layers))
        return fail("multi-GPU layer_owner must have one entry per layer");
    p.owned_layers.assign(c.devices.size(), 0);
    for (size_t owner : p.layer_owner) {
        if (owner >= c.devices.size()) return fail("multi-GPU invalid layer owner index");
        ++p.owned_layers[owner];
    }
    if (std::find(p.owned_layers.begin(), p.owned_layers.end(), 0) != p.owned_layers.end())
        return fail("multi-GPU every selected device must own at least one layer");
    const size_t block = g.gu_type == 10 ? 84 : 110;
    p.expert_bytes = 3 * size_t(g.n_embd) * size_t(g.n_ff / 256) * block;
    p.expert_stride = align256(p.expert_bytes);
    const size_t xbytes = size_t(g.n_embd) * sizeof(float);
    const size_t parts = size_t(g.top_k) * xbytes;
    p.workspace_bytes = xbytes + size_t(g.n_embd / 32) * 36 + parts +
                        scratch_bytes(g) + sizeof(Metadata);
    p.pinned_bytes = p.expert_bytes + xbytes + parts + sizeof(Metadata);
    for (size_t i = 0; i < c.devices.size(); ++i) {
        const size_t bytes = p.owned_layers[i] * size_t(g.n_experts) * p.expert_stride;
        if (c.device_bytes[i] < p.workspace_bytes ||
            bytes > c.device_bytes[i] - p.workspace_bytes)
            return fail("multi-GPU byte cap cannot hold all owned experts plus workspace; no CPU fallback");
        p.cache_bytes.push_back(bytes);
    }
    result = std::move(p);
    return true;
}

#if defined(DS41_MGPU_CUDA)
namespace {
void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
class Device {
public:
    explicit Device(int device) {
        check(cudaGetDevice(&previous_), "multi-GPU get device");
        check(cudaSetDevice(device), "multi-GPU set device");
    }
    void restore() {
        check(cudaSetDevice(previous_), "multi-GPU restore device");
        restored_ = true;
    }
    ~Device() {
        if (!restored_) {
            const auto status = cudaSetDevice(previous_);
            if (status != cudaSuccess)
                std::fprintf(stderr, "multi-GPU context restore: %s\n", cudaGetErrorString(status));
        }
    }
private:
    int previous_ = 0;
    bool restored_ = false;
};
struct Shard {
    int device;
    cudaStream_t stream = nullptr;
    void * cache = nullptr;
    void * x = nullptr;
    void * xq = nullptr;
    void * parts = nullptr;
    void * scratch = nullptr;
    Metadata * metadata = nullptr;
    void * host = nullptr;
    std::vector<size_t> layer_offset;
    explicit Shard(int id) : device(id) {}
    ~Shard() {
        // Partial construction and errors are cleaned up on the allocating device,
        // even if another shard failed or the caller changed its current device.
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
            std::fprintf(stderr, "multi-GPU cleanup cannot select CUDA%d\n", device);
            return;
        }
        auto release = [](cudaError_t status) {
            if (status != cudaSuccess)
                std::fprintf(stderr, "multi-GPU cleanup: %s\n", cudaGetErrorString(status));
        };
        if (stream) release(cudaStreamSynchronize(stream));
        for (void * p : {cache, x, xq, parts, scratch, static_cast<void *>(metadata)})
            if (p) release(cudaFree(p));
        if (host) release(cudaFreeHost(host));
        if (stream) release(cudaStreamDestroy(stream));
        release(cudaSetDevice(previous));
    }
};
}
#endif

struct Ds41MultiGpuMoe::Impl {
    Ds41MultiGpuPlan plan;
    Ds41MoeGeom geom;
#if defined(DS41_MGPU_CUDA)
    strata::kernels::NativeExpertLayout layout;
    std::vector<std::unique_ptr<Shard>> shards;
#endif
};

Ds41MultiGpuMoe::Ds41MultiGpuMoe() = default;
Ds41MultiGpuMoe::~Ds41MultiGpuMoe() { close(); }
void Ds41MultiGpuMoe::close() noexcept { impl_.reset(); }
bool Ds41MultiGpuMoe::ready() const { return bool(impl_); }
const Ds41MultiGpuPlan * Ds41MultiGpuMoe::plan() const { return impl_ ? &impl_->plan : nullptr; }

bool Ds41MultiGpuMoe::init(const Ds41MoeGeom & g, Ds41BlobSource & source,
                           const Ds41MultiGpuConfig & c, std::string & err) {
    close();
    err.clear();
#if !defined(DS41_MGPU_CUDA)
    (void) g; (void) source; (void) c;
    err = "multi-GPU execution requires the optional CUDA target; no CPU fallback";
    return false;
#else
    try {
        auto next = std::make_unique<Impl>();
        int count = 0;
        check(cudaGetDeviceCount(&count), "multi-GPU enumerate");
        if (!ds41_mgpu_plan(g, c, count, next->plan, err)) return false;
        next->geom = g;
        const auto & p = next->plan;
        next->layout = strata::kernels::native_expert_layout(g.gu_type, g.d_type, g.n_embd, g.n_ff);
        next->layout.swiglu_limit = g.swiglu_limit;
        if (!strata::kernels::native_expert_supported(g.gu_type, g.d_type, g.n_embd, g.n_ff) ||
            next->layout.bytes != p.expert_bytes ||
            strata::kernels::native_expert_scratch_bytes(g.top_k, g.n_ff) != scratch_bytes(g) ||
            strata::kernels::native_q8_1_bytes(int(g.n_embd)) != size_t(g.n_embd / 32) * 36)
            throw std::runtime_error("multi-GPU native kernel layout differs from plan");
        // Preflight every device before reading any model bytes or allocating weights.
        for (size_t i = 0; i < c.devices.size(); ++i) {
            Device device(c.devices[i]);
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total), "multi-GPU free VRAM");
            const size_t need = p.cache_bytes[i] + p.workspace_bytes;
            if (free < c.free_reserve_bytes || need > free - c.free_reserve_bytes)
                throw std::runtime_error("multi-GPU insufficient free VRAM on CUDA" + std::to_string(c.devices[i]));
            device.restore();
        }
        if (!source.open(err)) return false;
        if (source.blob_bytes() != p.expert_bytes)
            throw std::runtime_error("multi-GPU source blob size differs from plan");
        for (size_t i = 0; i < c.devices.size(); ++i) {
            Device device(c.devices[i]);
            next->shards.push_back(std::make_unique<Shard>(c.devices[i]));
            auto & s = *next->shards.back();
            s.layer_offset.resize(size_t(g.n_layers), size_t(-1));
            const size_t xbytes = size_t(g.n_embd) * sizeof(float);
            check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "multi-GPU stream");
            check(cudaMalloc(&s.cache, p.cache_bytes[i]), "multi-GPU resident cache");
            check(cudaMalloc(&s.x, xbytes), "multi-GPU activation");
            check(cudaMalloc(&s.xq, size_t(g.n_embd / 32) * 36), "multi-GPU quantized activation");
            check(cudaMalloc(&s.parts, size_t(g.top_k) * xbytes), "multi-GPU expert outputs");
            check(cudaMalloc(&s.scratch, scratch_bytes(g)), "multi-GPU scratch");
            check(cudaMalloc(reinterpret_cast<void **>(&s.metadata), sizeof(Metadata)), "multi-GPU metadata");
            check(cudaMallocHost(&s.host, p.pinned_bytes), "multi-GPU pinned staging");
            size_t offset = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                if (p.layer_owner[size_t(l)] != i) continue;
                if (source.blob_bytes(l) != p.expert_bytes)
                    throw std::runtime_error("multi-GPU source has nonuniform layer blobs");
                s.layer_offset[size_t(l)] = offset;
                for (int64_t e = 0; e < g.n_experts; ++e) {
                    source.read(l, e, static_cast<uint8_t *>(s.host));
                    check(cudaMemcpyAsync(static_cast<uint8_t *>(s.cache) + offset, s.host, p.expert_bytes,
                                          cudaMemcpyHostToDevice, s.stream), "multi-GPU seed");
                    check(cudaStreamSynchronize(s.stream), "multi-GPU seed completion");
                    offset += p.expert_stride;
                }
            }
            device.restore();
        }
        impl_ = std::move(next);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
#endif
}

bool Ds41MultiGpuMoe::run(int64_t layer, const int32_t * ids, const float * weights,
                          const float * x, float * out, std::string & err) {
    err.clear();
    if (!impl_ || layer < 0 || layer >= impl_->geom.n_layers || !ids || !weights || !x || !out) {
        err = "multi-GPU run needs an initialized runner, valid layer and host pointers";
        return false;
    }
#if !defined(DS41_MGPU_CUDA)
    return false;
#else
    const auto & g = impl_->geom;
    float routing_weights[8];
    for (int64_t k = 0; k < g.top_k; ++k) {
        if (ids[k] < 0 || ids[k] >= g.n_experts || !std::isfinite(weights[k])) {
            err = "multi-GPU invalid expert ID or nonfinite routing weight";
            return false;
        }
        routing_weights[k] = weights[k];
    }
    try {
        auto & s = *impl_->shards[impl_->plan.layer_owner[size_t(layer)]];
        Device device(s.device);
        const size_t xbytes = size_t(g.n_embd) * sizeof(float);
        auto * hx = static_cast<uint8_t *>(s.host) + impl_->plan.expert_bytes;
        auto * hp = reinterpret_cast<float *>(hx + xbytes);
        auto * hm = reinterpret_cast<Metadata *>(reinterpret_cast<uint8_t *>(hp) + size_t(g.top_k) * xbytes);
        Metadata metadata;
        metadata.groups = int32_t(g.top_k);
        for (int64_t k = 0; k < g.top_k; ++k) {
            metadata.ptr[k] = reinterpret_cast<unsigned long long>(s.cache) + s.layer_offset[size_t(layer)] +
                              size_t(ids[k]) * impl_->plan.expert_stride;
            metadata.start[k] = metadata.dst[k] = int32_t(k);
        }
        metadata.start[g.top_k] = int32_t(g.top_k);
        std::memcpy(hm, &metadata, sizeof(metadata));
        std::memcpy(hx, x, xbytes);
        check(cudaMemcpyAsync(s.x, hx, xbytes, cudaMemcpyHostToDevice, s.stream), "multi-GPU input");
        check(cudaMemcpyAsync(s.metadata, hm, sizeof(metadata), cudaMemcpyHostToDevice, s.stream), "multi-GPU routing");
        strata::kernels::native_quantize_q8_1(static_cast<const float *>(s.x), s.xq, int(g.n_embd), 1, s.stream);
        strata::kernels::native_expert_grouped(impl_->layout, s.metadata->ptr, s.metadata->start,
            &s.metadata->groups, s.metadata->dst, s.metadata->tok, g.top_k, g.top_k, s.xq, s.scratch,
            static_cast<float *>(s.parts), s.stream, g.top_k);
        check(cudaGetLastError(), "multi-GPU kernel launch");
        check(cudaMemcpyAsync(hp, s.parts, size_t(g.top_k) * xbytes, cudaMemcpyDeviceToHost, s.stream),
              "multi-GPU outputs");
        check(cudaStreamSynchronize(s.stream), "multi-GPU run completion");
        for (int64_t j = 0; j < g.n_embd; ++j) {
            double value = 0;
            for (int64_t k = 0; k < g.top_k; ++k)
                if (routing_weights[k] != 0) value += double(routing_weights[k]) * hp[k * g.n_embd + j];
            out[j] = float(value);
        }
        device.restore();
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        // Do not reuse pinned workspaces after a failed asynchronous operation.
        close();
        return false;
    }
#endif
}

} // namespace strata::v41
