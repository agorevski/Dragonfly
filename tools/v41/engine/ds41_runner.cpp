// Adapted from Xolotl's layer-sharded Ds41MoeRunner; private to the DS41 engine.
#include "ds41_runner.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#ifdef DS41_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace strata::v41 {

bool ds41_expert_devices(const std::string & text, int visible,
                        std::vector<int> & devices, std::string & error) {
    devices.clear();
    error.clear();
    if (text == "auto") {
        for (int i = 0; i < visible; ++i) devices.push_back(i);
        return true;
    }
    std::vector<int> parsed;
    size_t begin = 0;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const size_t stop = end == std::string::npos ? text.size() : end;
        int id = -1;
        const auto result = std::from_chars(text.data() + begin, text.data() + stop, id);
        if (result.ec != std::errc() || result.ptr != text.data() + stop || id < 0 || id >= visible ||
            std::find(parsed.begin(), parsed.end(), id) != parsed.end()) {
            error = "--expert-gpus requires auto or distinct CUDA-visible device IDs";
            return false;
        }
        parsed.push_back(id);
        begin = stop + 1;
        if (begin == text.size()) {
            error = "--expert-gpus cannot end with a comma";
            return false;
        }
    }
    if (parsed.empty()) {
        error = "--expert-gpus needs a device selection";
        return false;
    }
    devices = std::move(parsed);
    return true;
}

#ifdef DS41_HAVE_CUDA
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("multi-GPU CUDA: ") + cudaGetErrorString(status));
}
struct Device {
    int previous = 0;
    explicit Device(int id) { check(cudaGetDevice(&previous)); check(cudaSetDevice(id)); }
    ~Device() {
        const auto status = cudaSetDevice(previous);
        if (status != cudaSuccess)
            std::fprintf(stderr, "multi-GPU context restore: %s\n", cudaGetErrorString(status));
    }
};
}
#endif

Ds41MoeRunner::~Ds41MoeRunner() { close(); }

bool Ds41MoeRunner::init(const Ds41MoeGeom & g, Ds41BlobSource * source, const Ds41MoeConfig & cfg,
                         const std::vector<int> & devices, double reserve_gib, std::string & error) {
    close();
    if (devices.empty()) return Ds41MoeTier::init(g, source, cfg, error);
    error.clear();
#ifndef DS41_HAVE_CUDA
    (void) reserve_gib;
    error = "multi-GPU expert execution requires a CUDA build";
    return false;
#else
    try {
        int count = 0;
        check(cudaGetDeviceCount(&count));
        auto unique = devices;
        std::sort(unique.begin(), unique.end());
        if (!source || devices.size() < 2 || unique.front() < 0 || unique.back() >= count ||
            std::adjacent_find(unique.begin(), unique.end()) != unique.end() ||
            cfg.cpu_only || cfg.pcie_frac != 1 || cfg.pf_b != 0 || g.per_expert() ||
            g.n_layers < int64_t(devices.size()) || !cfg.exclude.empty() ||
            !std::isfinite(reserve_gib) || reserve_gib < 1 ||
            !std::isfinite(cfg.arena_gib) || cfg.arena_gib < 0) {
            error = "multi-GPU requires distinct devices, uniform layer formats, --pcie 1, --pf-b 0, "
                    "no pruning, and --vram-margin >= 1; use --expert-gpus 0 for the single-GPU path";
            return false;
        }
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (!g.routed(l) || g.gate_type(l) != g.gu_type || g.up_type(l) != g.gu_type ||
                g.down_type(l) != g.d_type || source->blob_bytes(l) != source->blob_bytes()) {
                error = "multi-GPU requires the same gate/up/down formats and blob sizes on every layer";
                return false;
            }
        hidden_ = g.n_embd;
        owners_.resize(size_t(g.n_layers));
        for (size_t l = 0; l < owners_.size(); ++l) owners_[l] = l * devices.size() / owners_.size();
        const size_t stride = (size_t(source->blob_bytes()) + 255) / 256 * 256;
        const size_t reserve = size_t(reserve_gib * 1073741824.0);
        std::vector<int64_t> slots(devices.size());
        double tail_gib = 0;
        for (size_t i = 0; i < devices.size(); ++i) {
            Device device(devices[i]);
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total));
            const int64_t owned = std::count(owners_.begin(), owners_.end(), i) * g.n_experts;
            if (free <= reserve + stride) throw std::runtime_error("multi-GPU: insufficient free VRAM");
            slots[i] = std::min({cfg.slots, owned, int64_t((free - reserve) / stride)});
            tail_gib += (owned - slots[i]) * double(source->blob_bytes()) / 1073741824.0;
        }
        if (tail_gib > cfg.arena_gib || tail_gib > cfg.max_arena_gib) {
            error = "multi-GPU pinned tail needs " + std::to_string(tail_gib) +
                    " GiB total; increase --arena-gib or use --expert-gpus 0";
            owners_.clear();
            return false;
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            Device device(devices[i]);
            Ds41MoeGeom local = g;
            local.gate_types.assign(size_t(g.n_layers), -1);
            local.up_types.assign(size_t(g.n_layers), -1);
            local.down_types.assign(size_t(g.n_layers), -1);
            std::vector<std::pair<int32_t, int32_t>> ranked;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                if (owners_[size_t(l)] != i) continue;
                local.gate_types[size_t(l)] = g.gate_type(l);
                local.up_types[size_t(l)] = g.up_type(l);
                local.down_types[size_t(l)] = g.down_type(l);
                for (int64_t e = 0; e < g.n_experts; ++e) ranked.emplace_back(int32_t(l), int32_t(e));
            }
            Ds41MoeConfig c = cfg;
            c.slots = slots[i];
            c.slot_gib = 0;
            c.defer_cache = c.no_cache = false;
            c.arena_skip_resident = c.persistent_cache = true;
            c.arena_adapt = c.vram_lru = c.chunk_prestage = c.chunk_prefetch = false;
            c.threads = 1;
            c.arena_gib = c.max_arena_gib =
                (ranked.size() - size_t(c.slots)) * double(source->blob_bytes()) / 1073741824.0 + 0.001;
            Shard s;
            s.device = devices[i];
            s.tier = std::make_unique<Ds41MoeTier>();
            shards_.push_back(std::move(s));
            if (!shards_.back().tier->init(local, source, c, error) ||
                !shards_.back().tier->seed_from_ranked(ranked, error)) {
                close();
                return false;
            }
            const auto & tier = *shards_.back().tier;
            if (tier.file_tier() != 0 || tier.resident() + tier.arena_experts() != int64_t(ranked.size()))
                throw std::runtime_error("multi-GPU shard did not cover every owned expert in VRAM or pinned RAM");
            std::fprintf(stderr, "multi-GPU CUDA%d: %zu owned experts, %lld resident, %lld pinned, 0 file\n",
                         devices[i], ranked.size(), (long long) tier.resident(), (long long) tier.arena_experts());
        }
        return true;
    } catch (const std::runtime_error & e) {
        error = e.what();
        close();
        return false;
    }
#endif
}

bool Ds41MoeRunner::dispatch(int64_t layer, const std::function<bool(Shard &)> & operation) {
#ifdef DS41_HAVE_CUDA
    if (layer < 0 || size_t(layer) >= owners_.size()) {
        std::fprintf(stderr, "multi-GPU: invalid layer %lld\n", (long long) layer);
        return false;
    }
    try {
        auto & shard = shards_[owners_[size_t(layer)]];
        Device device(shard.device);
        return operation(shard);
    } catch (const std::runtime_error & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return false;
    }
#else
    (void) layer; (void) operation;
    return false;
#endif
}

bool Ds41MoeRunner::run(int64_t l, const int32_t * ids, const float * w, const float * x, float * out) {
    if (!sharded()) return Ds41MoeTier::run(l, ids, w, x, out);
    return dispatch(l, [&](Shard & s) { return s.tier->run(l, ids, w, x, out); });
}
bool Ds41MoeRunner::run_multi(int64_t l, int n, const int32_t * ids, const float * w, const float * x, float * out) {
    if (!sharded()) return Ds41MoeTier::run_multi(l, n, ids, w, x, out);
    return dispatch(l, [&](Shard & s) { return s.tier->run_multi(l, n, ids, w, x, out); });
}
bool Ds41MoeRunner::run_chunk(int64_t l, int n, const int32_t * ids, const float * w, const float * x, float * out) {
    if (!sharded()) return Ds41MoeTier::run_chunk(l, n, ids, w, x, out);
    return dispatch(l, [&](Shard & s) { return s.tier->run_chunk(l, n, ids, w, x, out); });
}
bool Ds41MoeRunner::run_to_dev(int64_t l, const int32_t * ids, const float * w, const float * x,
                               void * out, const void * x_dev) {
    if (!sharded()) return Ds41MoeTier::run_to_dev(l, ids, w, x, out, x_dev);
    return run_device(l, 1, ids, w, x, out, 0, x_dev);
}
bool Ds41MoeRunner::run_multi_to_dev(int64_t l, int n, const int32_t * ids, const float * w,
                                     const float * x, void * out) {
    if (!sharded()) return Ds41MoeTier::run_multi_to_dev(l, n, ids, w, x, out);
    if (n < 1 || n > kMaxMulti) return false;
    return run_device(l, n, ids, w, x, out, 1);
}
bool Ds41MoeRunner::run_chunk_dev(int64_t l, int n, const int32_t * ids, const float * w,
                                 const float * x, void * out) {
    if (!sharded()) return Ds41MoeTier::run_chunk_dev(l, n, ids, w, x, out);
    return run_device(l, n, ids, w, x, out, 2);
}
bool Ds41MoeRunner::run_device(int64_t l, int n, const int32_t * ids, const float * w,
                              const float * x, void * out, int mode, const void * x_dev) {
#ifdef DS41_HAVE_CUDA
    if (n < 1 || !out || !ids || !w || (!x && !x_dev)) return false;
    int caller = 0;
    try { check(cudaGetDevice(&caller)); }
    catch (const std::runtime_error & e) { std::fprintf(stderr, "%s\n", e.what()); return false; }
    // Device input belongs to the dense caller, not the remote expert owner.
    std::vector<float> host;
    if (x_dev) {
        host.resize(size_t(n * hidden_));
        try { check(cudaMemcpy(host.data(), x_dev, host.size() * sizeof(float), cudaMemcpyDeviceToHost)); }
        catch (const std::runtime_error & e) { std::fprintf(stderr, "%s\n", e.what()); return false; }
        x = host.data();
    }
    return dispatch(l, [&](Shard & s) {
        const size_t bytes = size_t(n * hidden_) * sizeof(float);
        void * dst = out;
        if (s.device != caller) {
            if (s.output_bytes < bytes) {
                if (s.output) check(cudaFree(s.output));
                s.output = nullptr;
                s.output_bytes = 0;
                check(cudaMalloc(&s.output, bytes));
                s.output_bytes = bytes;
            }
            dst = s.output;
        }
        const bool ok = mode == 0 ? s.tier->run_to_dev(l, ids, w, x, dst) :
                        mode == 1 ? s.tier->run_multi_to_dev(l, n, ids, w, x, dst) :
                                    s.tier->run_chunk_dev(l, n, ids, w, x, dst);
        if (ok && s.device != caller) {
            check(cudaMemcpyPeerAsync(out, caller, dst, s.device, bytes, nullptr));
            check(cudaStreamSynchronize(nullptr)); // Dense consumes on a different nonblocking stream.
        }
        return ok;
    });
#else
    (void) l; (void) n; (void) ids; (void) w; (void) x; (void) out; (void) mode; (void) x_dev;
    return false;
#endif
}

bool Ds41MoeRunner::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>> & ranked, std::string & error) {
    return sharded() || Ds41MoeTier::seed_from_ranked(ranked, error);
}
bool Ds41MoeRunner::build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>> & ranked, std::string & error) {
    return sharded() || Ds41MoeTier::build_arena_ranked(ranked, error);
}
void Ds41MoeRunner::release_cache() {
    if (!sharded()) Ds41MoeTier::release_cache();
}
void Ds41MoeRunner::release_chunk() {
    if (!sharded()) { Ds41MoeTier::release_chunk(); return; }
    for (size_t l = 0; l < owners_.size(); ++l)
        if (l == 0 || owners_[l] != owners_[l-1])
            dispatch(int64_t(l), [](Shard & s) { s.tier->release_chunk(); return true; });
}
void Ds41MoeRunner::close() {
#ifdef DS41_HAVE_CUDA
    for (auto & s : shards_) {
        try {
            Device device(s.device);
            s.tier.reset();
            if (s.output) check(cudaFree(s.output));
        } catch (const std::runtime_error & e) { std::fprintf(stderr, "%s\n", e.what()); }
    }
#endif
    shards_.clear();
    owners_.clear();
    Ds41MoeTier::close();
}
const Ds41MoeStats & Ds41MoeRunner::stats() const {
    if (!sharded()) return Ds41MoeTier::stats();
    aggregate_ = {};
    for (const auto & s : shards_) aggregate_.add(s.tier->stats());
    return aggregate_;
}
void Ds41MoeRunner::reset_stats() {
    if (!sharded()) { Ds41MoeTier::reset_stats(); return; }
    for (auto & s : shards_) s.tier->reset_stats();
}
const char * Ds41MoeRunner::mode() const { return sharded() ? "gpu-layer-sharded" : Ds41MoeTier::mode(); }
int64_t Ds41MoeRunner::resident() const {
    if (!sharded()) return Ds41MoeTier::resident();
    int64_t n = 0; for (const auto & s : shards_) n += s.tier->resident(); return n;
}
int64_t Ds41MoeRunner::cache_slots() const {
    if (!sharded()) return Ds41MoeTier::cache_slots();
    int64_t n = 0; for (const auto & s : shards_) n += s.tier->cache_slots(); return n;
}
bool Ds41MoeRunner::resident(int64_t l, int64_t e) const {
    return sharded() ? shards_.at(owners_.at(size_t(l))).tier->resident(l, e) : Ds41MoeTier::resident(l, e);
}
bool Ds41MoeRunner::in_arena(int64_t l, int64_t e) const {
    return sharded() ? shards_.at(owners_.at(size_t(l))).tier->in_arena(l, e) : Ds41MoeTier::in_arena(l, e);
}
int64_t Ds41MoeRunner::arena_experts() const {
    if (!sharded()) return Ds41MoeTier::arena_experts();
    int64_t n = 0; for (const auto & s : shards_) n += s.tier->arena_experts(); return n;
}
int64_t Ds41MoeRunner::file_tier() const {
    if (!sharded()) return Ds41MoeTier::file_tier();
    int64_t n = 0; for (const auto & s : shards_) n += s.tier->file_tier(); return n;
}
double Ds41MoeRunner::arena_gib() const {
    if (!sharded()) return Ds41MoeTier::arena_gib();
    double n = 0; for (const auto & s : shards_) n += s.tier->arena_gib(); return n;
}

} // namespace strata::v41
