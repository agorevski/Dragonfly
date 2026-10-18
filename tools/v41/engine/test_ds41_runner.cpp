#include "ds41_runner.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#ifdef DS41_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace strata::v41;

static void require(bool ok, const std::string & what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); std::exit(1); }
}

int main(int argc, char ** argv) {
    std::vector<int> devices;
    std::string error;
    require(ds41_expert_devices("auto", 4, devices, error) && devices == std::vector<int>({0,1,2,3}), "auto selects all GPUs");
    require(ds41_expert_devices("2,0", 4, devices, error) && devices == std::vector<int>({2,0}), "explicit selection order");
    for (const char * invalid : {"", "0,", ",0", "0,,1", "0,0", "-1", "4", "1x", "9999999999999999"})
        require(!ds41_expert_devices(invalid, 4, devices, error) && !error.empty() && devices.empty(),
                "reject invalid selection");
    if (argc == 1) return 0;
    require(argc == 2 && std::strcmp(argv[1], "--cuda") == 0, "usage: test_ds41_runner [--cuda]");
#ifndef DS41_HAVE_CUDA
    return 77;
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    // On four cards this includes both NVLink pairs and cross-pair NODE transfers.
    count = std::min(count, 4);
    require(cudaSetDevice(count - 1) == cudaSuccess, "select dense caller");
    Ds41MoeGeom g;
    g.n_embd = 512; g.n_ff = 256; g.n_layers = count; g.n_experts = 8; g.top_k = 2;
    g.gu_type = GGML_TYPE_Q2_K; g.d_type = GGML_TYPE_Q3_K; g.swiglu_limit = 10;
    strata::kernels::cpu::NativeFmt fmt;
    require(ds41_moe_blob_layout(g, fmt, error), error);
    Ds41MemoryBlobs source(g, fmt.bytes);
    std::mt19937 rng(37);
    std::normal_distribution<float> normal(0, 0.03f);
    std::vector<float> matrix(size_t(g.n_embd * g.n_ff));
    std::vector<uint8_t> blob(fmt.bytes);
    for (int l = 0; l < count; ++l)
        for (int e = 0; e < 8; ++e) {
            for (auto & v : matrix) v = normal(rng);
            ggml_quantize_chunk(GGML_TYPE_Q2_K, matrix.data(), blob.data(), 0, g.n_ff, g.n_embd, nullptr);
            for (auto & v : matrix) v = normal(rng);
            ggml_quantize_chunk(GGML_TYPE_Q2_K, matrix.data(), blob.data() + fmt.up_off, 0, g.n_ff, g.n_embd, nullptr);
            for (auto & v : matrix) v = normal(rng);
            ggml_quantize_chunk(GGML_TYPE_Q3_K, matrix.data(), blob.data() + fmt.down_off, 0, g.n_embd, g.n_ff, nullptr);
            source.set(l, e, blob.data());
        }
    Ds41MoeConfig cfg;
    cfg.slots = 6; cfg.arena_gib = cfg.max_arena_gib = 1;
    cfg.pcie_frac = 1; cfg.pf_b = 0; cfg.threads = 1;
    for (int i = 0; i < count; ++i) devices.push_back(i);
    Ds41MoeRunner sharded;
    require(!sharded.init(g, &source, cfg, {0,0}, 1, error) && !error.empty(), "duplicate device rejection");
    auto too_small = cfg;
    too_small.arena_gib = too_small.max_arena_gib = 0;
    require(!sharded.init(g, &source, too_small, devices, 1, error) && !sharded.sharded(),
            "tail capacity preflight is transactional");
    require(sharded.init(g, &source, cfg, devices, 1, error), error);
    require(sharded.resident() == 6 * count && sharded.arena_experts() == 2 * count &&
            sharded.file_tier() == 0, "disjoint resident and pinned coverage");
    sharded.reset_stats();
    Ds41MoeRunner reference;
    require(reference.init(g, &source, cfg, {}, 1, error), error);
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int l = 0; l < count; ++l) for (int e = 0; e < 8; ++e) ranked.emplace_back(l, e);
    require(reference.seed_from_ranked(ranked, error), error);
    constexpr int n = 8;
    std::vector<float> x(size_t(n * g.n_embd)), weights(n * g.top_k, 0.5f);
    std::vector<float> actual(x.size()), expected(x.size());
    std::vector<int32_t> ids(n * g.top_k);
    for (auto & v : x) v = normal(rng);
    for (int i = 0; i < n; ++i) { ids[2*i] = i % 4; ids[2*i+1] = 7; }
    float * output = nullptr;
    float * input = nullptr;
    float * landing = nullptr;
    cudaStream_t consumer = nullptr;
    require(cudaMalloc(reinterpret_cast<void **>(&output), x.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&input), x.size() * sizeof(float)) == cudaSuccess &&
            cudaMallocHost(reinterpret_cast<void **>(&landing), x.size() * sizeof(float)) == cudaSuccess &&
            cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess, "caller buffers");
    require(cudaMemcpy(input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "device input");
    auto download = [&](size_t size) {
        require(cudaMemcpyAsync(landing, output, size * sizeof(float), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
                cudaStreamSynchronize(consumer) == cudaSuccess, "independent consumer stream");
        require(std::memcmp(landing, expected.data(), size * sizeof(float)) == 0, "bit-exact single-GPU reference");
    };
    for (int l = 0; l < count; ++l) {
        require(reference.run_to_dev(l, ids.data(), weights.data(), x.data(), output), "reference decode");
        require(cudaMemcpy(expected.data(), output, g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference read");
        require(sharded.run_to_dev(l, ids.data(), weights.data(), x.data(), output), "sharded decode");
        download(size_t(g.n_embd));
        require(sharded.run_to_dev(l, ids.data(), weights.data(), nullptr, output, input), "sharded device input");
        download(size_t(g.n_embd));
        require(reference.run_multi_to_dev(l, n, ids.data(), weights.data(), x.data(), output), "reference multi");
        require(cudaMemcpy(expected.data(), output, x.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference multi read");
        require(sharded.run_multi_to_dev(l, n, ids.data(), weights.data(), x.data(), output), "sharded multi");
        download(x.size());
        require(reference.run_chunk_dev(l, n, ids.data(), weights.data(), x.data(), output), "reference chunk");
        require(cudaMemcpy(expected.data(), output, x.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference chunk read");
        require(sharded.run_chunk_dev(l, n, ids.data(), weights.data(), x.data(), output), "sharded chunk");
        download(x.size());
        auto changed = x;
        for (auto & v : changed) v *= -0.5f;
        require(sharded.run_chunk_dev(l, n, ids.data(), weights.data(), changed.data(), output), "changed input");
        sharded.release_chunk();
        require(sharded.run_chunk_dev(l, n, ids.data(), weights.data(), x.data(), output), "chunk reallocate");
        download(x.size());
        int current = -1;
        require(cudaGetDevice(&current) == cudaSuccess && current == count - 1, "device restoration");
    }
    const Ds41MoeStats stats = sharded.stats();
    std::printf("aggregate: hits=%lld pcie=%lld cpu=%lld file=%lld\n", (long long) stats.hits,
                (long long) stats.pcie, (long long) stats.cpu, (long long) stats.file_tier);
    require(stats.hits > 0 && stats.pcie > 0 && stats.file_tier == 0, "aggregate counters");
    sharded.release_cache();
    require(sharded.resident() == 6 * count, "cache persists across requests");
    sharded.reset_stats();
    require(sharded.stats().lookups() == 0, "aggregate reset");
    sharded.close();
    require(!sharded.sharded(), "close");
    cudaFree(output); cudaFree(input); cudaFreeHost(landing); cudaStreamDestroy(consumer);
    std::printf("%d-GPU runner: mixed Q2_K/Q3_K, pinned tail, decode/multi/chunk, changed input, "
                "independent stream, device restoration and lifecycle PASS\n", count);
    return 0;
#endif
}
