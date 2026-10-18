// Bounded synthetic decode-tier diagnostic: no weights, files, or prompt kernels.
#include "ds41_moe.hpp"
#include "ggml.h"
#ifdef DS41_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::v41;

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

static void check(bool ok, const std::string & what) {
    check(ok, what.c_str());
}

static const char * setting(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "unset";
}

static void report(const char * name, std::vector<double> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    std::printf(" %s_us_min/median/max=%.3f/%.3f/%.3f", name, v.front(), v[v.size()/2], v.back());
}

static void profile(int type, bool cuda, bool bench) {
    Ds41MoeGeom g;
    g.n_embd = g.n_ff = 256;
    g.n_layers = 1;
    g.n_experts = 4;
    g.top_k = 2;
    g.gu_type = g.d_type = type;
    cpu::NativeFmt fmt;
    std::string error;
    check(ds41_moe_blob_layout(g, fmt, error), error);
    Ds41MemoryBlobs source(g, fmt.bytes);
    std::vector<float> weights(3 * 256 * 256);
    std::vector<uint8_t> blob(fmt.bytes);
    for (int e = 0; e < 4; ++e) {
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = 0.025f * std::sin(float(i * 17 % 997 + e * 31));
        check(ggml_quantize_chunk(ggml_type(type), weights.data(), blob.data(), 0, 3 * 256, 256, nullptr)
                  == fmt.bytes, "quantized fixture size");
        source.set(0, e, blob.data());
    }
    Ds41MoeConfig cfg;
    cfg.cpu_only = !cuda;
    cfg.threads = 2;
    cfg.slots = 2;
    cfg.pcie_frac = 1;
    cfg.pf_b = 0;
    cfg.arena_gib = cfg.max_arena_gib = 4.0 * fmt.bytes / 1073741824.0;
    Ds41MoeTier tier, reference;
    check(tier.init(g, &source, cfg, error), error);
    check(tier.seed_from_ranked({{0,0}, {0,1}, {0,2}, {0,3}}, error), error);
    cfg.cpu_only = true;
    check(reference.init(g, &source, cfg, error), error);
    const float mix[] = {0.375f, 0.625f};
    const int32_t routes[][2] = {{0,1}, {0,3}, {1,2}, {3,2}};
    std::vector<std::vector<float>> inputs(4, std::vector<float>(256)), expected(4, std::vector<float>(256));
    std::vector<float> out(256), cpu(256);
    uint64_t digest = 14695981039346656037ull;
    double worst = 0;
    for (int v = 0; v < 4; ++v) {
        for (int i = 0; i < 256; ++i) inputs[v][i] = 0.2f * std::cos(float(i * 11 + v));
        check(reference.run(0, routes[v], mix, inputs[v].data(), cpu.data()), "CPU reference");
        check(tier.run(0, routes[v], mix, inputs[v].data(), expected[v].data()), "tier reference");
        double num = 0, den = 0;
        for (int i = 0; i < 256; ++i) {
            check(std::isfinite(expected[v][i]), "finite tier output");
            const double d = expected[v][i] - cpu[i];
            num += d * d;
            den += double(cpu[i]) * cpu[i];
            uint32_t bits;
            std::memcpy(&bits, &expected[v][i], sizeof(bits));
            digest = (digest ^ bits) * 1099511628211ull;
        }
        worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
    }
    check(worst < 0.05, "CPU/GPU activation-rounding relative L2 < 0.05");
    auto run = [&](int v) {
        check(tier.run(0, routes[v], mix, inputs[v].data(), out.data()), "tier run");
        check(std::memcmp(out.data(), expected[v].data(), out.size() * sizeof(float)) == 0,
              "updated route/input replay is bit-identical");
    };
    for (int i = 0; i < 20; ++i) run(i % 4);
    std::printf("tier type=%d backend=%s blob_bytes=%zu slots=%lld CPU_relative_L2_max=%.6g digest=%016llx",
                type, cuda ? "CUDA" : "CPU", fmt.bytes, (long long) tier.cache_slots(),
                worst, (unsigned long long) digest);
    if (bench) {
        std::vector<double> wall, event;
#ifdef DS41_HAVE_CUDA
        cudaEvent_t start = nullptr, stop = nullptr;
        if (cuda) check(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess, "events");
#endif
        tier.reset_stats();
        for (int r = 0; r < 7; ++r) {
#ifdef DS41_HAVE_CUDA
            // The tier owns nonblocking streams; explicit device boundaries make
            // these submission-envelope events, NOT isolated kernel timings.
            if (cuda) {
                check(cudaDeviceSynchronize() == cudaSuccess, "boundary");
                check(cudaEventRecord(start) == cudaSuccess && cudaEventSynchronize(start) == cudaSuccess, "start");
            }
#endif
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; ++i) run(i % 4);
            const auto end = std::chrono::steady_clock::now();
            wall.push_back(std::chrono::duration<double, std::micro>(end - begin).count() / 64);
#ifdef DS41_HAVE_CUDA
            if (cuda) {
                check(cudaDeviceSynchronize() == cudaSuccess, "completion boundary");
                check(cudaEventRecord(stop) == cudaSuccess && cudaEventSynchronize(stop) == cudaSuccess, "stop");
                float ms = 0;
                check(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "elapsed");
                event.push_back(ms * 1000 / 64);
            }
#endif
        }
        report("wall_sync", wall);
        report("event_submission", event);
        const auto & s = tier.stats();
        std::printf(" hits=%lld pcie=%lld cpu=%lld file=%lld warmup=20 repeats=7 iterations=64",
                    (long long)s.hits, (long long)s.pcie, (long long)s.cpu, (long long)s.file_tier);
        check(cuda ? s.hits == 448 && s.pcie == 448 && s.cpu == 0 && s.file_tier == 0
                   : s.cpu == 896, "measured tier coverage");
#ifdef DS41_HAVE_CUDA
        if (cuda) { cudaEventDestroy(start); cudaEventDestroy(stop); }
#endif
    }
    std::puts(" replay=passed");
}

int main(int argc, char ** argv) {
    bool cuda = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cuda")) cuda = true;
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
        else { std::fprintf(stderr, "usage: %s [--cuda] [--bench]\n", argv[0]); return 1; }
    }
#ifdef DS41_HAVE_CUDA
    if (cuda) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
        check(cudaSetDevice(0) == cudaSuccess, "device");
    }
#else
    if (cuda) return 77;
#endif
    std::printf("STRATA_TIER_GRAPH=%s STRATA_DMA_OVERLAP=%s\n",
                setting("STRATA_TIER_GRAPH"), setting("STRATA_DMA_OVERLAP"));
    for (int type : {10, 11}) profile(type, cuda, bench);
}
