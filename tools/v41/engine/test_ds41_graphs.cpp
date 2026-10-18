// Synthetic activation-rounding/capture regression; no model or fixture files.
#include "ds41_ops.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace strata::v41;

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

struct Graph {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_gallocr_t alloc;
    Graph(ggml_backend_t backend, Ds41Quant mode, int64_t n) {
        ctx = ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true});
        check(ctx != nullptr, "context");
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(input);
        output = ds41_fake_quant(ctx, input, mode);
        ggml_set_output(output);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        // Same stable identity used by cached dense decode graphs.
        graph->uid = ggml_graph_next_uid();
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        check(ggml_gallocr_alloc_graph(alloc, graph), "graph allocation");
    }
    ~Graph() { ggml_gallocr_free(alloc); ggml_free(ctx); }
    std::vector<float> run(ggml_backend_t backend, const std::vector<float> & x) {
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> out(x.size());
        ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }
};

static bool same(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
}

static void equal(const std::vector<float> & a, const std::vector<float> & b, const char * what) {
    check(a.size() == b.size(), "size");
    for (size_t i = 0; i < a.size(); ++i) {
        if (!same(a[i], b[i])) {
            std::fprintf(stderr, "%s: [%zu] %.9g != %.9g\n", what, i, a[i], b[i]);
            check(false, what);
        }
    }
}

// Enumerate the positive format codes. This reference does not use the composite's
// Veltkamp split or magic-add rounding; ties select the even encoded value.
static float nearest(float x, int mbits, float min_normal, float maxv) {
    float best = 0, error = std::fabs(x);
    int code = 0;
    const int k = 1 << mbits;
    for (float p = min_normal; p <= maxv; p *= 2) {
        const int begin = p == min_normal ? 1 : k;
        for (int j = begin; j < 2 * k; ++j) {
            const float value = p * j / k;
            if (value > maxv) break;
            ++code;
            const float e = std::fabs(std::fabs(x) - value);
            if (e < error || (e == error && code % 2 == 0)) { best = value; error = e; }
        }
    }
    return std::copysign(best, x);
}

static std::vector<float> reference(const std::vector<float> & x, Ds41Quant mode) {
    const bool fp8 = mode == Ds41Quant::FP8_POW2_32;
    const bool e4 = mode == Ds41Quant::FP4_E4M3_16;
    const int bs = e4 ? 16 : 32;
    const float maxv = fp8 ? 448 : 6;
    const float floor = fp8 ? 1e-4f : e4 ? 6.0f / 512 : 6.0f * 0x1p-126f;
    std::vector<float> out(x.size());
    for (size_t b = 0; b < x.size(); b += bs) {
        float amax = floor;
        bool finite = true;
        for (int j = 0; j < bs; ++j) {
            finite &= std::isfinite(x[b+j]);
            amax = std::max(amax, std::fabs(x[b+j]));
        }
        if (!finite || amax > 2000) continue; // Nonfinite/overflow behavior is checked against first execution.
        const float t = amax / maxv;
        const float scale = e4 ? nearest(t, 3, 0x1p-6f, 448) : std::exp2(std::ceil(std::log2(t)));
        for (int j = 0; j < bs; ++j) {
            const float v = std::clamp(x[b+j] / scale, -maxv, maxv);
            const float q = nearest(v, fp8 ? 3 : 1, fp8 ? 0x1p-6f : 1, maxv);
            // The composite's magic-add RNE intentionally produces positive zero.
            out[b+j] = (q == 0 ? 0 : q) * scale;
        }
    }
    return out;
}

static std::vector<float> inputs(size_t n, int variant, int bs) {
    const float edge[] = {0, -0.0f, 0.25f, 0.5f, 0.75f, 1, 1.25f, 1.5f, 1.75f, 2.5f, 3.5f, 5,
                          0x1p-10f, 0x1p-9f, 0x1p-8f, 0x1p-7f};
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        const float v = edge[(i + variant) % 16];
        x[i] = ((i / 16) & 1) ? -v : v;
        if ((i / bs) % 5 == 1) x[i] *= 0x1p-12f;
        if ((i / bs) % 5 == 2) x[i] *= 32;
        if ((i / bs) % 5 == 3) x[i] = 0;
    }
    return x;
}

static void numeric(ggml_backend_t backend, Ds41Quant mode) {
    const int bs = mode == Ds41Quant::FP4_E4M3_16 ? 16 : 32;
    Graph g(backend, mode, 512);
    for (int variant = 0; variant < 4; ++variant) {
        auto x = inputs(512, variant, bs);
        equal(g.run(backend, x), reference(x, mode), "finite reference");
    }
    uint32_t rng = 41;
    for (int variant = 0; variant < 4; ++variant) {
        std::vector<float> x(512);
        for (float & v : x) {
            rng = rng * 1664525u + 1013904223u;
            v = (float(int(rng >> 8)) / 8388608.0f - 1.0f) * 16;
        }
        equal(g.run(backend, x), reference(x, mode), "random finite reference");
    }
    // Keep special values in separate blocks: changing amax affects every lane.
    auto x = inputs(512, 0, bs);
    x[0] = std::numeric_limits<float>::infinity();
    x[bs] = -std::numeric_limits<float>::infinity();
    x[2*bs] = std::numeric_limits<float>::quiet_NaN();
    x[3*bs] = std::numeric_limits<float>::max();
    Graph fresh(backend, mode, 512);
    const auto want = fresh.run(backend, x); // First call is eager on stock CUDA.
    for (int i = 0; i < 6; ++i) equal(fresh.run(backend, x), want, "special replay");
    for (int variant = 0; variant < 4; ++variant) {
        auto y = inputs(512, variant, bs);
        equal(fresh.run(backend, y), reference(y, mode), "updated input after capture");
    }
    std::printf("numeric mode=%d: ties/subnormals/zero/nonfinite/update/replay passed\n", int(mode));
}

static void timing(ggml_backend_t backend, Ds41Quant mode, int64_t n, bool cuda) {
    Graph g(backend, mode, n);
    const int bs = mode == Ds41Quant::FP4_E4M3_16 ? 16 : 32;
    const auto x = inputs(size_t(n), 0, bs);
    g.run(backend, x);
    for (int i = 0; i < 20; ++i)
        check(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS, "warmup");
    std::vector<double> wall, event;
    constexpr int repeats = 7, iterations = 64;
#ifdef DS41_HAVE_CUDA
    cudaEvent_t start = nullptr, stop = nullptr;
    if (cuda) {
        check(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess, "events");
    }
#endif
    for (int r = 0; r < repeats; ++r) {
        ggml_backend_synchronize(backend);
#ifdef DS41_HAVE_CUDA
        // Events bracket synchronized submission, not pure kernel execution.
        // Waiting for start and the backend before stop also orders nonblocking
        // backend streams without accessing ggml's private stream/context.
        if (cuda) {
            check(cudaEventRecord(start) == cudaSuccess && cudaEventSynchronize(start) == cudaSuccess, "start");
        }
#endif
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            check(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS, "timed compute");
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        wall.push_back(std::chrono::duration<double, std::micro>(end - begin).count() / iterations);
#ifdef DS41_HAVE_CUDA
        if (cuda) {
            check(cudaEventRecord(stop) == cudaSuccess && cudaEventSynchronize(stop) == cudaSuccess, "stop");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "elapsed");
            event.push_back(ms * 1000 / iterations);
        }
#endif
    }
    auto report = [](const char * name, std::vector<double> v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        std::printf(" %s_us_min/median/max=%.3f/%.3f/%.3f", name, v.front(), v[v.size()/2], v.back());
    };
    std::printf("timing mode=%d elements=%lld nodes=%d scratch_bytes=%zu warmup=20 repeats=7 iterations=64",
                int(mode), (long long)n, ggml_graph_n_nodes(g.graph), ggml_gallocr_get_buffer_size(g.alloc, 0));
    report("wall_sync", wall);
    report("event_submission", event);
    std::puts("");
#ifdef DS41_HAVE_CUDA
    if (cuda) { cudaEventDestroy(start); cudaEventDestroy(stop); }
#else
    (void)cuda;
#endif
}

int main(int argc, char ** argv) {
    bool cuda = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cuda")) cuda = true;
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
        else { std::fprintf(stderr, "usage: %s [--cuda] [--bench]\n", argv[0]); return 1; }
    }
    ggml_backend_t backend = nullptr;
#ifdef DS41_HAVE_CUDA
    if (cuda && ggml_backend_cuda_get_device_count()) backend = ggml_backend_cuda_init(0);
#endif
    if (cuda && !backend) { std::puts("CUDA unavailable"); return 77; }
    if (!cuda) backend = ggml_backend_cpu_init();
    check(backend != nullptr, "backend");
    if (!cuda) ggml_backend_cpu_set_n_threads(backend, 2);
    const char * disabled = std::getenv("GGML_CUDA_DISABLE_GRAPHS");
    std::printf("backend=%s GGML_CUDA_DISABLE_GRAPHS=%s\n", cuda ? "CUDA" : "CPU", disabled ? disabled : "unset");
    for (auto mode : {Ds41Quant::FP8_POW2_32, Ds41Quant::FP4_E4M3_16, Ds41Quant::FP4_POW2_32}) {
        numeric(backend, mode);
        if (bench) for (int64_t n : {512, 32768}) timing(backend, mode, n, cuda);
    }
    ggml_backend_free(backend);
    return 0;
}
