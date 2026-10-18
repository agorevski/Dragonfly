// Synthetic full-width experts only; no model or fork MMQ backend is required.
#include "strata/kernels/iq_kernels.hpp"
#include "ds41_moe.hpp"
#include "ggml.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
#define CK(x) do { const auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { CK(cudaMalloc(&p, n)); }
    ~Device() { cudaFree(p); }
    template<class T> T* as() const { return (T*) p; }
};

static std::vector<uint8_t> weights(int type, int rows, int width, std::mt19937& rng) {
    const auto* traits = ggml_get_type_traits((ggml_type) type);
    const size_t rb = k::iq_row_bytes(type, width);
    std::vector<uint8_t> q(rb * rows);
    std::vector<float> row(width);
    std::normal_distribution<float> nd(0, .02f);
    for (int r = 0; r < rows; ++r) {
        for (auto& v : row) v = nd(rng);
        traits->from_float_ref(row.data(), q.data() + r * rb, width);
    }
    return q;
}

static std::vector<double> roundtrip(const float* x, int width) {
    std::vector<double> y(width);
    for (int b = 0; b < width; b += 32) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(x[b + j]));
        const float d = amax / 127.f, dh = __half2float(__float2half(d));
        for (int j = 0; j < 32; ++j) y[b + j] = amax == 0 ? 0 : dh * std::round(x[b + j] / d);
    }
    return y;
}

static double dot(const uint8_t* q, int type, const std::vector<double>& x) {
    std::vector<float> w(x.size());
    ggml_get_type_traits((ggml_type) type)->to_float(q, w.data(), x.size());
    double s = 0;
    for (size_t j = 0; j < x.size(); ++j) s += w[j] * x[j];
    return s;
}

static void chunk_dispatch(const std::vector<uint8_t>& blobs, size_t bytes,
                           const std::vector<float>& inputs, int n) {
    using namespace strata::v41;
    Ds41MoeGeom g;
    g.n_embd = 5120; g.n_ff = 2304;
    g.n_layers = 1; g.n_experts = g.top_k = 2;
    g.gu_type = 10; g.d_type = 11; g.swiglu_limit = 10;
    Ds41MemoryBlobs src(g, bytes);
    for (int e = 0; e < 2; ++e) src.set(0, e, blobs.data() + e * bytes);
    std::vector<int32_t> ids(2 * n);
    std::vector<float> weights(2 * n), ref((size_t) n * g.n_embd), got(ref.size());
    for (int t = 0; t < n; ++t) {
        ids[2 * t] = t % 2; ids[2 * t + 1] = 1 - t % 2;
        weights[2 * t] = .25f; weights[2 * t + 1] = .75f;
    }
    for (bool resident : {false, true}) {
        for (bool batch : {false, true}) {
            Ds41MoeConfig cfg;
            REQUIRE(!cfg.chunk_batch);
            cfg.chunk_batch = batch;
            cfg.slots = resident ? 2 : 0; cfg.no_cache = !resident;
            cfg.arena_gib = cfg.max_arena_gib = 0;
            cfg.pcie_frac = 1; cfg.pf_b = 0; cfg.threads = 2;
            cfg.chunk_prefetch = false;
            Ds41MoeTier tier;
            std::string err;
            if (!tier.init(g, &src, cfg, err)) {
                std::fprintf(stderr, "chunk fixture init: %s\n", err.c_str());
                std::exit(1);
            }
            if (resident) REQUIRE(tier.seed_from_ranked({{0, 0}, {0, 1}}, err));
            REQUIRE(tier.run_chunk(0, n, ids.data(), weights.data(), inputs.data(),
                                   batch ? got.data() : ref.data()));
        }
        for (float v : got) REQUIRE(std::isfinite(v));
        REQUIRE(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0);
        std::printf("chunk dispatch default-OFF vs enabled %s: %d tokens, weighted output bitwise exact\n",
                    resident ? "resident" : "streamed", n);
    }
}

static void run(int gt, int ut, int per, bool oracle, std::mt19937& rng,
                bool dispatch = true, int gu_mode = -1) {
    constexpr int H = 5120, FF = 2304, NG = 3;
    // One empty group, nonzero entry offset, unequal lengths, and nonidentity routing.
    const int count = 2 * per + 1, cap = count + 2, ntok = per + 2;
    const int32_t starts[] = {1, 1 + per, 1 + per, 1 + count}, ng = NG;
    auto L = k::native_expert_layout3(gt, ut, 11, H, FF);
    L.swiglu_limit = 10;
    REQUIRE(k::native_expert_batch_supported(L));
    std::vector<uint8_t> blobs(2 * L.bytes);
    for (int e = 0; e < 2; ++e) {
        auto gate = weights(gt, FF, H, rng), up = weights(ut, FF, H, rng), down = weights(11, H, FF, rng);
        std::memcpy(blobs.data() + e * L.bytes, gate.data(), gate.size());
        std::memcpy(blobs.data() + e * L.bytes + L.up_off, up.data(), up.size());
        std::memcpy(blobs.data() + e * L.bytes + L.down_off, down.data(), down.size());
    }
    Device blob(blobs.size()), ptr(NG * 8), start(sizeof(starts)), num(4), tok(cap * 4), dst(cap * 4);
    Device x((size_t) ntok * H * 4), xq((size_t) ntok * H / 32 * 36);
    const size_t sb = k::native_expert_scratch_bytes(cap, FF);
    Device scratch(sb), out((size_t) cap * H * 4);
    CK(cudaMemcpy(blob.p, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    const unsigned long long pointers[] = {(unsigned long long) blob.p, (unsigned long long) blob.p,
                                          (unsigned long long) blob.p + L.bytes};
    std::vector<int32_t> tokens(cap), dest(cap);
    for (int e = 0; e < cap; ++e) { tokens[e] = (e * 7) % ntok; dest[e] = cap - e - 1; }
    CK(cudaMemcpy(ptr.p, pointers, sizeof(pointers), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(start.p, starts, sizeof(starts), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(num.p, &ng, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tok.p, tokens.data(), cap * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst.p, dest.data(), cap * 4, cudaMemcpyHostToDevice));
    std::vector<float> inputs((size_t) ntok * H);
    std::normal_distribution<float> nd(0, 20);
    for (auto& v : inputs) v = nd(rng);
    std::fill(inputs.begin(), inputs.begin() + H, 0); // zero q8 scale edge case
    CK(cudaMemcpy(x.p, inputs.data(), inputs.size() * 4, cudaMemcpyHostToDevice));
    k::quantize_q8_1_rows(x.as<float>(), ntok, H, xq.p, nullptr);
    auto launch = [&](bool batch) {
        if (gu_mode >= 0) k::native_expert_set_mode(batch ? 0 : gu_mode, 0);
        if (batch) REQUIRE(k::native_expert_batch(L, ptr.as<unsigned long long>(), start.as<int32_t>(),
            num.as<int32_t>(), dst.as<int32_t>(), tok.as<int32_t>(), NG, cap, per + 1,
            xq.p, scratch.p, out.as<float>(), nullptr));
        else k::native_expert_grouped(L, ptr.as<unsigned long long>(), start.as<int32_t>(), num.as<int32_t>(),
            dst.as<int32_t>(), tok.as<int32_t>(), NG, cap, xq.p, scratch.p, out.as<float>(), nullptr, NG);
    };
    std::vector<float> ref((size_t) cap * H), got(ref.size());
    std::vector<uint8_t> refscratch(sb), gotscratch(sb);
    CK(cudaMemset(out.p, 0x42, ref.size() * 4));
    CK(cudaMemset(scratch.p, 0, sb));
    launch(false);
    CK(cudaMemcpy(ref.data(), out.p, ref.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(refscratch.data(), scratch.p, sb, cudaMemcpyDeviceToHost));
    CK(cudaMemset(out.p, 0x42, ref.size() * 4));
    CK(cudaMemset(scratch.p, 0, sb));
    launch(true);
    CK(cudaMemcpy(got.data(), out.p, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gotscratch.data(), scratch.p, sb, cudaMemcpyDeviceToHost));
    size_t differences = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(std::isfinite(got[i]));
        differences += std::memcmp(&ref[i], &got[i], 4) != 0;
    }
    const size_t fa = ((size_t) cap * FF * 4 + 255) & ~(size_t) 255;
    const size_t hq_offset = 3 * fa;
    const size_t hq_bytes = (size_t) cap * FF / 32 * 36;
    REQUIRE(std::memcmp(refscratch.data() + hq_offset, gotscratch.data() + hq_offset, hq_bytes) == 0);
    REQUIRE(differences == 0);
    double output_norm = 0;
    for (int e = starts[0]; e < starts[NG]; ++e)
        for (int r = 0; r < H; ++r) {
            const double v = got[(size_t) dest[e] * H + r];
            output_norm += v * v;
        }
    REQUIRE(output_norm > 0);
    if (oracle) {
        double err = 0, norm = 0;
        int clamped = 0;
        for (int e : {1, count}) {
            const auto* b = blobs.data() + (e >= starts[2] ? L.bytes : 0);
            auto qx = roundtrip(inputs.data() + (size_t) tokens[e] * H, H);
            std::vector<float> hidden(FF);
            for (int r = 0; r < FF; ++r) {
                const float g = (float) dot(b + (size_t) r * L.gu_row, gt, qx);
                const size_t ur = ut == gt ? L.gu_row : L.up_row;
                const float u = (float) dot(b + L.up_off + (size_t) r * ur, ut, qx);
                clamped += g > 10 || std::abs(u) > 10;
                const float cg = std::min(g, 10.f), cu = std::clamp(u, -10.f, 10.f);
                hidden[r] = (cg / (1 + std::exp(-cg))) * cu;
            }
            auto qh = roundtrip(hidden.data(), FF);
            for (int r = 0; r < H; ++r) {
                const double want = dot(b + L.down_off + (size_t) r * L.d_row, 11, qh);
                const double d = got[(size_t) dest[e] * H + r] - want;
                err += d * d; norm += want * want;
            }
        }
        const double rel = std::sqrt(err / std::max(norm, 1e-30));
        std::printf("oracle %d/%d/11 per=%d clamp=%d relative-L2=%.6g\n", gt, ut, per, clamped, rel);
        REQUIRE(clamped > 0);
        REQUIRE(rel < .005);
    }
    auto time = [&](bool batch) {
        for (int i = 0; i < 3; ++i) launch(batch);
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a));
        for (int i = 0; i < 10; ++i) launch(batch);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        return ms / 10;
    };
    std::vector<float> baseline, batch;
    for (int repeat = 0; repeat < 5; ++repeat) {
        if (repeat % 2 == 0) { baseline.push_back(time(false)); batch.push_back(time(true)); }
        else { batch.push_back(time(true)); baseline.push_back(time(false)); }
    }
    std::sort(baseline.begin(), baseline.end());
    std::sort(batch.begin(), batch.end());
    std::printf("%d/%d/11 per=%3d entries=%3d bitwise-diff=%zu hidden-q8=exact "
                "5x10 launches median[min,max] grouped=%.3f[%.3f,%.3f]ms batch=%.3f[%.3f,%.3f]ms speedup=%.2fx\n",
                gt, ut, per, count, differences, baseline[2], baseline.front(), baseline.back(),
                batch[2], batch.front(), batch.back(), baseline[2] / batch[2]);
    const int32_t empty = 0;
    CK(cudaMemcpy(num.p, &empty, sizeof(empty), cudaMemcpyHostToDevice));
    launch(true);
    CK(cudaMemcpy(ref.data(), out.p, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    REQUIRE(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0);
    if (dispatch && oracle && gt == 10 && ut == 10) chunk_dispatch(blobs, L.bytes, inputs, ntok);
}

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    const bool probe = argc == 2 && std::strcmp(argv[1], "--probe") == 0;
    const bool gu_layouts = argc == 2 && std::strcmp(argv[1], "--gu-layouts") == 0;
    REQUIRE(argc == 1 || quick || probe || gu_layouts);
    if (gu_layouts) {
        REQUIRE(!std::getenv("STRATA_EXP_MODE_T10") && !std::getenv("STRATA_EXP_MODE_T11"));
        // Isolate GU changes: retain the default Q3_K down projection for both sides of the comparison.
#ifdef _WIN32
        REQUIRE(_putenv_s("STRATA_EXP_MODE_T11", "0") == 0);
#else
        REQUIRE(setenv("STRATA_EXP_MODE_T11", "0", 1) == 0);
#endif
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    char bus[32];
    CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 0));
    std::printf("device=%s sm=%d%d pci=%s CUDA_VISIBLE_DEVICES=%s synthetic-only <3GiB\n",
                p.name, p.major, p.minor, bus,
                std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "(unset)");
    std::printf("STRATA_DS41_BATCH_FUSED_GU=%s\n", std::getenv("STRATA_DS41_BATCH_FUSED_GU") ?
                std::getenv("STRATA_DS41_BATCH_FUSED_GU") : "(unset; unfused)");
    for (const char* name : {"STRATA_DS41_BATCH_COLS", "STRATA_DS41_BATCH_PAIR_GU",
                             "STRATA_DS41_BATCH_DOWN_WARPS"})
        std::printf("%s=%s\n", name, std::getenv(name) ? std::getenv(name) : "(unset)");
    auto L = k::native_expert_layout(10, 11, 5120, 2304);
    auto bad = L; bad.n_embd = 2560;
    REQUIRE(!k::native_expert_batch_supported(bad));
    REQUIRE(!k::native_expert_batch(bad, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1, 1,
                                   nullptr, nullptr, nullptr, nullptr));
    bad = L; bad.tail_off = bad.bytes;
    REQUIRE(!k::native_expert_batch_supported(bad));
    bad = L; ++bad.down_off;
    REQUIRE(!k::native_expert_batch_supported(bad));
    REQUIRE(!k::native_expert_batch(L, nullptr, nullptr, nullptr, nullptr, nullptr, 65536, 1, 1,
                                   nullptr, nullptr, nullptr, nullptr));
    REQUIRE(!k::native_expert_batch(L, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1, 262141,
                                   nullptr, nullptr, nullptr, nullptr));
    REQUIRE(k::native_expert_batch(L, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0,
                                  nullptr, nullptr, nullptr, nullptr));
    std::mt19937 rng(7319);
    if (probe) {
        for (int gt : {10, 11})
            for (int ut : {10, 11}) run(gt, ut, 9, true, rng, false);
        std::puts("PASS: bounded mixed-format batch probe");
        return 0;
    }
    if (gu_layouts) {
        for (int mode : {2, 4}) {
            std::printf("Q2_K GU rows/warp mode=%d vs unchanged batch oracle\n", mode);
            for (int per : {1, 3, 7}) run(10, 10, per, per == 7, rng, true, mode);
            run(10, 11, 3, true, rng, true, mode);
            run(11, 10, 3, true, rng, true, mode);
        }
        k::native_expert_set_mode(-1, 0);
        std::puts("PASS: full-width Q2_K GU layout modes and mixed-format fallback");
        return 0;
    }
    for (int per : {1, 3, 4, 7, 8, 9, 32, 128})
        if (!quick || per == 1 || per == 7 || per == 8 || per == 9)
            run(10, 10, per, per == 7, rng);
    run(10, 11, 7, true, rng);
    run(11, 10, 7, true, rng);
    run(11, 11, 7, true, rng);
    std::puts("PASS: synthetic batch expert correctness and timing");
}
