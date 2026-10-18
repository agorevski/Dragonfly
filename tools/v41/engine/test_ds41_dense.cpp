// test_ds41_dense <file.gguf> --tokens 1,2,3 --out DIR [--engram DIR1:DIR2:...] [--mode decode|prefill|both]
//                 [--threads N] [--chunk N]
// Runs the V4.1 dense half on the CPU over the tokens (decode: one token per pass; prefill: chunks of --chunk),
// with the routed experts computed by a plain reference (rows dequantized, F32 matvec) and writes per-layer block
// outputs, router ids / weights and the logits as raw files for compare_ds41.py (vs ds41_golden.py's golden.npz):
//   DIR/<mode>/x<L>.f32 [S, HC, D], pre<L>.f32 [S, HC], ids<L>.i32 [S, 6], wts<L>.f32 [S, 6], logits.f32 [S, V]
#include "ds41_dense.hpp"

#include "strata/kernels/cpu/ptq1_0.hpp"

#include "ggml-cpu.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

void save(const std::string & p, const void * d, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char *) d, (std::streamsize) bytes);
}

// the routed experts, by the book: rows dequantized to F32, clamped SwiGLU, weighted sum
struct RefExperts {
    const GgufModel * m = nullptr;
    const Ds41Geometry * g = nullptr;
    int threads = 8;
    std::vector<float> row(int type, const uint8_t * src, int64_t n) const {
        std::vector<float> y((size_t) n);
        if (type == 143) kernels::cpu::ptq1_0_dequant(src, y.data(), n);
        else ggml_get_type_traits((ggml_type) type)->to_float(src, y.data(), n);
        return y;
    }
    size_t row_bytes(int type, int64_t n) const {
        int be = 0, bb = 0;
        block_geometry((uint32_t) type, be, bb);
        return (size_t) (n / be) * (size_t) bb;
    }
    // y[D] += w * expert_e(x)
    void expert(int64_t L, int e, float w, const float * x, float * y) const {
        const Ds41ExpertMap & X = g->experts[(size_t) L];
        const int gt = X.gu_type[(size_t) e], dt = X.d_type[(size_t) e];
        const int64_t D = g->n_embd, FF = g->n_ff;
        auto base = [&](const std::string & name, int slot, int64_t rows, int64_t cols, int type) {
            size_t s = 0;
            const TensorInfo * t = m->find(name, &s);
            return m->shard(s).tensor_data(*t) + (size_t) slot * (size_t) rows * row_bytes(type, cols);
        };
        const uint8_t * Wg = base(X.gate_t.at(gt), X.gu_slot[(size_t) e], FF, D, gt);
        const uint8_t * Wu = base(X.up_t.at(gt), X.gu_slot[(size_t) e], FF, D, gt);
        const uint8_t * Wd = base(X.down_t.at(dt), X.d_slot[(size_t) e], D, FF, dt);
        std::vector<float> a((size_t) FF);
        const size_t rg = row_bytes(gt, D), rd = row_bytes(dt, FF);
        const float lim = (float) g->swiglu_limit;
        auto par = [&](int64_t n, auto fn) {
            std::vector<std::thread> th;
            for (int k = 0; k < threads; ++k) th.emplace_back([&, k] { for (int64_t i = k; i < n; i += threads) fn(i); });
            for (auto & t : th) t.join();
        };
        par(FF, [&](int64_t i) {
            const std::vector<float> rg_ = row(gt, Wg + (size_t) i * rg, D), ru = row(gt, Wu + (size_t) i * rg, D);
            double sg = 0, su = 0;
            for (int64_t k = 0; k < D; ++k) { sg += (double) rg_[(size_t) k] * x[k]; su += (double) ru[(size_t) k] * x[k]; }
            float gg = (float) sg, uu = (float) su;
            if (lim > 0) { uu = std::min(std::max(uu, -lim), lim); gg = std::min(gg, lim); }
            a[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu * w;
        });
        std::vector<float> out((size_t) D);
        par(D, [&](int64_t i) {
            const std::vector<float> r = row(dt, Wd + (size_t) i * rd, FF);
            double s = 0;
            for (int64_t k = 0; k < FF; ++k) s += (double) r[(size_t) k] * a[(size_t) k];
            out[(size_t) i] = (float) s;
        });
        for (int64_t i = 0; i < D; ++i) y[i] += out[(size_t) i];
    }
};

std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::stoi(x));
    return v;
}

}  // namespace

static int span_cache_test(const std::string & path) {
    std::vector<uint8_t> reference;
    for (int mode = 0; mode < 4; ++mode) {
        setenv("STRATA_DS41_PREFILL_GRAPH_CACHE", mode & 1 ? "1" : "0", 1);
        setenv("STRATA_DS41_PREFILL_ROW_CACHE", mode & 2 ? "1" : "0", 1);
        Ds41DenseConfig cfg;
        cfg.n_threads = 2;
        cfg.ctx_max = 512;
        cfg.max_pass = 3;
        cfg.prompt_batch = 257;
        Ds41Dense dn;
        std::string err;
        if (!dn.init(path, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        const int D = dn.geom().n_embd, V = dn.geom().vocab, NU = dn.geom().n_used;
        std::vector<uint8_t> result;
        auto append = [&](const void * p, size_t size) {
            const auto * bytes = static_cast<const uint8_t *>(p);
            result.insert(result.end(), bytes, bytes + size);
        };
        auto require = [&](bool ok) {
            if (!ok) { std::fprintf(stderr, "span cache mode %d: %s\n", mode, dn.last_error().c_str()); std::exit(1); }
        };
        int pos = 0;
        const auto start = std::chrono::steady_clock::now();
        // Repeated shapes with changed ids/positions, odd tails, column reallocations, compressed-cap growth,
        // and 65 spans (130 init/finish entries) to force the bounded graph cache to evict live shapes.
        for (int B : { 7, 7, 8, 8, 1, 11, 193, 193, 7 }) {
            require(dn.set_columns(B));
            std::vector<int> tids(B);
            for (int i = 0; i < B; ++i) tids[i] = (pos + i * 3) % V;
            require(dn.begin_batch(tids.data(), B, pos));
            std::vector<float> routed((size_t) B * D), logits((size_t) B * V);
            for (int layer = 0; layer < dn.n_layer(); ++layer) {
                for (int c = 0; c < B; c += cfg.max_pass)
                    require(dn.attn_router_span(layer, c, std::min(cfg.max_pass, B - c)));
                append(dn.batch_fn(), (size_t) B * D * sizeof(float));
                append(dn.batch_ids(), (size_t) B * NU * sizeof(int32_t));
                append(dn.batch_wts(), (size_t) B * NU * sizeof(float));
                for (size_t i = 0; i < routed.size(); ++i)
                    routed[i] = dn.batch_fn()[i] * .125f + .001f * (float) ((i + pos) % 7);
                require(dn.upload_routed(routed.data(), B));
                for (int c = 0; c < B; c += cfg.max_pass)
                    require(dn.finish_span(layer, c, std::min(cfg.max_pass, B - c)));
                std::vector<float> x, pre;
                dn.read_state(x, pre);
                append(x.data(), x.size() * sizeof(float));
                append(pre.data(), pre.size() * sizeof(float));
            }
            require(dn.logits(0, B, logits.data()));
            for (float value : logits) require(std::isfinite(value));
            append(logits.data(), logits.size() * sizeof(float));
            pos += B;
        }
        if (mode == 0) reference = std::move(result);
        else if (result != reference) {
            std::fprintf(stderr, "span cache mode %d: outputs differ from default OFF\n", mode);
            return 1;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("span cache graph=%d rows=%d: bitwise exact, %.3f s\n", mode & 1, (mode >> 1) & 1, seconds);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--span-cache-test") return span_cache_test(argv[2]);
    if (argc < 2) { std::fprintf(stderr, "usage: see the file header\n"); return 2; }
    std::string path = argv[1], out = "/tmp/ds41", mode = "both";
    std::vector<int> toks;
    std::vector<std::string> edirs;
    int threads = 8, chunk = 0;
    bool zero_experts = false, act_quant = true, kv_f32 = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--tokens") toks = parse_ints(next());
        else if (a == "--out") out = next();
        else if (a == "--mode") mode = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--zero-experts") zero_experts = true;
        else if (a == "--no-actquant") act_quant = false;
        else if (a == "--kv-f32") kv_f32 = true;
        else if (a == "--engram") {
            std::stringstream ss(next());
            std::string d;
            while (std::getline(ss, d, ':')) edirs.push_back(d);
        }
    }
    const int S = (int) toks.size();
    if (!S) { std::fprintf(stderr, "no tokens\n"); return 2; }
    if (chunk <= 0) chunk = S;
    for (const std::string m : { std::string("decode"), std::string("prefill") }) {
        if (mode != "both" && mode != m) continue;
        Ds41DenseConfig cfg;
        cfg.n_threads = threads;
        cfg.ctx_max = 4096;
        cfg.max_pass = m == "decode" ? 1 : chunk;
        cfg.engram_dirs = edirs;
        cfg.act_quant = act_quant;
        if (kv_f32) cfg.kv_type = GGML_TYPE_F32;
        Ds41Dense dn;
        std::string err;
        if (!dn.init(path, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        const Ds41Geometry & G = dn.geom();
        GgufModel model = GgufModel::open(path);
        RefExperts rx { &model, &G, threads };
        const int64_t nl = dn.n_layer(), D = G.n_embd, HC = G.hc, NU = G.n_used, V = G.vocab;
        std::vector<std::vector<float>> X((size_t) nl), P((size_t) nl), WT((size_t) nl);
        std::vector<std::vector<int>> ID((size_t) nl);
        std::vector<float> logits((size_t) S * (size_t) V);
        ::mkdir(out.c_str(), 0755);
        const auto t0 = std::chrono::steady_clock::now();
        const int step = m == "decode" ? 1 : chunk;
        for (int p0 = 0; p0 < S; p0 += step) {
            const int n = std::min(step, S - p0);
            if (!dn.begin(&toks[(size_t) p0], n, p0)) { std::fprintf(stderr, "begin: %s\n", dn.last_error().c_str()); return 1; }
            std::vector<int> ids((size_t) (n * NU));
            std::vector<float> wts((size_t) (n * NU)), routed((size_t) (n * D));
            for (int64_t l = 0; l < nl; ++l) {
                const float * fn = nullptr;
                if (!dn.attn_router((int) l, ids.data(), wts.data(), &fn)) { std::fprintf(stderr, "attn L%lld: %s\n", (long long) l, dn.last_error().c_str()); return 1; }
                std::fill(routed.begin(), routed.end(), 0.0f);
                if (!zero_experts)
                    for (int t = 0; t < n; ++t)
                        for (int k = 0; k < NU; ++k)
                            rx.expert(l, ids[(size_t) (t * NU + k)], wts[(size_t) (t * NU + k)], fn + (size_t) t * D, &routed[(size_t) t * D]);
                if (!dn.finish((int) l, routed.data())) { std::fprintf(stderr, "finish: %s\n", dn.last_error().c_str()); return 1; }
                {   // index source layers: the visible top-k rows per query (gate of the indexer / candidates)
                    std::vector<int> sl;
                    std::vector<float> sm;
                    int K = 0;
                    dn.read_sel((int) l, sl, sm, &K);
                    if (K > 0) {
                        std::ofstream f(out + "/" + m + "_sel" + std::to_string(l) + ".txt", std::ios::app);
                        for (int t = 0; t < n; ++t) {
                            f << (p0 + t);
                            for (int j = 0; j < K; ++j)
                                if (sm[(size_t) (t * K + j)] == 0.0f) f << ' ' << sl[(size_t) (t * K + j)];
                            f << '\n';
                        }
                    }
                }
                std::vector<float> xs, pr;
                dn.read_state(xs, pr);
                X[(size_t) l].insert(X[(size_t) l].end(), xs.begin(), xs.end());
                P[(size_t) l].insert(P[(size_t) l].end(), pr.begin(), pr.end());
                ID[(size_t) l].insert(ID[(size_t) l].end(), ids.begin(), ids.end());
                WT[(size_t) l].insert(WT[(size_t) l].end(), wts.begin(), wts.end());
            }
            if (!dn.logits(0, n, &logits[(size_t) p0 * (size_t) V])) { std::fprintf(stderr, "logits: %s\n", dn.last_error().c_str()); return 1; }
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::string dir = out + "/" + m;
        ::mkdir(out.c_str(), 0755);
        ::mkdir(dir.c_str(), 0755);
        for (int64_t l = 0; l < nl; ++l) {
            save(dir + "/x" + std::to_string(l) + ".f32", X[(size_t) l].data(), X[(size_t) l].size() * 4);
            save(dir + "/pre" + std::to_string(l) + ".f32", P[(size_t) l].data(), P[(size_t) l].size() * 4);
            save(dir + "/ids" + std::to_string(l) + ".i32", ID[(size_t) l].data(), ID[(size_t) l].size() * 4);
            save(dir + "/wts" + std::to_string(l) + ".f32", WT[(size_t) l].data(), WT[(size_t) l].size() * 4);
        }
        save(dir + "/logits.f32", logits.data(), logits.size() * 4);
        std::printf("%s: %d tokens, %lld layers in %.1f s -> %s (D %lld HC %lld)\n", m.c_str(), S, (long long) nl, sec, dir.c_str(),
                    (long long) D, (long long) HC);
    }
    return 0;
}
