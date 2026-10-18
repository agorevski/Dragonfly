// tools/v41/engine/test_ds41_dspark.cpp - the engine's DSpark draft stack on the offline harness's records.
//
//   test_ds41_dspark --sidecar V41-MTP-Vision.gguf -m V41-GSQ.gguf --main main.bin --gen gen.txt [--max N] [--dump F]
//
// main.bin / gen.txt = what tools/v41/dspark_accept.py reads (engine STRATA_DUMP_MAIN records {int32 token, int32 pos,
// 3 x n_embd f32}: the first 128 are the prompt tail that seeds the window, the rest decode steps).  The same loop as
// the harness: step j appends record j's main_hidden at its position and drafts with bonus = the next token; drafts are
// scored against the tokens that followed (accepted length = the matching prefix).  --dump writes one line per scored
// step: "j pos d0..d4 c0..c4" (compare with dspark_accept.py --dump-drafts).
#include "ds41_dspark.hpp"
#include "ds41_dense.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int rollback_test(const std::string & model) {
    Ds41DenseConfig cfg;
    cfg.max_pass = 17; cfg.ctx_max = 1024; cfg.n_threads = 2;
    cfg.engram_threads = 2; cfg.act_quant = false;
    Ds41Dense speculative, reference;
    std::string err;
    require(speculative.init(model, cfg, err), err);
    require(reference.init(model, cfg, err), err);
    const int dim = (int) reference.geom().n_embd, vocab = (int) reference.geom().vocab;
    auto pass = [&](Ds41Dense & dense, const std::vector<int> & tokens, int pos) {
        const int n = (int) tokens.size();
        require(dense.begin(tokens.data(), n, pos), dense.last_error());
        std::vector<int> experts((size_t) n * dense.geom().n_used);
        std::vector<float> weights(experts.size()), routed((size_t) n * dim, 0.0f);
        for (int l = 0; l < dense.n_layer(); ++l) {
            const float * fn = nullptr;
            require(dense.attn_router(l, experts.data(), weights.data(), &fn), dense.last_error());
            require(dense.finish(l, routed.data()), dense.last_error());
        }
        std::vector<float> logits((size_t) n * vocab);
        require(dense.logits(0, n, logits.data()), dense.last_error());
        return logits;
    };
    int cases = 0;
    for (int seed : {24, 25})
        for (int depth = 1; depth <= 5; ++depth)
            for (int accepted = 0; accepted <= depth; ++accepted) {
                // Exceed the SWA ring size, then overwrite a rejected verify tail at both compressor parities.
                for (int pos = 0; pos < seed; ++pos) {
                    pass(speculative, {pos % vocab}, pos);
                    pass(reference, {pos % vocab}, pos);
                }
                std::vector<int> verify;
                for (int i = 0; i <= depth; ++i) verify.push_back((seed + i) % vocab);
                pass(speculative, verify, seed);
                for (int i = 0; i <= accepted; ++i) pass(reference, {verify[i]}, seed + i);
                const int next = seed + accepted + 1;
                const int replacement = (next + 1) % vocab;
                const bool refeed = ds41_dspark_needs_refeed(next, accepted < depth);
                std::vector<int> restart;
                if (refeed) restart.push_back(verify[accepted]);
                restart.push_back(replacement);
                const auto actual = pass(speculative, restart, next - (refeed ? 1 : 0));
                const auto expected = pass(reference, {replacement}, next);
                for (int v = 0; v < vocab; ++v) {
                    const float value = actual[actual.size() - vocab + v];
                    require(std::isfinite(value) && std::fabs(value - expected[v]) < 2.e-5f,
                            "rollback changed logits: seed=" + std::to_string(seed) +
                            " depth=" + std::to_string(depth) + " accepted=" + std::to_string(accepted));
                }
                ++cases;
            }
    std::printf("DSpark CPU rollback: %d rejection/parity cases matched sequential logits with Engram, SWA and shared ratio-2 KV/index\n", cases);
    return 0;
}

// Deliberately tiny invented weights: this exercises graph/state mechanics, not checkpoint compatibility.
void fixture(const std::string & path, bool side, int defect = 0) {
    gguf_context * file = gguf_init_empty();
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, false });
    gguf_set_val_str(file, "general.architecture", "deepseek41");
    if (!side) {
        for (const auto & kv : std::initializer_list<std::pair<const char *, uint64_t>>{
                 { "hidden_size", 32 }, { "num_hidden_layers", 40 }, { "num_attention_heads", 1 },
                 { "head_dim", 32 }, { "qk_rope_head_dim", 16 }, { "q_lora_rank", 32 },
                 { "o_lora_rank", 32 }, { "o_groups", 1 }, { "sliding_window", 128 },
                 { "n_routed_experts", 4 }, { "num_experts_per_tok", 3 },
                 { "moe_intermediate_size", 32 }, { "n_shared_experts", 1 }, { "hc_mult", 4 },
                 { "hc_sinkhorn_iters", 4 }, { "index_n_heads", 1 }, { "index_head_dim", 32 },
                 { "index_topk", 1 }, { "candidate_source_layer_id", 0 }, { "candidate_topk_blocks", 1 },
                 { "candidate_block_size", 2 }, { "vocab_size", 8 }, { "max_position_embeddings", 1024 },
                 { "engram.pad_id", 0 }, { "engram.compressed_vocab_size", 8 } })
            if (defect != 4 || std::strcmp(kv.first, "num_attention_heads") != 0)
                gguf_set_val_u64(file, ("deepseek41." + std::string(kv.first)).c_str(), kv.second);
        for (const auto & kv : std::initializer_list<std::pair<const char *, double>>{
                 { "routed_scaling_factor", 1 }, { "hc_eps", 1.e-6 }, { "rms_norm_eps", 1.e-6 },
                 { "rope_theta", 10000 }, { "compress_rope_theta", 10000 }, { "rope_scaling.factor", 1 },
                 { "rope_scaling.beta_fast", 32 }, { "rope_scaling.beta_slow", 1 },
                 { "rope_scaling.original_max_position_embeddings", 1024 }, { "swiglu_limit", 10 } })
            gguf_set_val_f64(file, ("deepseek41." + std::string(kv.first)).c_str(), kv.second);
        gguf_set_val_bool(file, "deepseek41.norm_topk_prob", true);
        gguf_set_val_str(file, "deepseek41.scoring_func", "sqrtsoftplus");
        gguf_set_val_str(file, "deepseek41.config",
                         "{\"engram_n_heads\":1,\"engram_head_dim\":32,\"engram_max_ngram_size\":3}");
        const uint64_t zero[40] = {};
        const uint64_t tokens[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        gguf_set_arr_data(file, "deepseek41.compress_ratios", GGUF_TYPE_UINT64, zero, 40);
        gguf_set_arr_data(file, "deepseek41.engram.token_map", GGUF_TYPE_UINT64, tokens, 8);
        for (const char * key : { "kv_source_layer_ids", "index_source_layer_ids", "engram.layer_ids",
                                 "engram.rows", "engram.primes", "engram.multipliers" })
            gguf_set_arr_data(file, ("deepseek41." + std::string(key)).c_str(), GGUF_TYPE_UINT64, zero, 0);
    }
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape, bool norm = false, bool half = false) {
        if (defect == 1 && name == "mtp.1.attn.wkv.weight") return;
        std::vector<int64_t> dims(shape);
        if (defect == 2 && name == "mtp.2.markov_head.head.weight") dims[1] = 7;
        const ggml_type type = defect == 3 && name == "mtp.2.markov_head.embed.weight" ? GGML_TYPE_I32 :
                               defect == 5 && name == "mtp.0.attn.attn_sink" ? GGML_TYPE_F16 :
                               half ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor * t = ggml_new_tensor(ctx, type, (int) dims.size(), dims.data());
        ggml_set_name(t, name.c_str());
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            const float value = norm ? 1.0f : 0.02f * std::sin((float) (i + 1));
            if (type == GGML_TYPE_F16) ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(value);
            else ((float *) t->data)[i] = value;
        }
        gguf_add_tensor(file, t);
    };
    if (!side) {
        add("output.weight", { 32, 8 });
        add("token_embd.weight", { 32, 8 }, false, true);
    } else {
        add("mtp.0.main_proj.weight", { 96, 32 });
        add("mtp.0.main_norm.weight", { 32 }, true);
        for (int b = 0; b < 3; ++b) {
            const std::string p = "mtp." + std::to_string(b) + ".";
            for (const char * site : { "attn", "ffn" }) {
                add(p + "hc_" + site + "_fn", { 128, 24 });
                add(p + "hc_" + site + "_scale", { 3 });
                add(p + "hc_" + site + "_base", { 24 });
                add(p + site + "_norm.weight", { 32 }, true);
            }
            for (const char * name : { "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                                      "attn.wo_a.weight", "attn.wo_b.weight",
                                      "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
                                      "ffn.shared_experts.w3.weight" }) add(p + name, { 32, 32 });
            add(p + "attn.q_norm.weight", { 32 }, true);
            add(p + "attn.kv_norm.weight", { 32 }, true);
            add(p + "attn.attn_sink", { 1 });
            add(p + "ffn.gate.weight", { 32, 4 });
            add(p + "ffn.gate.bias", { 4 });
            for (const char * role : { "w1", "w2", "w3" })
                add(p + "ffn.experts." + role + ".weight", { 32, 32, 4 });
        }
        add("mtp.2.norm.weight", { 32 }, true);
        add("mtp.2.markov_head.embed.weight", { 4, 8 });
        add("mtp.2.markov_head.head.weight", { 4, 8 });
        add("mtp.2.confidence_head.proj.weight", { 36, 1 });
    }
    const bool written = gguf_write_to_file(file, path.c_str(), false);
    gguf_free(file);
    ggml_free(ctx);
    require(written, "cannot write fixture " + path);
}

int self_test(const std::string & directory) {
    const std::filesystem::path dir(directory);
    require(std::filesystem::create_directory(dir), "fixture directory must not already exist");
    ggml_backend_t be = ggml_backend_cpu_init();
    require(be != nullptr, "no CPU backend");
    ggml_backend_cpu_set_n_threads(be, 2);
    try {
        fixture((dir / "main.gguf").string(), false);
        fixture((dir / "side.gguf").string(), true);
        fixture((dir / "missing.gguf").string(), true, 1);
        fixture((dir / "mismatch.gguf").string(), true, 2);
        fixture((dir / "integer.gguf").string(), true, 3);
        fixture((dir / "metadata.gguf").string(), false, 4);
        fixture((dir / "sink-half.gguf").string(), true, 5);
        std::filesystem::copy_file(dir / "side.gguf", dir / "truncated.gguf");
        std::filesystem::resize_file(dir / "truncated.gguf", std::filesystem::file_size(dir / "truncated.gguf") - 32);
        Ds41DSparkConfig cfg;
        cfg.backend = be; cfg.main_gguf = (dir / "main.gguf").string();
        cfg.sidecar = (dir / "side.gguf").string(); cfg.noise_id = 7;
        std::string err;
        Ds41DSpark ds;
        float hidden[8 * 96];
        for (size_t i = 0; i < 8 * 96; ++i) hidden[i] = 0.1f * std::cos((float) i);
        int ids[5], pos[8] = {}; float conf[5];
        require(!ds.step(hidden, pos, 1, 0, ids, conf), "step before init accepted");
        for (int defect = 0; defect < 12; ++defect) {
            auto bad = cfg;
            if (defect == 0) bad.sidecar = (dir / "absent.gguf").string();
            if (defect == 1) bad.sidecar = (dir / "missing.gguf").string();
            if (defect == 2) bad.sidecar = (dir / "mismatch.gguf").string();
            if (defect == 3) bad.noise_id = 8;
            if (defect == 4) bad.block = 6;
            if (defect == 5) bad.max_append = 129;
            if (defect == 6) bad.n_used = 5;
            if (defect == 7) bad.sidecar = (dir / "truncated.gguf").string();
            if (defect == 8) bad.sidecar = (dir / "integer.gguf").string();
            if (defect == 9) bad.main_gguf = (dir / "metadata.gguf").string();
            if (defect == 10) bad.sidecar = (dir / "sink-half.gguf").string();
            if (defect == 11) bad.backend = nullptr;
            require(!ds.init(bad, err) && !err.empty(), "invalid sidecar/config accepted");
            if (defect == 1) require(err.find("mtp.1.attn.wkv.weight") != std::string::npos, err);
            if (defect == 2) require(err.find("mtp.2.markov_head.head.weight") != std::string::npos, err);
            if (defect == 7) require(err.find("incomplete tensor") != std::string::npos, err);
            if (defect == 8) require(err.find("unsupported type") != std::string::npos, err);
            if (defect == 9) require(err.find("num_attention_heads") != std::string::npos, err);
            if (defect == 10) require(err.find("unsupported type") != std::string::npos, err);
            require(!ds.step(hidden, pos, 1, 0, ids, conf), "failed init left a runnable stack");
        }
        require(ds.init(cfg, err), err);
        require(!ds.step(hidden, pos, 1, 8, ids, conf), "out-of-vocabulary bonus accepted");
        require(!ds.step(nullptr, pos, 1, 0, ids, conf), "null input accepted");
        require(!ds.step(hidden, pos, 1, 0, nullptr, conf), "null output accepted");
        pos[0] = -1;
        require(!ds.step(hidden, pos, 1, 0, ids, conf), "negative position accepted");
        pos[0] = std::numeric_limits<int>::max() - 1;
        require(!ds.step(hidden, pos, 1, 0, ids, conf), "draft position overflow accepted");
        pos[0] = 0; pos[1] = 128;
        require(!ds.step(hidden, pos, 2, 0, ids, conf), "aliased ring writes accepted");
        pos[1] = 0;
        require(!ds.step(hidden, pos, 2, 0, ids, conf), "duplicate positions accepted");
        const int drafts[5] = { 2, 3, 4, 5, 6 };
        std::mt19937_64 rng(1017);
        int replacement = -1;
        require(ds41_dspark_spec_accept({ { 2, 1.0 } }, 2, rng, replacement), "certain draft was rejected");
        require(!ds41_dspark_spec_accept({ { 2, 1.0 } }, 3, rng, replacement) && replacement == 2,
                "filtered-out draft was not replaced by target token");
        const std::vector<std::pair<int, double>> distribution = { { 0, 0.2 }, { 1, 0.3 }, { 2, 0.5 } };
        for (int draft = 0; draft <= 3; ++draft) {
            int counts[3] = {};
            for (int trial = 0; trial < 40000; ++trial) {
                const bool accepted = ds41_dspark_spec_accept(distribution, draft, rng, replacement);
                const int token = accepted ? draft : replacement;
                require(token >= 0 && token < 3 && (accepted || token != draft), "invalid rejection sample");
                ++counts[token];
            }
            for (int token = 0; token < 3; ++token)
                require(std::fabs(counts[token] / 40000.0 - distribution[token].second) < 0.015,
                        "speculative sampling changed the target distribution");
        }
        for (int k = 0; k <= 5; ++k)
            for (int accepted = 0; accepted <= k; ++accepted) {
                int verified[6] = { 2, 3, 4, 5, 6, 7 };
                if (accepted < k) verified[accepted] = 0;
                require(ds41_dspark_accepted_prefix(drafts, verified, k) == accepted, "accepted non-prefix draft");
                // The verifier fed [bonus, drafts...]; only bonus + the accepted prefix enter the draft window.
                ds.reset();
                for (int i = 0; i <= accepted; ++i) pos[i] = i;
                require(ds.step(hidden, pos, accepted + 1, verified[accepted], ids, conf), ds.last_error());
                pos[0] = accepted;
                require(!ds.step(hidden, pos, 1, 0, ids, conf), "rewind accepted without reset");
                pos[0] = accepted + 1;
                require(ds.step(hidden, pos, 1, -1, nullptr, nullptr), ds.last_error());
            }
        // Change graph shapes repeatedly past the ring wrap, then compare reset/replay to a fresh stack.
        ds.reset();
        Ds41DSpark fresh;
        require(fresh.init(cfg, err), err);
        for (int p = 0; p < 144; p += 8) {
            for (int i = 0; i < 8; ++i) pos[i] = p + i;
            require(ds.step(hidden, pos, 8, -1, nullptr, nullptr), ds.last_error());
            for (int i = 0; i < 8; ++i)
                require(fresh.step(hidden + i * 96, pos + i, 1, -1, nullptr, nullptr), fresh.last_error());
        }
        pos[0] = 144;
        require(ds.step(hidden, pos, 1, 2, ids, conf), ds.last_error());
        int reference[5]; float reference_conf[5];
        require(fresh.step(hidden, pos, 1, 2, reference, reference_conf), fresh.last_error());
        for (int i = 0; i < 5; ++i)
            require(ids[i] == reference[i] && std::isfinite(conf[i]) &&
                    std::fabs(conf[i] - reference_conf[i]) < 1.e-6f, "batched append changed the accepted window");
        ds.reset();
        fresh.reset();
        pos[0] = 0;
        require(ds.step(hidden, pos, 1, 2, ids, conf), ds.last_error());
        require(fresh.step(hidden, pos, 1, 2, reference, reference_conf), fresh.last_error());
        for (int i = 0; i < 5; ++i)
            require(ids[i] == reference[i] && std::isfinite(conf[i]) &&
                    std::fabs(conf[i] - reference_conf[i]) < 1.e-6f, "reset leaked stale ring state");
        require(ds.init(cfg, err), "reinitialization failed: " + err);
        require(ds.step(hidden, pos, 1, 2, ids, conf), ds.last_error());
    } catch (...) {
        ggml_backend_free(be);
        std::filesystem::remove_all(dir);
        throw;
    }
    ggml_backend_free(be);
    std::filesystem::remove_all(dir);
    std::puts("DSpark CPU synthetic: prefix rejection, sampling, accepted-window state, reset/reinit and sidecar failures passed");
    return 0;
}
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--rollback-test") == 0) {
        try { return rollback_test(argv[2]); }
        catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    }
    if (argc == 3 && std::strcmp(argv[1], "--self-test") == 0) {
        try { return self_test(argv[2]); }
        catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    }
    std::string side, model, mainf, genf, dump;
#ifdef DS41_HAVE_CUDA
    std::string backend = "cuda";
#else
    std::string backend = "cpu";
#endif
    int max = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--sidecar") side = nx();
        else if (a == "-m") model = nx();
        else if (a == "--main") mainf = nx();
        else if (a == "--gen") genf = nx();
        else if (a == "--max") max = std::atoi(nx().c_str());
        else if (a == "--dump") dump = nx();
        else if (a == "--backend") backend = nx();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (side.empty() || model.empty() || mainf.empty() || genf.empty()) {
        std::fprintf(stderr, "usage: test_ds41_dspark --sidecar F -m F --main main.bin --gen gen.txt "
                             "[--max N] [--dump F] [--backend cpu|cuda]\n");
        return 2;
    }
    ggml_backend_t be = nullptr;
#ifdef DS41_HAVE_CUDA
    if (backend == "cuda") be = ggml_backend_cuda_init(0);
#endif
    if (backend == "cpu") be = ggml_backend_cpu_init();
    if (!be) { std::fprintf(stderr, "no requested backend: %s\n", backend.c_str()); return 1; }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_owner(be, ggml_backend_free);
    Ds41DSparkConfig cfg;
    cfg.backend = be;
    cfg.sidecar = side;
    cfg.main_gguf = model;
    Ds41DSpark ds;
    std::string err;
    if (!ds.init(cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int64_t MD = ds.main_dim();
    std::fprintf(stderr, "DSpark stack: %.2f GB of weights on the device\n", ds.weight_bytes() / 1e9);

    // records
    std::ifstream f(mainf, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t rs = 8 + (size_t) MD * 4;
    if (raw.empty() || raw.size() % rs) { std::fprintf(stderr, "%s: not a whole number of records\n", mainf.c_str()); return 1; }
    const int n = (int) (raw.size() / rs);
    std::vector<int> tok((size_t) n), pos((size_t) n);
    for (int j = 0; j < n; ++j) {
        std::memcpy(&tok[(size_t) j], raw.data() + (size_t) j * rs, 4);
        std::memcpy(&pos[(size_t) j], raw.data() + (size_t) j * rs + 4, 4);
    }
    std::vector<int> gen;
    {
        std::ifstream g(genf);
        std::string w;
        while (g >> w) {
            bool num = !w.empty();
            for (size_t i = (w[0] == '-'); i < w.size(); ++i) num = num && std::isdigit((unsigned char) w[i]);
            if (num) gen.push_back(std::atoi(w.c_str()));
        }
    }
    std::vector<int> seq(tok);
    if (gen.empty()) { std::fprintf(stderr, "%s: no generated token IDs\n", genf.c_str()); return 1; }
    seq.push_back(gen.back());
    const int BS = cfg.block, n_seed = std::min(128, n);
    FILE * df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "w");
    if (!dump.empty() && !df) { std::fprintf(stderr, "cannot open dump %s\n", dump.c_str()); return 1; }
    std::vector<int> acc;
    int first_ok = 0;
    double ms_sum = 0;
    int ms_n = 0;
    for (int j = 0; j < n; ++j) {
        const float * mh = (const float *) (raw.data() + (size_t) j * rs + 8);
        const bool score = j >= n_seed && j + 1 + BS < (int) seq.size();
        int ids[8];
        float conf[8];
        if (!ds.step(mh, &pos[(size_t) j], 1, score ? seq[(size_t) j + 1] : -1, ids, conf)) {
            std::fprintf(stderr, "step %d: %s\n", j, ds.last_error().c_str());
            return 1;
        }
        if (!score) continue;
        ms_sum += ds.last_ms();
        ++ms_n;
        int k = 0;
        while (k < BS && ids[k] == seq[(size_t) j + 2 + k]) ++k;
        acc.push_back(k);
        first_ok += ids[0] == seq[(size_t) j + 2];
        if (df) {
            std::fprintf(df, "%d %d", j, pos[(size_t) j]);
            for (int i = 0; i < BS; ++i) std::fprintf(df, " %d", ids[i]);
            for (int i = 0; i < BS; ++i) std::fprintf(df, " %.4f", conf[i]);
            std::fprintf(df, "\n");
        }
        if (max && (int) acc.size() >= max) break;
    }
    if (df) std::fclose(df);
    double mean = 0;
    std::vector<int> hist((size_t) BS + 1, 0);
    for (int a : acc) { mean += a; ++hist[(size_t) a]; }
    mean /= std::max<size_t>(1, acc.size());
    std::printf("engine DSpark over %zu steps: mean accepted %.3f -> tokens per verify pass %.3f; first draft %.1f%%; "
                "draft %.2f ms/step\n", acc.size(), mean, 1 + mean, 100.0 * first_ok / std::max<size_t>(1, acc.size()),
                ms_sum / std::max(1, ms_n));
    std::printf("  distribution:");
    for (int k = 0; k <= BS; ++k) std::printf(" %d:%.1f%%", k, 100.0 * hist[(size_t) k] / std::max<size_t>(1, acc.size()));
    std::printf("\n");
    for (int d = 1; d <= BS; ++d) {
        double t = 0;
        for (int a : acc) t += std::min(a, d);
        std::printf("  verify depth %d: tokens per pass %.3f\n", d, 1 + t / std::max<size_t>(1, acc.size()));
    }
    return 0;
}
