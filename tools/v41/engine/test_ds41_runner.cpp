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
    {
        Ds41RouteProfile profile;
        profile.layers = 2; profile.experts = 3; profile.top_k = 2;
        profile.prompt.assign(6, 0); profile.decode.assign(6, 0);
        const int32_t ids[] = {2, 0, 1, 2};
        profile.observe(1, 2, ids);
        profile.prefill = false;
        profile.observe(0, 1, ids);
        require(profile.prompt == std::vector<uint64_t>({0,0,0,1,1,2}) &&
                profile.decode == std::vector<uint64_t>({1,0,1,0,0,0}),
                "profile separates token-major prefill and decode counts");
        bool rejected = false;
        const int32_t invalid[] = {-1, 0};
        try { profile.observe(0, 1, invalid); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "profile rejects invalid expert IDs");
        profile.decode[2] = UINT64_MAX;
        rejected = false;
        try { profile.observe(0, 1, ids); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "profile rejects count overflow");
        const std::string path = ".ds41-profile-seed-test";
        profile.metadata = "model abc 123\ngeometry 2 3 2 256 256\n";
        {
            std::ofstream out(path);
            out << "DS41_SEED_V1\n" << profile.metadata << "select 1 2\nselect 0 0\n";
        }
        require(profile.read_seed(path) == std::vector<std::pair<int32_t,int32_t>>({{1,2},{0,0}}),
                "exact seed order round trips");
        {
            std::ofstream out(path, std::ios::app);
            out << "select 1 2\n";
        }
        rejected = false;
        try { profile.read_seed(path); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "seed rejects duplicates");
        {
            std::ofstream out(path);
            out << "DS41_SEED_V1\nmodel def 123\nselect 1 2\n";
        }
        rejected = false;
        try { profile.read_seed(path); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "seed rejects incompatible identity");
        for (const auto & body : {
                profile.metadata,
                profile.metadata + "select -1 0\n",
                profile.metadata + "select 0 3\n",
                profile.metadata + "select 0 0 extra\n",
                profile.metadata + "select 0 0\ngeometry 2 3 2 256 256\n"}) {
            {
                std::ofstream out(path);
                out << "DS41_SEED_V1\n" << body;
            }
            rejected = false;
            try { profile.read_seed(path); } catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "seed rejects empty, out-of-bounds, trailing, or reordered data");
        }
        std::remove(path.c_str());
    }
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
    {
        Ds41MoeRunner seeded;
        std::vector<std::pair<int32_t,int32_t>> selection;
        for (int l = 0; l < count - 1; ++l) selection.emplace_back(l, 7);
        require(seeded.init(g, &source, cfg, devices, 1, error, &selection), error);
        require(seeded.resident() == count - 1 &&
                seeded.arena_experts() == 8 * count - (count - 1) && seeded.file_tier() == 0,
                "explicit seed quotas include a zero-resident card and complete pinned tail");
        for (int l = 0; l < count; ++l) for (int e = 0; e < 8; ++e)
            require(seeded.resident(l, e) == (l < count - 1 && e == 7), "exact selected IDs resident");
        const int32_t route[] = {7, 0};
        const float w[] = {0.5f, 0.5f};
        std::vector<float> input(size_t(g.n_embd), 0.02f), actual(size_t(g.n_embd)), baseline(size_t(g.n_embd));
        float * device_input = nullptr, * device_output = nullptr, * landing = nullptr;
        cudaStream_t producer = nullptr, consumer = nullptr;
        const size_t bytes = input.size() * sizeof(float);
        require(cudaMalloc(reinterpret_cast<void **>(&device_input), bytes) == cudaSuccess &&
                cudaMalloc(reinterpret_cast<void **>(&device_output), bytes) == cudaSuccess &&
                cudaMallocHost(reinterpret_cast<void **>(&landing), bytes) == cudaSuccess &&
                cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess &&
                cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess, "seeded device buffers");
        for (int l = 0; l < count; ++l) {
            require(seeded.run(l, route, w, input.data(), actual.data()) &&
                    sharded.run(l, route, w, input.data(), baseline.data()), "seeded output execution");
            require(actual == baseline, "exact seed preserves resident/pinned numeric output");
            std::memcpy(landing, input.data(), bytes);
            require(cudaMemcpyAsync(device_input, landing, bytes, cudaMemcpyHostToDevice, producer) == cudaSuccess &&
                    cudaStreamSynchronize(producer) == cudaSuccess, "seeded ready nonblocking producer");
            require(seeded.run_to_dev(l, route, w, nullptr, device_output, device_input), "seeded device decode");
            require(cudaMemcpyAsync(landing, device_output, bytes, cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
                    cudaStreamSynchronize(consumer) == cudaSuccess &&
                    std::memcmp(landing, baseline.data(), bytes) == 0, "seeded independent consumer exact output");
            int current = -1;
            require(cudaGetDevice(&current) == cudaSuccess && current == count - 1, "seeded device restoration");
            for (auto & v : input) v *= -1.5f;
        }
        const char* direct = std::getenv("STRATA_MOE_DEVICE_INPUT");
        if (direct && direct[0] == '1' && !std::getenv("STRATA_X_HOST") &&
            !std::getenv("DS41_CHECK_POOL_BATCH") && !std::getenv("DS4_CHECK_GPU"))
            require(seeded.stats().device_activation_calls == count, "seeded pinned tail uses device activations");
        cudaFree(device_input); cudaFree(device_output); cudaFreeHost(landing);
        cudaStreamDestroy(producer); cudaStreamDestroy(consumer);
        seeded.release_cache();
        require(seeded.resident() == count - 1, "selected cache persists");
        seeded.close();
        selection.emplace_back(0, 7);
        require(!seeded.init(g, &source, cfg, devices, 1, error, &selection), "duplicate seed rejected");
        selection.clear();
        for (int e = 0; e < 8; ++e) selection.emplace_back(0, e);
        require(!seeded.init(g, &source, cfg, devices, 1, error, &selection) && !seeded.sharded(),
                "explicit selection larger than per-card slots rejected");
    }
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
    cudaStream_t producer = nullptr;
    require(cudaMalloc(reinterpret_cast<void **>(&output), x.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&input), x.size() * sizeof(float)) == cudaSuccess &&
            cudaMallocHost(reinterpret_cast<void **>(&landing), x.size() * sizeof(float)) == cudaSuccess &&
            cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess, "caller buffers");
    auto upload = [&](const float* data) {
        std::memcpy(landing, data, x.size() * sizeof(float));
        require(cudaMemcpyAsync(input, landing, x.size() * sizeof(float), cudaMemcpyHostToDevice, producer) == cudaSuccess &&
                cudaStreamSynchronize(producer) == cudaSuccess, "ready independent producer stream");
    };
    auto download = [&](size_t size) {
        require(cudaMemcpyAsync(landing, output, size * sizeof(float), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
                cudaStreamSynchronize(consumer) == cudaSuccess, "independent consumer stream");
        require(std::memcmp(landing, expected.data(), size * sizeof(float)) == 0, "bit-exact single-GPU reference");
    };
    for (int caller = 0; caller < count; ++caller) {
        require(cudaSetDevice(caller) == cudaSuccess, "select transfer caller");
        float * caller_input = nullptr, * caller_output = nullptr, * caller_landing = nullptr;
        cudaStream_t caller_producer = nullptr, caller_consumer = nullptr;
        const size_t bytes = size_t(g.n_embd) * sizeof(float);
        require(cudaMalloc(reinterpret_cast<void **>(&caller_input), bytes) == cudaSuccess &&
                cudaMalloc(reinterpret_cast<void **>(&caller_output), bytes) == cudaSuccess &&
                cudaMallocHost(reinterpret_cast<void **>(&caller_landing), bytes) == cudaSuccess &&
                cudaStreamCreateWithFlags(&caller_producer, cudaStreamNonBlocking) == cudaSuccess &&
                cudaStreamCreateWithFlags(&caller_consumer, cudaStreamNonBlocking) == cudaSuccess, "all-pairs buffers");
        for (int l = 0; l < count; ++l) {
            require(sharded.run(l, ids.data(), weights.data(), x.data(), actual.data()), "all-pairs reference");
            std::memcpy(caller_landing, x.data(), bytes);
            require(cudaMemcpyAsync(caller_input, caller_landing, bytes, cudaMemcpyHostToDevice, caller_producer) == cudaSuccess &&
                    cudaStreamSynchronize(caller_producer) == cudaSuccess, "all-pairs ready producer");
            require(sharded.run_to_dev(l, ids.data(), weights.data(), nullptr, caller_output, caller_input),
                    "all-pairs device decode");
            require(cudaMemcpyAsync(caller_landing, caller_output, bytes, cudaMemcpyDeviceToHost, caller_consumer) == cudaSuccess &&
                    cudaStreamSynchronize(caller_consumer) == cudaSuccess &&
                    std::memcmp(caller_landing, actual.data(), bytes) == 0, "all-pairs exact consumer");
            int current = -1;
            require(cudaGetDevice(&current) == cudaSuccess && current == caller, "all-pairs context restoration");
        }
        cudaFree(caller_input); cudaFree(caller_output); cudaFreeHost(caller_landing);
        cudaStreamDestroy(caller_producer); cudaStreamDestroy(caller_consumer);
    }
    require(cudaSetDevice(count - 1) == cudaSuccess, "restore primary test caller");
    for (int l = 0; l < count; ++l) {
        require(reference.run_to_dev(l, ids.data(), weights.data(), x.data(), output), "reference decode");
        require(cudaMemcpy(expected.data(), output, g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference read");
        require(sharded.run_to_dev(l, ids.data(), weights.data(), x.data(), output), "sharded decode");
        download(size_t(g.n_embd));
        upload(x.data());
        require(sharded.run_to_dev(l, ids.data(), weights.data(), nullptr, output, input), "sharded device input");
        download(size_t(g.n_embd));
        auto changed = x;
        for (auto & v : changed) v *= -0.5f;
        require(reference.run_to_dev(l, ids.data(), weights.data(), changed.data(), output), "changed reference decode");
        require(cudaMemcpy(expected.data(), output, g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
                "changed reference read");
        require(std::any_of(expected.begin(), expected.begin() + g.n_embd, [](float v) { return v != 0; }),
                "nonzero decode reference");
        upload(changed.data());
        require(sharded.run_to_dev(l, ids.data(), weights.data(), nullptr, output, input), "changed device input");
        download(size_t(g.n_embd));
        require(reference.run_multi_to_dev(l, n, ids.data(), weights.data(), x.data(), output), "reference multi");
        require(cudaMemcpy(expected.data(), output, x.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference multi read");
        require(sharded.run_multi_to_dev(l, n, ids.data(), weights.data(), x.data(), output), "sharded multi");
        download(x.size());
        require(reference.run_chunk_dev(l, n, ids.data(), weights.data(), x.data(), output), "reference chunk");
        require(cudaMemcpy(expected.data(), output, x.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "reference chunk read");
        require(sharded.run_chunk_dev(l, n, ids.data(), weights.data(), x.data(), output), "sharded chunk");
        download(x.size());
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
    const char* direct = std::getenv("STRATA_MOE_DEVICE_INPUT");
    const bool host_diagnostic = std::getenv("STRATA_X_HOST") || std::getenv("DS41_CHECK_POOL_BATCH") ||
                                 std::getenv("DS4_CHECK_GPU");
    require(stats.device_activation_calls == ((direct && direct[0] == '1' && !host_diagnostic) ? (2 + count) * count : 0),
            "opt-in selects device activation path only");
    sharded.release_cache();
    require(sharded.resident() == 6 * count, "cache persists across requests");
    sharded.reset_stats();
    require(sharded.stats().lookups() == 0, "aggregate reset");
    sharded.close();
    require(!sharded.sharded(), "close");
    cudaFree(output); cudaFree(input); cudaFreeHost(landing); cudaStreamDestroy(consumer); cudaStreamDestroy(producer);
    std::printf("%d-GPU runner: mixed Q2_K/Q3_K, pinned tail, decode/multi/chunk, changed input, "
                "independent stream, device restoration and lifecycle PASS\n", count);
    return 0;
#endif
}
