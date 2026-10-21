#include "ds41_mgpu.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace strata::v41;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, err.c_str()); return 1; } } while (0)

// A source exception after the first shard has been loaded exercises transactional
// cleanup on multiple devices, without provoking a disruptive physical OOM.
struct FailingSource : Ds41MemoryBlobs {
    using Ds41MemoryBlobs::Ds41MemoryBlobs;
    void read(int64_t l, int64_t e, uint8_t * dst) const override {
        if (l >= 2) throw std::runtime_error("injected source failure");
        Ds41MemoryBlobs::read(l, e, dst);
    }
};

static double relative(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += double(a[i] - b[i]) * (a[i] - b[i]);
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

int main(int argc, char ** argv) {
    // Refuse to enumerate or create any CUDA context without an explicit device list.
    if (argc != 3 || std::strcmp(argv[1], "--devices")) {
        std::fprintf(stderr, "usage: test_ds41_mgpu_cuda --devices <leased CUDA-visible IDs, e.g. 2,3>\n");
        return 2;
    }
    std::string err, item;
    Ds41MultiGpuConfig config;
    std::istringstream input(argv[2]);
    while (std::getline(input, item, ',')) {
        try {
            size_t end = 0;
            const int id = std::stoi(item, &end);
            if (end != item.size()) return 2;
            config.devices.push_back(id);
        } catch (...) { return 2; }
    }
    if (config.devices.size() < 2) return 2;
    // Synthetic weights/workspace stay below 128 MiB combined (contexts excluded).
    config.device_bytes.assign(config.devices.size(), size_t(64) << 20);
    REQUIRE(cudaSetDevice(config.devices.back()) == cudaSuccess);
    const int caller = config.devices.back();
    auto restored = [&]() {
        int current = -1;
        return cudaGetDevice(&current) == cudaSuccess && current == caller;
    };
    std::mt19937 rng(7301);
    std::normal_distribution<float> normal(0, 0.03f);
    std::vector<size_t> free_before(config.devices.size());
    auto free_bytes = [&]() {
        std::vector<size_t> result;
        for (int device : config.devices) {
            size_t free = 0, total = 0;
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaMemGetInfo(&free, &total) != cudaSuccess)
                return std::vector<size_t>();
            result.push_back(free);
        }
        if (cudaSetDevice(caller) != cudaSuccess) return std::vector<size_t>();
        return result;
    };
    for (int type : {10, 11}) {
        Ds41MoeGeom g;
        g.n_embd = g.n_ff = 256;
        g.n_layers = int64_t(2 * config.devices.size());
        g.n_experts = 4;
        g.top_k = 2;
        g.gu_type = g.d_type = type;
        g.swiglu_limit = 1.0f;
        const auto * traits = ggml_get_type_traits(ggml_type(type));
        const size_t row_bytes = ggml_row_size(ggml_type(type), 256);
        const size_t blob_bytes = 3 * 256 * row_bytes;
        Ds41MemoryBlobs source(g, blob_bytes);
        FailingSource broken(g, blob_bytes);
        std::vector<uint8_t> blob(blob_bytes);
        std::vector<float> row(256);
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_experts; ++e) {
                for (size_t r = 0; r < 3 * 256; ++r) {
                    for (float & v : row) v = normal(rng) * float(1 + l + e);
                    traits->from_float_ref(row.data(), blob.data() + r * row_bytes, 256);
                }
                source.set(l, e, blob.data());
                broken.set(l, e, blob.data());
            }
        Ds41MoeConfig cpu;
        cpu.cpu_only = true;
        cpu.threads = 1;
        Ds41MoeTier reference;
        REQUIRE(reference.init(g, &source, cpu, err));
        Ds41MultiGpuMoe runner;
        // Warm per-device CUDA contexts/modules before checking allocation cleanup.
        REQUIRE(runner.init(g, source, config, err));
        runner.close();
        free_before = free_bytes();
        REQUIRE(free_before.size() == config.devices.size());
        REQUIRE(!runner.init(g, broken, config, err));
        REQUIRE(err.find("injected source failure") != std::string::npos);
        REQUIRE(!runner.ready() && restored());
        const auto free_after_failure = free_bytes();
        REQUIRE(free_after_failure.size() == free_before.size());
        for (size_t i = 0; i < free_before.size(); ++i)
            REQUIRE(free_after_failure[i] + (size_t(1) << 20) >= free_before[i]);
        REQUIRE(runner.init(g, source, config, err));
        REQUIRE(restored());
        const auto * plan = runner.plan();
        size_t requested = 0;
        for (size_t bytes : plan->cache_bytes) requested += bytes + plan->workspace_bytes;
        REQUIRE(requested < (size_t(128) << 20));
        std::vector<float> x(256), got(256), want(256), again(256);
        for (float & v : x) v = normal(rng) * 10;
        double max_error = 0;
        for (int64_t l = g.n_layers - 1; l >= 0; --l) {
            for (bool duplicate : {false, true}) {
                int32_t ids[2] = {int32_t(l % 4), duplicate ? int32_t(l % 4) : int32_t((l + 1) % 4)};
                const float weights[2] = {0.7f, 0.3f};
                REQUIRE(reference.run(l, ids, weights, x.data(), want.data()));
                REQUIRE(runner.run(l, ids, weights, x.data(), got.data(), err));
                REQUIRE(restored());
                REQUIRE(runner.run(l, ids, weights, x.data(), again.data(), err));
                REQUIRE(got == again);
                const double error = relative(got, want);
                max_error = std::max(max_error, error);
                REQUIRE(error < 0.05); // CUDA q8_1 vs CPU q8_K activation rounding
            }
        }
        int32_t invalid[2] = {-1, 0};
        const float weights[2] = {1, 0};
        REQUIRE(!runner.run(0, invalid, weights, x.data(), got.data(), err));
        REQUIRE(!runner.run(g.n_layers, invalid, weights, x.data(), got.data(), err));
        REQUIRE(runner.ready() && restored());
        // Fail preflight after a successful init, then rebuild: no stale shards survive.
        auto too_small = config;
        too_small.device_bytes[0] = 1;
        REQUIRE(!runner.init(g, source, too_small, err));
        REQUIRE(!runner.ready() && restored());
        auto no_free_room = config;
        no_free_room.free_reserve_bytes = std::numeric_limits<size_t>::max();
        REQUIRE(!runner.init(g, source, no_free_room, err));
        REQUIRE(!runner.ready() && restored());
        REQUIRE(runner.init(g, source, config, err));
        REQUIRE(restored());
        runner.close();
        runner.close();
        REQUIRE(restored());
        std::printf("type=%d devices=%zu requested=%zu bytes max CPU-reference rel=%.6g: PASS\n",
                    type, config.devices.size(), requested, max_error);
    }
}
