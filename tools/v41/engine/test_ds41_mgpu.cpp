#include "ds41_mgpu.hpp"

#include <cstdio>
#include <limits>

using namespace strata::v41;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, err.c_str()); return 1; } } while (0)

int main() {
    std::string err;
    Ds41MoeGeom g;
    g.n_embd = g.n_ff = 256;
    g.n_layers = 7;
    g.n_experts = 4;
    g.top_k = 2;
    g.gu_type = g.d_type = 10;
    Ds41MultiGpuConfig c;
    c.devices = {0, 1, 2, 3};
    c.device_bytes.assign(4, size_t(1) << 30);
    Ds41MultiGpuPlan p;
    REQUIRE(ds41_mgpu_plan(g, c, 4, p, err));
    REQUIRE(p.layer_owner == std::vector<size_t>({0, 0, 1, 1, 2, 2, 3}));
    REQUIRE(p.owned_layers == std::vector<size_t>({2, 2, 2, 1}));
    REQUIRE(p.expert_bytes == 3 * 256 * 84);
    // Every expert is counted once, never duplicated across the two NVLink pairs.
    size_t total = 0;
    for (size_t bytes : p.cache_bytes) total += bytes;
    REQUIRE(total == size_t(g.n_layers * g.n_experts) * p.expert_stride);
    REQUIRE(p.pinned_bytes < p.cache_bytes[0]); // not a pinned copy of every resident expert
    for (size_t i = 0; i < c.devices.size(); ++i)
        c.device_bytes[i] = p.cache_bytes[i] + p.workspace_bytes;
    REQUIRE(ds41_mgpu_plan(g, c, 4, p, err)); // exact byte limit is inclusive
    --c.device_bytes[3];
    REQUIRE(!ds41_mgpu_plan(g, c, 4, p, err));
    REQUIRE(p.layer_owner.empty()); // failed plans cannot be mistaken for valid ones
    c.device_bytes.assign(4, size_t(1) << 30);
    auto rejects = [&](const Ds41MoeGeom & geometry, const Ds41MultiGpuConfig & config) {
        return !ds41_mgpu_plan(geometry, config, 4, p, err);
    };
    auto bad = c;
    bad.devices[2] = 1;
    REQUIRE(rejects(g, bad));
    bad = c; bad.devices[0] = -1;
    REQUIRE(rejects(g, bad));
    bad = c; bad.devices[3] = 4;
    REQUIRE(rejects(g, bad));
    bad = c; bad.device_bytes.pop_back();
    REQUIRE(rejects(g, bad));
    bad = c; bad.layer_owner = {0, 1};
    REQUIRE(rejects(g, bad));
    bad = c; bad.layer_owner.assign(7, 0);
    REQUIRE(rejects(g, bad));
    bad = c; bad.layer_owner = {0, 1, 2, 3, 4, 0, 1};
    REQUIRE(rejects(g, bad));
    auto geometry = g; geometry.up_types = {10};
    REQUIRE(rejects(geometry, c)); // malformed partial per-layer metadata is rejected safely
    geometry = g; geometry.eclass = {0};
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.n_embd = std::numeric_limits<int64_t>::max();
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.top_k = 9;
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.n_ff = 257;
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.swiglu_limit = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.d_type = 11;
    REQUIRE(rejects(geometry, c));
    geometry = g; geometry.gu_type = geometry.d_type = 11;
    REQUIRE(ds41_mgpu_plan(geometry, c, 4, p, err));
    REQUIRE(p.expert_bytes == 3 * 256 * 110);
    c.devices = {3, 1};
    c.device_bytes.assign(2, size_t(1) << 30);
    c.layer_owner = {1, 1, 0, 0, 1, 0, 0}; // noncontiguous explicit ownership
    REQUIRE(ds41_mgpu_plan(g, c, 4, p, err));
    REQUIRE(p.layer_owner == c.layer_owner);
    Ds41MultiGpuMoe runner;
    Ds41MemoryBlobs source(g, p.expert_bytes);
    // CPU target intentionally refuses execution even on a CUDA-capable host.
    REQUIRE(!runner.init(g, source, c, err));
    REQUIRE(!runner.ready() && runner.plan() == nullptr);
    REQUIRE(!runner.run(0, nullptr, nullptr, nullptr, nullptr, err));
    runner.close();
    runner.close();
    std::puts("multi-GPU CPU planning / fail-closed execution: PASS (no CUDA calls)");
}
