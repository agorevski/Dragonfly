#pragma once

#include "ds41_moe.hpp"
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace strata::v41 {

// A versioned aggregate format, deliberately containing no tokens or prompt text.
class Ds41RouteProfile {
public:
    bool prefill = true;
    std::string metadata;
    std::vector<uint64_t> prompt, decode;
    int64_t experts = 0, layers = 0, top_k = 0;

    void configure(const Ds41MoeGeom & g, const std::string & model,
                   const std::vector<int> & devices) {
        if (g.n_layers <= 0 || g.n_experts <= 0 || g.top_k <= 0 || g.top_k > g.n_experts ||
            g.n_layers > 1000000 / g.n_experts)
            throw std::runtime_error("route profile: invalid geometry");
        std::ifstream in(model, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("route profile: cannot open model identity");
        const auto bytes = in.tellg();
        if (bytes <= 0) throw std::runtime_error("route profile: empty model identity");
        in.seekg(0);
        uint64_t hash = 14695981039346656037ull;
        char c;
        for (int i = 0; i < 65536 && in.get(c); ++i) {
            hash ^= static_cast<unsigned char>(c);
            hash *= 1099511628211ull;
        }
        std::ostringstream s;
        s << "model " << std::hex << hash << std::dec << ' ' << bytes << '\n';
        s << "geometry " << g.n_layers << ' ' << g.n_experts << ' ' << g.top_k
          << ' ' << g.n_embd << ' ' << g.n_ff << '\n';
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.routed(l)) continue;
            for (int64_t e = 0; e < g.n_experts; ++e) {
                cpu::NativeFmt f;
                std::string error;
                if (!ds41_moe_class_layout(g, g.cls(l, e), f, error))
                    throw std::runtime_error("route profile: " + error);
                const uint64_t blob = f.bytes;
                if (!blob || blob > UINT64_MAX - 255)
                    throw std::runtime_error("route profile: invalid blob size");
                s << "expert " << l << ' ' << e << ' '
                  << (devices.empty() ? 0 : devices[size_t(l) * devices.size() / size_t(g.n_layers)])
                  << ' ' << blob << ' ' << ((blob + 255) / 256 * 256) << ' '
                  << (g.per_expert() ? g.cls_gu[size_t(g.cls(l,e))] : g.gate_type(l)) << ' '
                  << (g.per_expert() ? g.cls_gu[size_t(g.cls(l,e))] : g.up_type(l)) << ' '
                  << (g.per_expert() ? g.cls_d[size_t(g.cls(l,e))] : g.down_type(l)) << '\n';
            }
        }
        metadata = s.str();
        experts = g.n_experts; layers = g.n_layers; top_k = g.top_k;
        prompt.assign(size_t(layers * experts), 0);
        decode = prompt;
    }

    void observe(int64_t l, int n, const int32_t * ids) {
        if (prompt.empty()) return;
        if (l < 0 || l >= layers || n <= 0 || !ids)
            throw std::runtime_error("route profile: invalid route dimensions");
        auto & counts = prefill ? prompt : decode;
        for (int64_t i = 0; i < int64_t(n) * top_k; ++i) {
            if (ids[i] < 0 || ids[i] >= experts)
                throw std::runtime_error("route profile: invalid expert ID");
            auto & count = counts[size_t(l * experts + ids[i])];
            if (count == UINT64_MAX) throw std::runtime_error("route profile: count overflow");
            ++count;
        }
    }

    mutable std::vector<uint8_t> resident_snapshot;

    bool write(const std::string & path, const std::function<bool(int64_t,int64_t)> & resident = {}) const {
        if (resident_snapshot.size() != prompt.size()) resident_snapshot.assign(prompt.size(), 0);
        if (resident)
            for (size_t i = 0; i < prompt.size(); ++i)
                resident_snapshot[i] = resident(int64_t(i / size_t(experts)), int64_t(i % size_t(experts)));
        std::ofstream out(path);
        out << "DS41_ROUTES_V1\n" << metadata;
        for (size_t i = 0; i < prompt.size(); ++i)
            out << "count " << i / size_t(experts) << ' ' << i % size_t(experts)
                << ' ' << prompt[i] << ' ' << decode[i] << ' ' << int(resident_snapshot[i]) << '\n';
        out.flush();
        return bool(out);
    }

    std::vector<std::pair<int32_t,int32_t>> read_seed(const std::string & path) const {
        std::ifstream in(path);
        std::string line, actual;
        if (!std::getline(in, line) || line != "DS41_SEED_V1")
            throw std::runtime_error("cache seed: missing DS41_SEED_V1");
        std::vector<std::pair<int32_t,int32_t>> selected;
        std::vector<bool> seen(prompt.size());
        bool selection = false;
        while (std::getline(in, line)) {
            if (line.rfind("select ", 0) != 0) {
                if (selection) throw std::runtime_error("cache seed: metadata after selection");
                actual += line + '\n';
                continue;
            }
            selection = true;
            std::istringstream s(line.substr(7));
            int64_t l, e;
            std::string extra;
            if (!(s >> l >> e) || (s >> extra) || l < 0 || l >= layers || e < 0 || e >= experts ||
                seen[size_t(l * experts + e)])
                throw std::runtime_error("cache seed: invalid or duplicate selection");
            seen[size_t(l * experts + e)] = true;
            selected.emplace_back(int32_t(l), int32_t(e));
        }
        if (!in.eof() || selected.empty() || actual != metadata)
            throw std::runtime_error("cache seed: empty or incompatible model/geometry/devices/layout");
        return selected;
    }
};
} // namespace strata::v41
