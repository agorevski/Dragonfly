#include "ds41_engram.hpp"

#include <cstdio>
#include <cstring>

using namespace strata::v41;

static void decode(const void * src, float * dst, int64_t n) {
    std::memcpy(dst, src, (size_t) n * sizeof(float));
}

int main() {
    Ds41Geometry g;
    g.engram_layers = {1, 14};
    g.engram_heads = 1;
    g.engram_dim = 2;
    g.engram_ngram = 3;
    g.engram_pad = 2;
    g.engram_mult = {3, 5, 7, 11, 13, 17};
    g.engram_primes = {5, 7, 11, 13};
    g.engram_rows = {12, 24};
    g.engram_token_map = {0, 1, 2, 3};
    std::vector<float> a(24), b(48);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (float) i;
    for (size_t i = 0; i < b.size(); ++i) b[i] = 100.0f + (float) i;
    Ds41Engram eng;
    std::string err;
    if (!eng.open_quantized(g, {{(const uint8_t *) a.data(), 8, decode},
                                {(const uint8_t *) b.data(), 8, decode}}, 2, err)) return 1;
    eng.set_token(0, 1);
    eng.set_token(1, 3);
    std::vector<std::vector<float>> out;
    auto batch = eng.start(0, 2, out, err);
    if (!batch || !eng.wait(batch, err)) return 1;
    for (int e = 0; e < 2; ++e) {
        const auto & data = e ? b : a;
        for (int t = 0; t < 2; ++t) {
            const int64_t current = t ? 3 : 1, previous = t ? 1 : 2;
            const int64_t rolling = (current * g.engram_mult[e * 3]) ^ (previous * g.engram_mult[e * 3 + 1]);
            const int64_t p0 = (int64_t) g.engram_primes[e * 2], p1 = (int64_t) g.engram_primes[e * 2 + 1];
            const int64_t rows[] = {rolling % p0, ((rolling ^ (2 * g.engram_mult[e * 3 + 2])) % p1) + p0};
            for (int c = 0; c < 2; ++c)
                for (int d = 0; d < 2; ++d)
                    if (out[e][t * 4 + c * 2 + d] != data[rows[c] * 2 + d]) return 1;
        }
    }
    if (eng.start(0, 0, out, err) || err.empty()) return 1;
    eng.set_token(1, 0);
    if (!eng.rows(1, 1, out, err)) return 1;
    if (out[0][0] != a[(5 % 5) * 2] || out[1][0] != b[(13 % 11) * 2]) return 1;
    if (!eng.enabled()) return 1;
    Ds41Engram invalid;
    if (invalid.open_quantized(g, {{nullptr, 8, decode}, {nullptr, 8, decode}}, 1, err) || err.empty()) return 1;
    std::puts("quantized Engram: async rows, per-layer hashes, padding, overwrite and invalid length passed");
}
