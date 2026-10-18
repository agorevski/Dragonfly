#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cpu = strata::kernels::cpu;

static void fill_rows(uint8_t* dst, int type, size_t stride, int width, int rows, float value) {
    std::vector<float> row(width, value);
    std::vector<uint8_t> packed(stride);
    if (type == 143) cpu::ptq1_0_quantize_ref(row.data(), packed.data(), width);
    else ggml_get_type_traits((ggml_type) type)->from_float_ref(row.data(), packed.data(), width);
    for (int r = 0; r < rows; ++r) std::memcpy(dst + r * stride, packed.data(), stride);
}

int main() {
    constexpr int H = 5120, FF = 2304, NT = 2;
    cpu::NativeFmt formats[2];
    std::string error;
    for (int e = 0; e < 2; ++e) {
        const int type = e ? 143 : 8;
        if (!cpu::native_fmt(type, type, H, FF, formats[e], error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        formats[e].swiglu_limit = 1.f;
    }
    std::vector<uint8_t> blobs[2], acts[2][NT], hidden[2][NT];
    std::vector<float> intermediates[2][NT], expected[2][NT], got[2][NT];
    cpu::ExpertJobMulti jobs[2];
    for (int e = 0; e < 2; ++e) {
        const auto& f = formats[e];
        blobs[e].resize(f.bytes);
        fill_rows(blobs[e].data(), f.gu_type, f.gu_row, H, FF, .03f);
        fill_rows(blobs[e].data() + f.up_off, f.gu_type, f.gu_row, H, FF, -.03f);
        fill_rows(blobs[e].data() + f.down_off, f.d_type, f.d_row, FF, H, .01f);
        jobs[e].blob = blobs[e].data();
        jobs[e].nt = NT;
        jobs[e].fmt = &f;
        const void* ap[NT];
        const void* hp[NT];
        float* fp[NT];
        float* out[NT];
        for (int t = 0; t < NT; ++t) {
            std::vector<float> x(H, t ? .2f : .1f);
            acts[e][t].resize(f.act_bytes);
            hidden[e][t].resize(f.h_bytes);
            intermediates[e][t].resize(FF);
            expected[e][t].resize(H);
            got[e][t].assign(H, NAN);
            cpu::native_quant_act(f, x.data(), acts[e][t].data());
            jobs[e].nact[t] = ap[t] = acts[e][t].data();
            jobs[e].out[t] = got[e][t].data();
            fp[t] = intermediates[e][t].data();
            hp[t] = hidden[e][t].data();
            out[t] = expected[e][t].data();
        }
        cpu::native_gu_rows(f, blobs[e].data(), ap, NT, fp, 0, FF);
        const float clamped = -1.f / (1.f + std::exp(-1.f));
        for (int t = 0; t < NT; ++t) {
            if (std::fabs(fp[t][0] - clamped) > 1.e-6f || std::fabs(fp[t][FF - 1] - clamped) > 1.e-6f)
                return 1;
            cpu::native_quant_h(f, fp[t], hidden[e][t].data());
        }
        cpu::native_down_rows(f, blobs[e].data(), hp, NT, out, 0, H);
    }
    cpu::ExpertPool pool(2, false);
    pool.run_split_multi_native(formats[0], jobs, 2);
    for (int e = 0; e < 2; ++e)
        for (int t = 0; t < NT; ++t)
            if (std::memcmp(got[e][t].data(), expected[e][t].data(), H * sizeof(float))) {
                std::fprintf(stderr, "mixed pool differs: expert %d token %d\n", e, t);
                return 1;
            }
    std::puts("mixed Q8_0/PTQ1_0 pool: 5120x2304, all rows, two tokens, finite SwiGLU clamp passed");
}
