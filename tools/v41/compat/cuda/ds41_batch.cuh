// Included by iq_kernels.cu: reuse its native weight decoders and q8_1 rounding.
namespace {

constexpr int DS41_BATCH_COLS = 4;

template<int TY>
__device__ __forceinline__ void ds41_batch_dot(const uint8_t* row, const block_q8_1* x,
                                              const int (&off)[DS41_BATCH_COLS], int n, int nb,
                                              int lane, float (&sums)[DS41_BATCH_COLS]) {
#pragma unroll
    for (int c = 0; c < DS41_BATCH_COLS; ++c) sums[c] = 0;
    for (int k = lane; k < nb * Fmt<TY>::ipb; k += 32) {
        const int kb = k / Fmt<TY>::ipb, iqs = k % Fmt<TY>::ipb;
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
        const int bo = 4 * (iqs / QI8_1);
        int vi[4], sc[4], mins[4];
        float d, dm = 0;
        if constexpr (TY == 10) {
            const auto* w = (const block_q2_K*) row + kb;
            const int v = get_int_b4(w->qs, iqs);
            const float2 scales = __half22float2(w->dm);
            d = scales.x;
            dm = scales.y;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int scale = w->scales[scale_offset + 2 * i];
                vi[i] = (v >> (2 * i)) & 0x03030303;
                sc[i] = scale & 15;
                mins[i] = (scale >> 4) * 0x01010101;
            }
        } else {
            const auto* w = (const block_q3_K*) row + kb;
            const int vl = get_int_b2(w->qs, iqs);
            const int vh = ~get_int_b2(w->hmask, iqs % (QI3_K / 2)) >> bo;
            d = __half2float(w->d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int isc = scale_offset + 2 * i;
                const int lo = (w->scales[isc % 8] >> (4 * (isc / 8))) & 15;
                const int hi = ((w->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
                sc[i] = (lo | hi) - 32;
                vi[i] = __vsubss4((vl >> (2 * i)) & 0x03030303, ((vh >> i) << 2) & 0x04040404);
            }
        }
#pragma unroll
        for (int c = 0; c < DS41_BATCH_COLS; ++c) {
            if (c >= n) continue;
            const auto* a = x + off[c] + kb * 8 + bo;
            float sd = 0, sm = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int u = get_int_b4(a[i].qs, iqs % QI8_1);
                const float da = __low2float(a[i].ds);
                sd += da * (ggml_cuda_dp4a(vi[i], u, 0) * sc[i]);
                if constexpr (TY == 10) sm += da * ggml_cuda_dp4a(mins[i], u, 0);
            }
            if constexpr (TY == 10) sums[c] += d * sd - dm * sm;
            else sums[c] += d * sd;
        }
    }
#pragma unroll
    for (int c = 0; c < DS41_BATCH_COLS; ++c)
        if (c < n) sums[c] = warp_sum(sums[c]);
}

template<int GT, int UT, bool DOWN>
__global__ void ds41_batch_rows(const NativeExpertLayout L, const unsigned long long* grp_ptr,
                               const int32_t* grp_start, const int32_t* n_groups,
                               const int32_t* ent_dst, const int32_t* ent_tok,
                               const block_q8_1* xq, float* gate, float* up, float* out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e = grp_start[g] + blockIdx.z * DS41_BATCH_COLS;
    const int n = min(DS41_BATCH_COLS, grp_start[g + 1] - e);
    if (n <= 0) return;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int rows = DOWN ? L.n_embd : 2 * L.n_ff;
    if (row >= rows) return;
    const bool is_up = !DOWN && row >= L.n_ff;
    const int r = is_up ? row - L.n_ff : row;
    const int width = DOWN ? L.n_ff : L.n_embd;
    const int xb = width / 32;
    const size_t offset = DOWN ? L.down_off + (size_t) r * L.d_row :
                                (is_up ? L.up_off : 0) +
                                (size_t) r * (is_up && L.up_type >= 0 ? L.up_row : L.gu_row);
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + offset;
    int off[DS41_BATCH_COLS];
#pragma unroll
    for (int c = 0; c < DS41_BATCH_COLS; ++c)
        off[c] = (DOWN ? e + min(c, n - 1) : ent_tok[e + min(c, n - 1)]) * xb;
    float sums[DS41_BATCH_COLS];
    if constexpr (DOWN) ds41_batch_dot<11>(wr, xq, off, n, width / 256, lane, sums);
    else if (is_up) ds41_batch_dot<UT>(wr, xq, off, n, width / 256, lane, sums);
    else ds41_batch_dot<GT>(wr, xq, off, n, width / 256, lane, sums);
    if (lane < n) {
        if constexpr (DOWN) out[(size_t) ent_dst[e + lane] * L.n_embd + r] = sums[lane];
        else (is_up ? up : gate)[(size_t) (e + lane) * L.n_ff + r] = sums[lane];
    }
}

// One CTA owns a complete 32-row q8 block for each token. Dot products keep the
// unfused kernel's warp reduction; only the FP32 gate/up memory roundtrip is removed.
template<int GT, int UT>
__global__ void ds41_batch_fused_gu(const NativeExpertLayout L, const unsigned long long* grp_ptr,
                                   const int32_t* grp_start, const int32_t* n_groups,
                                   const int32_t* ent_tok, const block_q8_1* xq, block_q8_1* hq) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e = grp_start[g] + blockIdx.z * DS41_BATCH_COLS;
    const int n = min(DS41_BATCH_COLS, grp_start[g + 1] - e);
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * 32;
    __shared__ float hidden[DS41_BATCH_COLS][32];
    int off[DS41_BATCH_COLS];
#pragma unroll
    for (int c = 0; c < DS41_BATCH_COLS; ++c)
        off[c] = ent_tok[e + min(c, n - 1)] * (L.n_embd / 32);
    const auto* weights = (const uint8_t*) grp_ptr[g];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = r0 + warp + j * 8;
        float gate[DS41_BATCH_COLS], up[DS41_BATCH_COLS];
        ds41_batch_dot<GT>(weights + (size_t) r * L.gu_row, xq, off, n,
                           L.n_embd / 256, lane, gate);
        const size_t ur = L.up_type < 0 ? L.gu_row : L.up_row;
        ds41_batch_dot<UT>(weights + L.up_off + (size_t) r * ur, xq, off, n,
                           L.n_embd / 256, lane, up);
#pragma unroll
        for (int c = 0; c < DS41_BATCH_COLS; ++c)
            if (c < n && lane == c) {
                const float cg = fminf(gate[c], L.swiglu_limit);
                const float cu = fminf(fmaxf(up[c], -L.swiglu_limit), L.swiglu_limit);
                hidden[c][warp + j * 8] = __fmul_rn(cg / (1.0f + __expf(-cg)), cu);
            }
    }
    __syncthreads();
    if (warp < n)
        q8_1_store(hidden[warp][lane], hq, (long long) (e + warp) * L.n_ff + r0 + lane);
}

} // namespace

bool native_expert_batch_supported(const NativeExpertLayout& L) noexcept {
    const int ut = L.up_type < 0 ? L.gu_type : L.up_type;
    if (!(L.n_embd == 5120 && L.n_ff == 2304 && (L.gu_type == 10 || L.gu_type == 11) &&
          (ut == 10 || ut == 11) && L.d_type == 11 && L.tail_off == 0)) return false;
    const auto canonical = native_expert_layout3(L.gu_type, ut, L.d_type, L.n_embd, L.n_ff);
    return L.gu_row == canonical.gu_row && L.d_row == canonical.d_row &&
           L.up_row == canonical.up_row && L.up_off == canonical.up_off &&
           L.down_off == canonical.down_off && L.bytes == canonical.bytes;
}

bool native_expert_batch(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                         int max_group_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (!native_expert_batch_supported(L) || cap_groups < 0 || cap_groups > 65535 ||
        cap_entries < 0 || cap_entries > std::numeric_limits<int32_t>::max() / (L.n_ff / 32) ||
        max_group_entries < 0 || max_group_entries > 65535 * DS41_BATCH_COLS) return false;
    if (cap_groups == 0 || cap_entries == 0 || max_group_entries == 0) return true;
    const size_t f = (size_t) cap_entries * L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    auto* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const auto s = (cudaStream_t) stream;
    const unsigned tiles = (max_group_entries + DS41_BATCH_COLS - 1) / DS41_BATCH_COLS;
    const dim3 gu((2 * L.n_ff + 7) / 8, (unsigned) cap_groups, tiles);
    const int ut = L.up_type < 0 ? L.gu_type : L.up_type;
    static const bool fused_gu = [] {
        const char* v = std::getenv("STRATA_DS41_BATCH_FUSED_GU");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (fused_gu) {
        const dim3 fg(L.n_ff / 32, (unsigned) cap_groups, tiles);
#define DS41_BATCH_FUSED(G, U) ds41_batch_fused_gu<G, U><<<fg, 256, 0, s>>>(L, grp_ptr, grp_start, n_groups, ent_tok, X, hq)
        if (L.gu_type == 10 && ut == 10) DS41_BATCH_FUSED(10, 10);
        else if (L.gu_type == 10) DS41_BATCH_FUSED(10, 11);
        else if (ut == 10) DS41_BATCH_FUSED(11, 10);
        else DS41_BATCH_FUSED(11, 11);
#undef DS41_BATCH_FUSED
    } else {
#define DS41_BATCH_GU(G, U) ds41_batch_rows<G, U, false><<<gu, 256, 0, s>>>(L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, out)
        if (L.gu_type == 10 && ut == 10) DS41_BATCH_GU(10, 10);
        else if (L.gu_type == 10) DS41_BATCH_GU(10, 11);
        else if (ut == 10) DS41_BATCH_GU(11, 10);
        else DS41_BATCH_GU(11, 11);
#undef DS41_BATCH_GU
        // This is the existing finite-clamp, FP32-scale -> integer -> FP16-scale quantizer.
        const long long nh = cap_entries * L.n_ff;
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(
            gate, up, grp_start, n_groups, (int) L.n_ff, hq, L.swiglu_limit);
    }
    const dim3 down((L.n_embd + 7) / 8, (unsigned) cap_groups, tiles);
    ds41_batch_rows<10, 10, true><<<down, 256, 0, s>>>(
        L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, hq, gate, up, out);
    check("native_expert_batch");
    return true;
}
