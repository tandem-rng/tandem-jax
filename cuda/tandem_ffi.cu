/* The tandem-cuda fills as one XLA FFI handler, run on the stream XLA hands over.
 *
 * The kernels are the tile and normal kernels of tandem.cuh, with the key, the position and the
 * bounds read from device memory instead of taken as kernel arguments. Under jit they are traced
 * values, and copying them to the host would stall the stream. The geometry of a fill depends on
 * the position, so each kernel derives it on the device, and the grid is sized for the worst
 * alignment: blocks past the fill exit before any work.
 *
 * Batched calls (vmap) carry one parameter row per fill and write the fills back to back, row b
 * to out + b n, as grid row b.
 */
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <cuda_runtime.h>

#include "tandem.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace tandem;
using namespace tandem::detail;

/* Bounded 32-bit draws whose range may be 2^32, which a 64-bit output type allows: then every
 * draw is a value. Other ranges are tandem.cuh's below32. */
template <class O> struct bounded32 {};

namespace tandem::detail {
template <class O> struct elem<bounded32<O>> {
    using out_t = O;
    static constexpr unsigned bits = 32;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t g, const Ctx &x) {
        using U = std::make_unsigned_t<O>;
        uint32_t v = x.range == (1ull << 32)
                         ? w[i]
                         : below_u32_t(w[i], (uint32_t)x.range, (uint32_t)x.thresh, x.key, x.K, g);
        return (O)(U)((U)x.low + (U)v);
    }
};
} // namespace tandem::detail

namespace {

/* key[4], then position, range and low bound as (low, high) word pairs. */
constexpr unsigned PARAMS = 10;
constexpr unsigned GROUPS = THREADS / 8;
constexpr unsigned SLOTS = GROUPS * TILE_STEPS * 8;
constexpr unsigned MAX_GRID_Y = 65535;

struct Params {
    uint32_t key[4];
    uint64_t pos, range, low;
};

__device__ __forceinline__ Params load(const uint32_t *p) {
    Params q;
    for (int w = 0; w < 4; w++) q.key[w] = p[w];
    q.pos = p[4] | (uint64_t)p[5] << 32;
    q.range = p[6] | (uint64_t)p[7] << 32;
    q.low = p[8] | (uint64_t)p[9] << 32;
    return q;
}

/* fill_tile_kernel of tandem.cuh on the parameters of one row. */
template <class E>
__device__ __forceinline__ void fill_tile(uint4 *tile, const Params &q, uint32_t K, uint64_t n,
                                          Bound bd, typename elem<E>::out_t *out) {
    constexpr unsigned bits = elem<E>::bits;
    uint64_t p0 = align_pos(q.pos, bits), p1 = p0 + n * bits;
    uint64_t r1 = (p1 - 1) >> 10, b0 = p0 / 8, b1 = p1 / 8;
    uint64_t gb = (p0 >> 10) / K + blockIdx.x * (uint64_t)GROUPS;
    if (gb > r1 / K) return; /* uniform over the block, so before any __syncthreads */
    bool aligned = ((reinterpret_cast<uintptr_t>(out) - b0) & 15u) == 0;
    unsigned gi = threadIdx.x >> 3, lane = threadIdx.x & 7u;
    bool mine = gb + gi <= r1 / K;
    uint32_t o[4], h[4];
    F_keyed(q.key, 8u * (gb + gi) + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
    const Ctx x{q.key, K, bd.range, bd.low, bd.thresh};
    uint64_t block_first = gb * K * 128u;
    for (uint32_t jb = 0; jb < K; jb += TILE_STEPS) {
        if (block_first + jb * 128u > b1) break;
        if (mine) {
            for (unsigned j = 0; j < TILE_STEPS; j++) {
                T(o, h);
                tile[(gi * TILE_STEPS + j) * 8 + lane] = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        if constexpr (sizeof(typename elem<E>::out_t) == 2 * bits / 8)
            store_tile_widened<E>(tile, SLOTS, gb, K, jb, b0, b1, x, out);
        else
            for (unsigned s = threadIdx.x; s < SLOTS; s += THREADS) {
                unsigned sg = s / (TILE_STEPS * 8), within = s % (TILE_STEPS * 8);
                uint64_t first = ((gb + sg) * K + jb) * 128u + within * 16u;
                if (first >= b1) continue;
                uint4 v = tile[s];
                uint32_t w[4] = {v.x, v.y, v.z, v.w};
                if (aligned) store_block<E, true>(out, b0, b1, first, w, x);
                else store_block<E, false>(out, b0, b1, first, w, x);
            }
        __syncthreads();
    }
}

template <class E>
__global__ void __launch_bounds__(THREADS)
    fill_kernel(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0,
                typename elem<E>::out_t *out) {
    __shared__ uint4 tile[SLOTS];
    uint64_t row = row0 + blockIdx.y;
    fill_tile<E>(tile, load(prm + PARAMS * row), K, n, Bound{0, 0, 0}, out + n * row);
}

/* `width` 0 takes the draw width from the range, 64 bits above 2^32. tandem.cuh stores outputs as
 * wide as the draw or wider, so 64-bit draws need a 64-bit O. */
template <class O>
__global__ void __launch_bounds__(THREADS)
    below_kernel(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, int width, O *out) {
    __shared__ uint4 tile[SLOTS];
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    out += n * row;
    if constexpr (sizeof(O) == 8)
        if (width == 64 || (width == 0 && q.range > (1ull << 32))) {
            fill_tile<below64<O>>(tile, q, K, n, Bound{q.range, q.low, below_threshold_u64(q.range)}, out);
            return;
        }
    fill_tile<bounded32<O>>(tile, q, K, n, Bound{q.range, q.low, below_threshold_u32((uint32_t)q.range)},
                            out);
}

/* fill_normal_kernel<double, ODD> of tandem.cuh. */
template <bool ODD>
__device__ __forceinline__ void normal64(const uint32_t key[4], uint32_t K, uint64_t g0,
                                         uint64_t ba, uint64_t bb, uint64_t n, double *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (ODD) F_keyed(key, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
    for (uint32_t j = 0; j < K; j++) {
        uint64_t beta = (g * K + j) * 8u + lane;
        if (beta > bb) break;
        T(o, h);
        if (ODD && (lane || j)) T(po, ph);
        if (beta < ba) continue;
        uint32_t q[4];
        const uint32_t *prev = po;
        if (ODD && lane == 0 && j == 0) {
            block(key, 8u * (g - 1u) + 7u, K - 1u, q);
            prev = q;
        }
        uint64_t u = ODD ? prev[2] | ((uint64_t)prev[3] << 32) : o[0] | ((uint64_t)o[1] << 32);
        uint64_t v = ODD ? o[0] | ((uint64_t)o[1] << 32) : o[2] | ((uint64_t)o[3] << 32);
        Pair2<double> z = box_muller2(to_f64(u), to_f64(v));
        uint64_t e = 2u * (beta - ba);
        if (e + 1 < n) {
            if (vec) *reinterpret_cast<double2 *>(out + e) = make_double2(z.z0, z.z1);
            else { out[e] = z.z0; out[e + 1] = z.z1; }
        } else {
            out[e] = z.z0;
        }
    }
}

__global__ void __launch_bounds__(THREADS)
    normal64_kernel(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, double *out) {
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    uint64_t pairs = (n + 1) / 2, p0 = align_pos(q.pos, 64);
    bool odd = (p0 >> 6) & 1u;
    uint64_t ba = (p0 >> 7) + (odd ? 1u : 0u), bb = ba + pairs - 1u, g0 = (ba >> 3) / K;
    if (odd) normal64<true>(q.key, K, g0, ba, bb, n, out + n * row);
    else normal64<false>(q.key, K, g0, ba, bb, n, out + n * row);
}

/* fill_normal32_kernel<ODD> of tandem.cuh. */
template <bool ODD>
__device__ __forceinline__ void normal32(const uint32_t key[4], uint32_t K, uint64_t g0,
                                         uint64_t s0, uint64_t n, uint64_t np, uint64_t ba,
                                         uint64_t bb, float *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 7u) == 0;
    const bool quad = !ODD && (s0 & 3u) == 0 && (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (ODD) F_keyed(key, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
    for (uint32_t j = 0; j < K; j++) {
        uint64_t beta = (g * K + j) * 8u + lane;
        if (beta > bb) break;
        T(o, h);
        if (ODD && (lane || j)) T(po, ph);
        if (beta < ba) continue;
        if (quad && 4u * (beta - ba) + 4u <= n) {
            Pair2<float> z0 = normal_step_f32(to_f32(o[0]), to_f32(o[1]));
            Pair2<float> z1 = normal_step_f32(to_f32(o[2]), to_f32(o[3]));
            *reinterpret_cast<float4 *>(out + 4u * (beta - ba)) = make_float4(z0.z0, z0.z1, z1.z0, z1.z1);
            continue;
        }
        int64_t e = ((int64_t)(4u * beta) - (ODD ? 1 : 0) - (int64_t)s0) / 2;
        uint32_t q[4];
        const uint32_t *prev = po;
        if (ODD && lane == 0 && j == 0 && g > 0) {
            block(key, 8u * (g - 1u) + 7u, K - 1u, q);
            prev = q;
        }
        for (int k = 0; k < 2; k++) {
            int64_t pj = e + k;
            if (pj < 0 || pj >= (int64_t)np) continue;
            uint32_t ua = ODD ? (k ? o[1] : prev[3]) : o[2 * k];
            uint32_t ub = ODD ? (k ? o[2] : o[0]) : o[2 * k + 1];
            Pair2<float> z = normal_step_f32(to_f32(ua), to_f32(ub));
            uint64_t at = 2u * (uint64_t)pj;
            if (at + 1 < n) {
                if (vec) *reinterpret_cast<float2 *>(out + at) = make_float2(z.z0, z.z1);
                else { out[at] = z.z0; out[at + 1] = z.z1; }
            } else {
                out[at] = z.z0;
            }
        }
    }
}

__global__ void __launch_bounds__(THREADS)
    normal32_kernel(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, float *out) {
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    uint64_t np = (n + 1) / 2, s0 = align_pos(q.pos, 32) >> 5;
    uint64_t ba = s0 >> 2, bb = (s0 + 2u * np - 1u) >> 2, g0 = (ba >> 3) / K;
    if (s0 & 1u) normal32<true>(q.key, K, g0, s0, n, np, ba, bb, out + n * row);
    else normal32<false>(q.key, K, g0, s0, n, np, ba, bb, out + n * row);
}

/* Grid widths for any start position: a fill of `blocks` stream blocks of 16 bytes spans at most
 * blocks / (8 K) + 2 groups, one more is slack. */
unsigned tile_grid(uint64_t n, unsigned bits, uint32_t K) {
    uint64_t groups = (n * bits - 1) / (1024u * K) + 3;
    return (unsigned)((groups + GROUPS - 1) / GROUPS);
}

unsigned normal_grid(uint64_t blocks, uint32_t K) {
    uint64_t groups = blocks / (8u * K) + 3;
    return (unsigned)((8u * groups + THREADS - 1) / THREADS);
}

ffi::Error Fill(cudaStream_t stream, ffi::Buffer<ffi::U32> prm, ffi::Result<ffi::AnyBuffer> out,
                int64_t n, int64_t chunk, std::string_view kind, int32_t width) {
    uint64_t rows = prm.element_count() / PARAMS, m = (uint64_t)n;
    uint32_t K = (uint32_t)chunk;
    if (prm.element_count() % PARAMS || rows * m != out->element_count())
        return ffi::Error::InvalidArgument("tandem_fill: parameter rows and output size disagree");
    if (kind != "normal" && K < TILE_STEPS)
        return ffi::Error::InvalidArgument("tandem_fill: chunk length below 8");
    if (m == 0 || rows == 0) return ffi::Error::Success();
    const uint32_t *p = prm.typed_data();
    void *o = out->untyped_data();
    ffi::DataType t = out->element_type();
    using D = ffi::DataType;
    for (uint64_t row0 = 0; row0 < rows; row0 += MAX_GRID_Y) {
        unsigned y = (unsigned)(rows - row0 < MAX_GRID_Y ? rows - row0 : MAX_GRID_Y);
        if (kind == "stream") {
            unsigned bits = t == D::U32 || t == D::S32 || t == D::F32 ? 32 : 64;
            dim3 grid(tile_grid(m, bits, K), y);
            if (t == D::U32 || t == D::S32)
                fill_kernel<uint32_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, (uint32_t *)o);
            else if (t == D::U64 || t == D::S64)
                fill_kernel<uint64_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, (uint64_t *)o);
            else if (t == D::F32)
                fill_kernel<float><<<grid, THREADS, 0, stream>>>(p, K, m, row0, (float *)o);
            else if (t == D::F64)
                fill_kernel<double><<<grid, THREADS, 0, stream>>>(p, K, m, row0, (double *)o);
            else
                return ffi::Error::InvalidArgument("tandem_fill: stream needs u32, s32, u64, s64, f32 or f64");
        } else if (kind == "below") {
            if (width == 64 && (t == D::U32 || t == D::S32))
                return ffi::Error::InvalidArgument("tandem_fill: 64-bit draws need a 64-bit output");
            bool narrow = width == 32 || t == D::U32 || t == D::S32;
            dim3 grid(tile_grid(m, narrow ? 32 : 64, K), y);
            if (t == D::U32)
                below_kernel<uint32_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, width, (uint32_t *)o);
            else if (t == D::S32)
                below_kernel<int32_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, width, (int32_t *)o);
            else if (t == D::U64)
                below_kernel<uint64_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, width, (uint64_t *)o);
            else if (t == D::S64)
                below_kernel<int64_t><<<grid, THREADS, 0, stream>>>(p, K, m, row0, width, (int64_t *)o);
            else
                return ffi::Error::InvalidArgument("tandem_fill: below needs u32, s32, u64 or s64");
        } else if (kind == "normal") {
            uint64_t pairs = (m + 1) / 2;
            if (t == D::F64)
                normal64_kernel<<<dim3(normal_grid(pairs, K), y), THREADS, 0, stream>>>(p, K, m, row0, (double *)o);
            else if (t == D::F32)
                normal32_kernel<<<dim3(normal_grid((2 * pairs + 2) / 4, K), y), THREADS, 0, stream>>>(
                    p, K, m, row0, (float *)o);
            else
                return ffi::Error::InvalidArgument("tandem_fill: normal needs f32 or f64");
        } else {
            return ffi::Error::InvalidArgument("tandem_fill: kind must be stream, below or normal");
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    return ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(TandemFill, Fill,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("chunk")
                                  .Attr<std::string_view>("kind")
                                  .Attr<int32_t>("width"));
