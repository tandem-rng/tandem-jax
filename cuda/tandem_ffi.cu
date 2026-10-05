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
#include <optional>
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

/* Float64 normals: element e of a row is the ziggurat of UInt64 draw d0 + e. Short fills take
 * fill_normal64_fused of tandem.cuh, which continues each miss where it finds it. Longer ones
 * take two kernels as tandem.cuh does, so that the 0.43 % of misses do not stall whole warps: a
 * table pass writes the fast path of every element and queues its misses in shared memory, which a
 * block appends to its row's list with one atomic add, and a second kernel continues the listed
 * misses, one thread each. A row whose list overflows continues every
 * miss on a second walk. */
__global__ void __launch_bounds__(THREADS)
    normal64_fused(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, double *out) {
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    out += n * row;
    uint64_t d0 = align_pos(q.pos, 64) >> 6, g0 = (d0 >> 4) / K;
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    const NormalSpan sp(c, K, d0, n);
    sp.walk(q.key, c, [&](uint64_t d, uint64_t r0, uint64_t r1, bool) {
        if (sp.in(d)) out[d - d0] = normal_f64(r0, q.key, K, d);
        if (sp.in(d + 1u)) out[d + 1u - d0] = normal_f64(r1, q.key, K, d + 1u);
    });
}

/* The table pass of fill_normal64_kernel<SHIFT> of tandem.cuh, stores included. */
template <bool SHIFT>
__device__ __forceinline__ void normal64_table_pass(const uint32_t key[4], uint32_t K, uint64_t c,
                                                    const NormalSpan &sp, double *out,
                                                    double *firsts, auto &&push) {
    const uint64_t d0 = sp.d0;
    auto fast = [&](uint64_t d, uint64_t r0, uint64_t r1, bool edge, double &z0, double &z1) {
        bool h0, h1;
        z0 = normal_f64_fast(r0, h0);
        z1 = normal_f64_fast(r1, h1);
        if (__builtin_expect(!(h0 && h1), 0)) {
            if (!h0 && (!edge || sp.in(d))) push(d - d0, r0);
            if (!h1 && (!edge || sp.in(d + 1u))) push(d + 1u - d0, r1);
        }
    };
    uint32_t o[4], h[4];
    if constexpr (!SHIFT) {
        /* Each block is stored one step late, so that the store does not wait for the step's
         * table reads. */
        int steps = sp.jl + 1;
        F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
        double z0 = 0.0, z1 = 0.0;
        bool prev_edge = true;
        uint64_t d = 2u * sp.beta0;
        double *dst = out + (d - d0) - 16;
        int j = 0;
        for (; j < steps; j++, d += 16u, dst += 16) {
            T(o, h);
            uint64_t r0 = o[0] | ((uint64_t)o[1] << 32), r1 = o[2] | ((uint64_t)o[3] << 32);
            bool edge = j <= sp.jf || j >= sp.jl;
            double n0, n1;
            fast(d, r0, r1, edge, n0, n1);
            if (!prev_edge) {
                *reinterpret_cast<double2 *>(dst) = make_double2(z0, z1);
            } else if (j) {
                if (sp.in(d - 16u)) dst[0] = z0;
                if (sp.in(d - 15u)) dst[1] = z1;
            }
            z0 = n0, z1 = n1, prev_edge = edge;
        }
        if (j) {
            if (sp.in(d - 16u)) dst[0] = z0;
            if (sp.in(d - 15u)) dst[1] = z1;
        }
    } else {
        /* Every thread of the warp takes every step, so that the shuffles see the whole warp. */
        unsigned wl = threadIdx.x & 31u, lane = threadIdx.x & 7u;
        uint64_t bw = sp.beta0 - lane - (wl >> 3) * K * 8u; /* the warp's first block at step 0 */
        int steps = bw > sp.bb ? 0 : (int)((sp.bb - bw) >> 3 < K - 1u ? (sp.bb - bw) >> 3 : K - 1u) + 1;
        F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
        double z0 = 0.0, z1 = 0.0, first = 0.0;
        bool prev_edge = true;
        uint64_t d = 2u * sp.beta0;
        double *dst = out + (d - d0) - 15;
        unsigned src = lane < 7 ? wl + 1u : wl - 7u;
        int j = 0;
        for (; j < steps; j++, d += 16u, dst += 16) {
            T(o, h);
            uint64_t r0 = o[0] | ((uint64_t)o[1] << 32), r1 = o[2] | ((uint64_t)o[3] << 32);
            bool edge = j <= sp.jf || j >= sp.jl;
            double n0, n1;
            fast(d, r0, r1, edge, n0, n1);
            double lo = __shfl_sync(0xffffffffu, lane ? z0 : n0, src);
            if (!prev_edge) {
                *reinterpret_cast<double2 *>(dst) = make_double2(z1, lo);
            } else if (j) {
                if (sp.in(d - 15u)) dst[0] = z1;
                if (sp.in(d - 14u)) dst[1] = lo;
            } else if (threadIdx.x == 0 && sp.in(d)) {
                dst[15] = n0;
            }
            if (j == 0) first = n0;
            z0 = n0, z1 = n1, prev_edge = edge;
        }
        unsigned warp = threadIdx.x >> 5;
        if (wl == 0) firsts[warp] = first;
        __syncthreads();
        bool next = wl < 31 || warp + 1 < THREADS / 32;
        double lo = __shfl_sync(0xffffffffu, lane ? z0 : first, wl < 31 ? wl + 1u : wl);
        if (wl == 31 && next) lo = firsts[warp + 1];
        if (j) {
            bool b = next && sp.in(d - 14u);
            if (sp.in(d - 15u) && b)
                *reinterpret_cast<double2 *>(dst) = make_double2(z1, lo);
            else {
                if (sp.in(d - 15u)) dst[0] = z1;
                if (b) dst[1] = lo;
            }
        }
    }
}

__global__ void __launch_bounds__(THREADS)
    normal64_table(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, double *out,
                   NormalMiss *lists, unsigned long long *counts, uint64_t cap) {
    __shared__ NormalMiss queue[NORMAL_QUEUE];
    __shared__ unsigned nq;
    __shared__ unsigned long long base;
    __shared__ double firsts[THREADS / 32];
    if (threadIdx.x == 0) nq = 0;
    __syncthreads();
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    out += n * row;
    NormalMiss *list = lists + cap * row;
    unsigned long long *count = counts + row;
    uint64_t d0 = align_pos(q.pos, 64) >> 6, g0 = (d0 >> 4) / K;
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    const NormalSpan sp(c, K, d0, n);
    auto push = [&](uint64_t e, uint64_t r) {
        unsigned s = atomicAdd(&nq, 1u);
        if (s < NORMAL_QUEUE) {
            queue[s] = NormalMiss{e, r};
        } else {
            unsigned long long t = atomicAdd(count, 1ull);
            if (t < cap) list[t] = NormalMiss{e, r};
        }
    };
    /* The same for the whole grid row, so the shuffles and barriers of SHIFT see every thread. */
    if (((reinterpret_cast<uintptr_t>(out) - 8u * d0) & 15u) == 0)
        normal64_table_pass<false>(q.key, K, c, sp, out, firsts, push);
    else
        normal64_table_pass<true>(q.key, K, c, sp, out, firsts, push);
    __syncthreads();
    unsigned m = nq < NORMAL_QUEUE ? nq : NORMAL_QUEUE;
    if (m == 0) return;
    if (threadIdx.x == 0) base = atomicAdd(count, (unsigned long long)m);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < m; i += THREADS)
        if (base + i < cap) list[base + i] = queue[i];
}

__global__ void __launch_bounds__(THREADS)
    normal64_misses(const uint32_t *prm, uint32_t K, uint64_t n, uint64_t row0, double *out,
                    const NormalMiss *lists, const unsigned long long *counts, uint64_t cap) {
    uint64_t row = row0 + blockIdx.y;
    Params q = load(prm + PARAMS * row);
    out += n * row;
    const NormalMiss *list = lists + cap * row;
    uint64_t m = counts[row], d0 = align_pos(q.pos, 64) >> 6;
    uint64_t i0 = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, stride = gridDim.x * (uint64_t)blockDim.x;
    if (m <= cap) {
        for (uint64_t i = i0; i < m; i += stride) {
            NormalMiss x = list[i];
            out[x.e] = normal_f64_slow(x.r, q.key, K, d0 + x.e);
        }
        return;
    }
    uint64_t g0 = (d0 >> 4) / K, chunks = 8u * (((d0 + n - 1u) >> 4) / K - g0 + 1u);
    for (uint64_t i = i0; i < chunks; i += stride) {
        uint64_t c = 8u * g0 + i;
        NormalSpan sp(c, K, d0, n);
        sp.walk(q.key, c, [&](uint64_t d, uint64_t r0, uint64_t r1, bool) {
            bool h0, h1;
            normal_f64_fast(r0, h0);
            normal_f64_fast(r1, h1);
            if (sp.in(d) && !h0) out[d - d0] = normal_f64_slow(r0, q.key, K, d);
            if (sp.in(d + 1u) && !h1) out[d + 1u - d0] = normal_f64_slow(r1, q.key, K, d + 1u);
        });
    }
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

ffi::Error Fill(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::Buffer<ffi::U32> prm,
                ffi::Result<ffi::AnyBuffer> out, int64_t n, int64_t chunk, std::string_view kind,
                int32_t width) {
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
    /* The miss lists of the f64 normals, room for twice the expected misses of each row, from
     * XLA's scratch allocator. tandem.cuh takes them from the stream-ordered pool, but XLA's event
     * syncs let that pool release them, and each call then paid about 0.6 ms on the host to map
     * the memory again. Without scratch memory the fused kernel runs. */
    NormalMiss *list = nullptr;
    unsigned long long *counts = nullptr;
    uint64_t cap = m / 128u, head = (rows * sizeof *counts + 15u) & ~(uint64_t)15u;
    if (kind == "normal" && t == D::F64 && m >= NORMAL_LIST_MIN) {
        if (std::optional<void *> s = scratch.Allocate(head + rows * cap * sizeof(NormalMiss), 16)) {
            counts = static_cast<unsigned long long *>(*s);
            list = reinterpret_cast<NormalMiss *>(static_cast<char *>(*s) + head);
            cudaMemsetAsync(counts, 0, rows * sizeof *counts, stream);
        }
    }
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
            if (t == D::F64) {
                dim3 grid(normal_grid(pairs + 1, K), y);
                if (!list)
                    normal64_fused<<<grid, THREADS, 0, stream>>>(p, K, m, row0, (double *)o);
                else {
                    normal64_table<<<grid, THREADS, 0, stream>>>(p, K, m, row0, (double *)o, list, counts, cap);
                    /* One thread per expected miss, at 0.43 %. The grid-stride loop takes any excess. */
                    dim3 mgrid((unsigned)((m / 200u + THREADS) / THREADS), y);
                    normal64_misses<<<mgrid, THREADS, 0, stream>>>(p, K, m, row0, (double *)o, list, counts, cap);
                }
            } else if (t == D::F32)
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
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("chunk")
                                  .Attr<std::string_view>("kind")
                                  .Attr<int32_t>("width"));
