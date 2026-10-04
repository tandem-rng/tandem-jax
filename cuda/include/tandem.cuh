/* Tandem8x32 for CUDA: a noncryptographic pseudorandom number generator, fast on CPUs and
 * GPUs alike. Header only, built as C++20 or later, on the portable C++17 core include/tandem/core.hpp.
 *
 * Implements https://github.com/tandem-rng/spec and produces the stream it defines, bit for
 * bit. Entry points:
 *
 *   - tandem::fill_u32/u64/f32/f64: fill device memory from a key and stream position. One
 *     thread per chunk, blocks stored lane-interleaved so a warp writes whole 128-byte lines.
 *   - tandem::fill_u32_below/u64_below, fill_normal_f64/f32: bounded integers and normals, which
 *     are not part of the specification. A bounded element consumes one draw and a rejected draw
 *     retries on a fallback stream, a normal is Box-Muller of two Float64 draws of the fill.
 *     See the README for the stream contract.
 *   - tandem::generator: key, position and K on the host, whose fill_* calls advance the
 *     position.
 *   - tandem::device_rng: a per-thread generator with one cached chunk state, for kernels
 *     that draw scalars. Its draws equal the C library's tandem_next_* calls.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include "tandem/core.hpp"

namespace tandem {

/* ---- Per-thread generator ---------------------------------------------------------------- */

/* tandem::Rng with the state in public fields: key, pos and K are the transport form, plus
 * one cached chunk state, about 20 registers. The draw API (bounded draws, normals, at_*,
 * split, sub, fork) is Rng's, from Draws<device_rng>. */
struct device_rng : GenState, Draws<device_rng> {
    __host__ __device__ static device_rng from_key(const uint32_t key[4], uint64_t pos,
                                                   uint32_t K) {
        device_rng r;
        r.init(key, pos, K);
        return r;
    }

    __host__ __device__ static device_rng seed(uint64_t seed_lo, uint64_t seed_hi,
                                               uint32_t K) {
        uint32_t k[4];
        seed_key(seed_lo, seed_hi, k);
        return from_key(k, 0, K);
    }

    __host__ __device__ GenState &st() { return *this; }
    __host__ __device__ const GenState &st() const { return *this; }

    __host__ __device__ void skip_to(uint64_t p) { pos = p; }
    __host__ __device__ void load(uint64_t p) { cache.load(key, K, p); }
    __host__ __device__ uint64_t read(uint64_t p, unsigned w) { return cache.read(key, K, p, w); }
    __host__ __device__ uint64_t next(unsigned w) { return cache.next(key, K, pos, w); }

    __host__ __device__ bool next_bool() { return next(1) != 0; }
    __host__ __device__ uint8_t next_u8() { return (uint8_t)next(8); }
    __host__ __device__ uint16_t next_u16() { return (uint16_t)next(16); }
    __host__ __device__ uint32_t next_u32() { return (uint32_t)next(32); }
    __host__ __device__ uint64_t next_u64() { return next(64); }
    __host__ __device__ uint16_t next_f16_bits() { return to_f16_bits(next_u16()); }
    __host__ __device__ float next_f32() { return to_f32(next_u32()); }
    __host__ __device__ double next_f64() { return to_f64(next_u64()); }
};

/* ---- Fills ------------------------------------------------------------------------------ */

namespace detail {

/* Kinds that share an output type with another kind. */
struct bool_bits {}; /* one stream bit per bool, stored as a byte */
struct f16_bits {};  /* binary16 bit patterns, stored as uint16_t */
/* Lemire bounded draws over the u32 (u64) fill, see PURPOSE_BELOW32, stored as O after adding
 * a low bound. O may be wider than the draw. */
template <class O> struct below32 {};
template <class O> struct below64 {};

/* The range and low bound of a bounded fill, and its rejection threshold, computed once. */
struct Bound {
    uint64_t range, low, thresh;
};

/* What a kind needs beyond the block words: the fill's key and chunk length, and the range
 * of the bounded kinds. */
struct Ctx {
    const uint32_t *key;
    uint32_t K;
    uint64_t range, low, thresh;
};

/* How an output element is made from the four words of a block. `i` is the element's index in
 * the block, `bits` its width in the stream. */
template <class E> struct elem;

template <> struct elem<uint32_t> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) { return w[i]; }
};
template <> struct elem<float> {
    using out_t = float;
    static constexpr unsigned bits = 32;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) { return to_f32(w[i]); }
};
template <> struct elem<uint64_t> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) {
        return w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
    }
};
template <> struct elem<double> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) {
        return to_f64(w[2 * i] | ((uint64_t)w[2 * i + 1] << 32));
    }
};
template <> struct elem<uint16_t> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) {
        return (uint16_t)(w[i >> 1] >> ((i & 1u) * 16u));
    }
};
template <> struct elem<f16_bits> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &x) {
        return to_f16_bits(elem<uint16_t>::make(w, i, 0, x));
    }
};
template <> struct elem<uint8_t> {
    using out_t = uint8_t;
    static constexpr unsigned bits = 8;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) {
        return (uint8_t)(w[i >> 2] >> ((i & 3u) * 8u));
    }
};
template <class O> struct elem<below32<O>> {
    using out_t = O;
    static constexpr unsigned bits = 32;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t g, const Ctx &x) {
        using U = std::make_unsigned_t<O>;
        uint32_t v = below_u32_t(w[i], (uint32_t)x.range, (uint32_t)x.thresh, x.key, x.K, g);
        return (O)(U)((U)x.low + (U)v);
    }
};
template <class O> struct elem<below64<O>> {
    using out_t = O;
    static constexpr unsigned bits = 64;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t g, const Ctx &x) {
        using U = std::make_unsigned_t<O>;
        uint64_t v = below_u64_t(w[2 * i] | ((uint64_t)w[2 * i + 1] << 32), x.range, x.thresh,
                                 x.key, x.K, g);
        return (O)(U)((U)x.low + (U)v);
    }
};
template <> struct elem<bool_bits> {
    using out_t = bool;
    static constexpr unsigned bits = 1;
};

/* The vector type that stores 16 bytes of output. */
template <class E> struct vec4 { using type = uint4; };
template <> struct vec4<float> { using type = float4; };
template <> struct vec4<uint64_t> { using type = ulonglong2; };
template <> struct vec4<double> { using type = double2; };

/* Store the elements of one block that fall inside the output. `first` is the byte offset
 * of the block in the stream, `b0` and `b1` bound the output's bytes. With ALIGNED the
 * output's blocks sit at 16-byte addresses, and a block fully inside is one vector store. */
template <class E, bool ALIGNED>
__device__ __forceinline__ void store_block(typename elem<E>::out_t *out, uint64_t b0,
                                            uint64_t b1, uint64_t first, const uint32_t w[4],
                                            const Ctx &x) {
    using out_t = typename elem<E>::out_t;
    constexpr unsigned size = elem<E>::bits / 8;
    constexpr unsigned per_block = 16 / size;
    out_t v[per_block];
    /* The global draw index of element i, which keys the bounded fallback, so that a fill cut
     * anywhere equals the whole fill. `first` is a byte of the stream, a multiple of size. */
    for (unsigned i = 0; i < per_block; i++) v[i] = elem<E>::make(w, i, first / size + i, x);
    if constexpr (sizeof(out_t) != size) {
        /* The output element is wider than its draw. A block inside the output goes out as
         * 16-byte stores when its first element is 16-byte aligned, element stores otherwise. */
        if (first >= b0 && first + 16 <= b1) {
            out_t *dst = out + (first - b0) / size;
            if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
                constexpr unsigned vecs = per_block * sizeof(out_t) / 16;
                for (unsigned k = 0; k < vecs; k++)
                    reinterpret_cast<uint4 *>(dst)[k] = reinterpret_cast<const uint4 *>(v)[k];
                return;
            }
        }
        for (unsigned i = 0; i < per_block; i++) {
            uint64_t at = first + i * size;
            if (at >= b0 && at + size <= b1) out[(at - b0) / size] = v[i];
        }
        return;
    }
    char *dst = reinterpret_cast<char *>(out) + (first - b0);
    if (ALIGNED && first >= b0 && first + 16 <= b1) {
        *reinterpret_cast<typename vec4<E>::type *>(dst) =
            *reinterpret_cast<const typename vec4<E>::type *>(v);
        return;
    }
    for (unsigned i = 0; i < per_block; i++) {
        uint64_t at = first + i * size;
        if (at >= b0 && at + size <= b1) reinterpret_cast<out_t *>(dst)[i] = v[i];
    }
}

constexpr unsigned THREADS = 256;
constexpr unsigned TILE_STEPS = 8;

/* One thread per chunk, direct stores. The thread walks its K blocks and stores those inside
 * rows r0 .. r1 (inclusive). Rows are 128 bytes, blocks 16, lanes are the eight chunks of a
 * group, so the eight threads of a group store one line per step. Any K. */
template <class E, bool ALIGNED>
__global__ void fill_rows_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3,
                                 uint32_t K, uint64_t g0, uint64_t r0, uint64_t r1, uint64_t b0,
                                 uint64_t b1, Bound bd, typename elem<E>::out_t *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g > r1 / K) return;
    const uint32_t key[4] = {key0, key1, key2, key3};
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    const Ctx x{key, K, bd.range, bd.low, bd.thresh};
    uint64_t row = g * K;
    uint32_t j0 = row < r0 ? (uint32_t)(r0 - row) : 0u;
    uint32_t j1 = (uint32_t)(r1 - row < K - 1u ? r1 - row : K - 1u);
    for (uint32_t j = 0; j <= j1; j++) {
        T(o, h);
        if (j >= j0) store_block<E, ALIGNED>(out, b0, b1, (row + j) * 128u + lane * 16u, o, x);
    }
}

/* The write phase of the tile kernel for outputs twice as wide as their draws (32-bit bounded
 * draws into 8-byte elements). Consecutive threads take consecutive 16-byte output slots, two
 * draws each, so a warp stores 512 contiguous bytes per instruction. Per stream block, two
 * 16-byte stores 32 bytes apart left every sector half written by each instruction. */
template <class E>
__device__ __forceinline__ void store_tile_widened(const uint4 *tile, unsigned slots, uint64_t gb,
                                                   uint32_t K, uint32_t jb, uint64_t b0,
                                                   uint64_t b1, const Ctx &x,
                                                   typename elem<E>::out_t *out) {
    using out_t = typename elem<E>::out_t;
    constexpr unsigned size = elem<E>::bits / 8, per_tile_group = TILE_STEPS * 8;
    const uint2 *half = reinterpret_cast<const uint2 *>(tile);
    for (unsigned s = threadIdx.x; s < 2 * slots; s += THREADS) {
        unsigned sg = s / (2 * per_tile_group), within = s % (2 * per_tile_group);
        uint64_t first = ((gb + sg) * K + jb) * 128u + within * 8u; /* stream byte of the pair */
        if (first >= b1) continue;
        uint2 v = half[s];
        uint32_t w[4] = {v.x, v.y, 0u, 0u};
        alignas(16) out_t o[2] = {elem<E>::make(w, 0, first / size, x),
                                  elem<E>::make(w, 1, first / size + 1, x)};
        if (first >= b0 && first + 2 * size <= b1) {
            out_t *dst = out + (first - b0) / size;
            if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
                *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(o);
                continue;
            }
        }
        for (unsigned i = 0; i < 2; i++) {
            uint64_t at = first + i * size;
            if (at >= b0 && at + size <= b1) out[(at - b0) / size] = o[i];
        }
    }
}

/* One thread per chunk, 32 groups per block, output staged through shared memory. Every
 * TILE_STEPS steps the block holds, for each of its groups, TILE_STEPS consecutive rows,
 * which are 1024 contiguous bytes of the stream. The write phase hands consecutive 16-byte
 * slots to consecutive threads, so a warp writes 512 contiguous bytes. Needs K >= TILE_STEPS. */
template <class E, bool ALIGNED>
__global__ void __launch_bounds__(THREADS)
    fill_tile_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3, uint32_t K,
                     uint64_t g0, uint64_t r1, uint64_t b0, uint64_t b1, Bound bd,
                     typename elem<E>::out_t *out) {
    constexpr unsigned GROUPS = THREADS / 8, SLOTS = GROUPS * TILE_STEPS * 8;
    __shared__ uint4 tile[SLOTS];
    uint64_t gb = g0 + blockIdx.x * (uint64_t)GROUPS; /* first group of this block */
    unsigned gi = threadIdx.x >> 3, lane = threadIdx.x & 7u;
    uint64_t c = 8u * (gb + gi) + lane;
    bool mine = gb + gi <= r1 / K;
    const uint32_t key[4] = {key0, key1, key2, key3};
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    const Ctx x{key, K, bd.range, bd.low, bd.thresh};
    uint64_t block_first = gb * K * 128u; /* stream byte of the block's first row */
    for (uint32_t jb = 0; jb < K; jb += TILE_STEPS) {
        if (block_first + jb * 128u > b1) break;
        if (mine) {
            for (unsigned j = 0; j < TILE_STEPS; j++) {
                T(o, h);
                tile[(gi * TILE_STEPS + j) * 8 + lane] = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        if constexpr (sizeof(typename elem<E>::out_t) == 2 * elem<E>::bits / 8)
            store_tile_widened<E>(tile, SLOTS, gb, K, jb, b0, b1, x, out);
        else
            for (unsigned s = threadIdx.x; s < SLOTS; s += THREADS) {
                unsigned sg = s / (TILE_STEPS * 8), within = s % (TILE_STEPS * 8);
                uint64_t first = ((gb + sg) * K + jb) * 128u + within * 16u;
                if (first >= b1) continue;
                uint4 v = tile[s];
                uint32_t w[4] = {v.x, v.y, v.z, v.w};
                store_block<E, ALIGNED>(out, b0, b1, first, w, x);
            }
        __syncthreads();
    }
}

/* Bool fill: one stream bit becomes one output byte, so a block of 128 bits is 128 bytes and
 * a row is 1024 contiguous bytes. One thread per chunk, 32 groups per block, one step at a
 * time through shared memory, so each warp writes whole 16-byte slots in stream order. A
 * thread's eight slots are rotated by its lane, which keeps the shared stores free of bank
 * conflicts. `p0` and `p1` bound the output in stream bits, which are also output bytes. */
template <bool ALIGNED>
__global__ void __launch_bounds__(THREADS)
    fill_bool_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3, uint32_t K,
                     uint64_t g0, uint64_t r1, uint64_t p0, uint64_t p1, bool *out) {
    constexpr unsigned GROUPS = THREADS / 8, SLOTS = GROUPS * 64;
    __shared__ uint4 tile[SLOTS];
    uint64_t gb = g0 + blockIdx.x * (uint64_t)GROUPS;
    unsigned gi = threadIdx.x >> 3, lane = threadIdx.x & 7u;
    uint64_t c = 8u * (gb + gi) + lane;
    bool mine = gb + gi <= r1 / K;
    const uint32_t key[4] = {key0, key1, key2, key3};
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    for (uint32_t j = 0; j < K; j++) {
        if ((gb * K + j) * 1024u >= p1) break;
        T(o, h);
        if (mine) {
            for (unsigned s = 0; s < 8; s++) {
                uint32_t x = o[s >> 1] >> ((s & 1u) * 16u), y[4];
                /* Spread four bits over four bytes. */
                for (unsigned t = 0; t < 4; t++) y[t] = (((x >> (4 * t)) & 0xfu) * 0x00204081u) & 0x01010101u;
                tile[threadIdx.x * 8 + ((s + lane) & 7u)] = make_uint4(y[0], y[1], y[2], y[3]);
            }
        }
        __syncthreads();
        for (unsigned s = threadIdx.x; s < SLOTS; s += THREADS) {
            unsigned sg = s >> 6, within = s & 63u;
            uint64_t q = ((gb + sg) * K + j) * 1024u + within * 16u;
            if (q >= p1 || q + 16u <= p0) continue;
            uint4 v = tile[(s & ~7u) + (((s & 7u) + (within >> 3)) & 7u)];
            bool *dst = out + (q - p0);
            if (ALIGNED && q >= p0 && q + 16u <= p1) {
                *reinterpret_cast<uint4 *>(dst) = v;
            } else {
                const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&v);
                for (unsigned k = 0; k < 16; k++)
                    if (q + k >= p0 && q + k < p1) dst[k] = bytes[k];
            }
        }
        __syncthreads();
    }
}

/* Normal fill. Pair j, the elements 2j (cos half) and 2j + 1 (sin half), is one Box-Muller step
 * of the Float64 draws 2j and 2j + 1 of the fill, so the first draw of the fill sits at the
 * position aligned to 64 bits. With that position at an even draw, both draws of a pair are
 * the halves of one block. At an odd draw (ODD) they
 * are the high half of one block and the low half of the next, so each thread also steps the
 * chunk that holds its predecessor block: chunk c - 1, or for lane 0 the last lane of the
 * previous row, which is chunk 8g + 7 one step behind or, at step 0, the previous group's. Block
 * `ba` is the first block that yields a pair, `bb` the last. One pair per block. */
template <class O, bool ODD>
__global__ void __launch_bounds__(THREADS)
    fill_normal_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3, uint32_t K,
                       uint64_t g0, uint64_t ba, uint64_t bb, uint64_t n, O *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    const uint32_t key[4] = {key0, key1, key2, key3};
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
            else { out[e] = (O)z.z0; out[e + 1] = (O)z.z1; }
        } else {
            out[e] = (O)z.z0;
        }
    }
}

template <class O>
inline uint64_t fill_normal(const uint32_t key[4], uint64_t pos, uint32_t K, O *out, size_t n,
                            cudaStream_t stream) {
    K = K ? K : DEFAULT_K;
    if (n == 0) return pos; /* consumes no draws, so no alignment either */
    uint64_t pairs = ((uint64_t)n + 1u) / 2u;
    uint64_t p0 = align_pos(pos, 64), p1 = p0 + pairs * 128u;
    bool odd = (p0 >> 6) & 1u;
    uint64_t ba = (p0 >> 7) + (odd ? 1u : 0u), bb = ba + pairs - 1u;
    uint64_t g0 = (ba >> 3) / K, g1 = (bb >> 3) / K;
    unsigned blocks = (unsigned)((8u * (g1 - g0 + 1u) + THREADS - 1) / THREADS);
    if (odd)
        fill_normal_kernel<O, true><<<blocks, THREADS, 0, stream>>>(key[0], key[1], key[2],
                                                                    key[3], K, g0, ba, bb, n, out);
    else
        fill_normal_kernel<O, false><<<blocks, THREADS, 0, stream>>>(key[0], key[1], key[2],
                                                                     key[3], K, g0, ba, bb, n, out);
    return p1;
}

/* The float Box-Muller step of the fill kernel. It is box_muller2_f32 with the angle through
 * the fast __sincosf, which with precise logf and sqrtf makes the fill memory bound instead of
 * compute bound. __sincosf is accurate only on [-pi, pi], so the angle is shifted by half a turn.
 * The result stays within 16 ulps + 1e-6 of the precise step (measured at most 1.5e-6 absolute).
 * Define TANDEM_PRECISE_F32_NORMAL for sincospif. __logf is not used: its absolute error near
 * 1 distorts small radii by thousands of ulps. */
__device__ __forceinline__ Pair2<float> normal_step_f32(float a, float b) {
#if defined(TANDEM_PRECISE_F32_NORMAL)
    return box_muller2_f32(a, b);
#else
    float r = sqrtf(-2.0f * logf(1.0f - a)), s, c;
    __sincosf(6.2831853071795864769f * (b - 0.5f), &s, &c);
    return Pair2<float>{-r * c, -r * s};
#endif
}

/* Float normal fill. Pair j is one float Box-Muller step of the Float32 draws 2j and 2j + 1 of
 * the fill that starts at the position aligned to 32 bits, giving elements 2j and 2j + 1. With
 * s0 the index of the first Float32 draw, a block holds two pairs: slots 0 and 1, 2 and 3 when
 * s0 is even, or slot 3 of the previous block with slot 0, and slots 1 and 2, when s0 is odd.
 * The odd case steps the predecessor chunk as fill_normal_kernel does. `ba` and `bb` bound the
 * blocks that can hold a pair, `np` is the number of pairs. */
template <bool ODD>
__global__ void __launch_bounds__(THREADS)
    fill_normal32_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3, uint32_t K,
                         uint64_t g0, uint64_t s0, uint64_t n, uint64_t np, uint64_t ba,
                         uint64_t bb, float *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 7u) == 0;
    /* Common case: a block starts a pair, both pairs fit, and the four outputs are one 16-byte
     * store. */
    const bool quad = !ODD && (s0 & 3u) == 0 && (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    const uint32_t key[4] = {key0, key1, key2, key3};
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
        /* Index of the pair that starts at slot 4 beta (or 4 beta - 1), then the next one. */
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

inline uint64_t fill_normal_f32_impl(const uint32_t key[4], uint64_t pos, uint32_t K,
                                     float *out, size_t n, cudaStream_t stream) {
    K = K ? K : DEFAULT_K;
    if (n == 0) return pos;
    uint64_t np = ((uint64_t)n + 1u) / 2u;
    uint64_t p0 = align_pos(pos, 32), p1 = p0 + np * 64u;
    uint64_t s0 = p0 >> 5, ba = s0 >> 2, bb = (s0 + 2u * np - 1u) >> 2;
    uint64_t g0 = (ba >> 3) / K, g1 = (bb >> 3) / K;
    unsigned blocks = (unsigned)((8u * (g1 - g0 + 1u) + THREADS - 1) / THREADS);
    if (s0 & 1u)
        fill_normal32_kernel<true><<<blocks, THREADS, 0, stream>>>(
            key[0], key[1], key[2], key[3], K, g0, s0, n, np, ba, bb, out);
    else
        fill_normal32_kernel<false><<<blocks, THREADS, 0, stream>>>(
            key[0], key[1], key[2], key[3], K, g0, s0, n, np, ba, bb, out);
    return p1;
}

/* `tile` selects the shared-memory kernel where K allows; the tests and the bench also run
 * the direct kernel at every K. */
template <class E>
inline uint64_t fill(const uint32_t key[4], uint64_t pos, uint32_t K,
                     typename elem<E>::out_t *out, size_t n, cudaStream_t stream,
                     bool tile = true, Bound bd = Bound{0, 0, 0}) {
    constexpr unsigned bits = elem<E>::bits;
    K = K ? K : DEFAULT_K;
    uint64_t p0 = align_pos(pos, bits), p1 = p0 + (uint64_t)n * bits;
    if (n == 0) return p1;
    uint64_t r0 = p0 >> 10, r1 = (p1 - 1) >> 10;
    uint64_t g0 = r0 / K, g1 = r1 / K;
    constexpr bool is_bool = std::is_same<E, bool_bits>::value;
    uint64_t b0 = p0 / 8, b1 = p1 / 8;
    /* Blocks land on 16-byte addresses when the output's first byte and the fill's first
     * stream byte agree modulo 16. A bool output byte is one stream bit. */
    bool aligned = ((reinterpret_cast<uintptr_t>(out) - (is_bool ? p0 : b0)) & 15u) == 0;
    uint64_t groups = g1 - g0 + 1u;
    if constexpr (is_bool) {
        unsigned blocks = (unsigned)((groups + THREADS / 8 - 1) / (THREADS / 8));
        if (aligned)
            fill_bool_kernel<true><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, p0, p1, out);
        else
            fill_bool_kernel<false><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, p0, p1, out);
        return p1;
    } else if (tile && K >= TILE_STEPS) {
        unsigned blocks = (unsigned)((groups + THREADS / 8 - 1) / (THREADS / 8));
        if (aligned)
            fill_tile_kernel<E, true><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, b0, b1, bd, out);
        else
            fill_tile_kernel<E, false><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, b0, b1, bd, out);
    } else {
        unsigned blocks = (unsigned)((8u * groups + THREADS - 1) / THREADS);
        if (aligned)
            fill_rows_kernel<E, true><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r0, r1, b0, b1, bd, out);
        else
            fill_rows_kernel<E, false><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r0, r1, b0, b1, bd, out);
    }
    return p1;
}

} // namespace detail

/* Fill n elements of device memory with the draws that start at stream position pos, as
 * the C library's tandem_fill_* would. Return the position after the fill. */
inline uint64_t fill_u32(const uint32_t key[4], uint64_t pos, uint32_t K, uint32_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill<uint32_t>(key, pos, K, out, n, stream);
}
inline uint64_t fill_u64(const uint32_t key[4], uint64_t pos, uint32_t K, uint64_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill<uint64_t>(key, pos, K, out, n, stream);
}
inline uint64_t fill_f32(const uint32_t key[4], uint64_t pos, uint32_t K, float *out, size_t n,
                         cudaStream_t stream = 0) {
    return detail::fill<float>(key, pos, K, out, n, stream);
}
inline uint64_t fill_f64(const uint32_t key[4], uint64_t pos, uint32_t K, double *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill<double>(key, pos, K, out, n, stream);
}
/* One stream bit per element. Each bool is stored as one byte, 0 or 1. */
inline uint64_t fill_bool(const uint32_t key[4], uint64_t pos, uint32_t K, bool *out, size_t n,
                          cudaStream_t stream = 0) {
    return detail::fill<detail::bool_bits>(key, pos, K, out, n, stream);
}
inline uint64_t fill_u8(const uint32_t key[4], uint64_t pos, uint32_t K, uint8_t *out, size_t n,
                        cudaStream_t stream = 0) {
    return detail::fill<uint8_t>(key, pos, K, out, n, stream);
}
inline uint64_t fill_u16(const uint32_t key[4], uint64_t pos, uint32_t K, uint16_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill<uint16_t>(key, pos, K, out, n, stream);
}
/* binary16 bit patterns of the specification's Float16 draws, (raw >> 5) * 2^-11. */
inline uint64_t fill_f16_bits(const uint32_t key[4], uint64_t pos, uint32_t K, uint16_t *out,
                              size_t n, cudaStream_t stream = 0) {
    return detail::fill<detail::f16_bits>(key, pos, K, out, n, stream);
}
/* Signed integers reinterpret the unsigned draw of the same width in two's complement. */
inline uint64_t fill_i8(const uint32_t key[4], uint64_t pos, uint32_t K, int8_t *out, size_t n,
                        cudaStream_t stream = 0) {
    return fill_u8(key, pos, K, reinterpret_cast<uint8_t *>(out), n, stream);
}
inline uint64_t fill_i16(const uint32_t key[4], uint64_t pos, uint32_t K, int16_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return fill_u16(key, pos, K, reinterpret_cast<uint16_t *>(out), n, stream);
}
inline uint64_t fill_i32(const uint32_t key[4], uint64_t pos, uint32_t K, int32_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return fill_u32(key, pos, K, reinterpret_cast<uint32_t *>(out), n, stream);
}
inline uint64_t fill_i64(const uint32_t key[4], uint64_t pos, uint32_t K, int64_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return fill_u64(key, pos, K, reinterpret_cast<uint64_t *>(out), n, stream);
}

/* Bounded fills: n draws uniform on [0, range), by Lemire's method as Rng::urand(range). Each
 * element consumes exactly one draw of the u32 (u64) fill, so the fill advances the position
 * by 32 n (64 n) bits whatever the draws are. A rejected draw retries on a fallback stream,
 * see PURPOSE_BELOW32 in core.hpp. Not part of the specification. */
inline uint64_t fill_u32_below(const uint32_t key[4], uint64_t pos, uint32_t K, uint32_t range,
                               uint32_t *out, size_t n, cudaStream_t stream = 0) {
    if (n == 0) return pos;
    return detail::fill<detail::below32<uint32_t>>(
        key, pos, K, out, n, stream, true, detail::Bound{range, 0, below_threshold_u32(range)});
}
inline uint64_t fill_u64_below(const uint32_t key[4], uint64_t pos, uint32_t K, uint64_t range,
                               uint64_t *out, size_t n, cudaStream_t stream = 0) {
    if (n == 0) return pos;
    return detail::fill<detail::below64<uint64_t>>(
        key, pos, K, out, n, stream, true, detail::Bound{range, 0, below_threshold_u64(range)});
}

/* Bounded fills with a low bound: out[e] = low + draw[e] on [low, low + range), wrapping in the
 * element type, so a signed output is the same two's complement value. The draw consumption and
 * the draws are those of the fills above. The output may be wider than the draw: a 32-bit
 * range (one UInt32 draw per element) can store 64-bit elements, and the sum is formed in the
 * output type. They fuse the offset and the widening into the store, so no second pass. */
namespace detail {
template <class O, class W>
inline uint64_t fill_below_low(const uint32_t key[4], uint64_t pos, uint32_t K, W range, O low,
                               O *out, size_t n, cudaStream_t stream) {
    if (n == 0) return pos;
    if constexpr (sizeof(W) == 4)
        return fill<below32<O>>(key, pos, K, out, n, stream, true,
                                Bound{range, (uint64_t)(int64_t)low, below_threshold_u32(range)});
    else
        return fill<below64<O>>(key, pos, K, out, n, stream, true,
                                Bound{range, (uint64_t)(int64_t)low, below_threshold_u64(range)});
}
} // namespace detail

#define TANDEM_BELOW_LOW(NAME, W, O)                                                               \
    inline uint64_t NAME(const uint32_t key[4], uint64_t pos, uint32_t K, W range, O low, O *out, \
                         size_t n, cudaStream_t stream = 0) {                                      \
        return detail::fill_below_low(key, pos, K, range, low, out, n, stream);                    \
    }
TANDEM_BELOW_LOW(fill_u32_below, uint32_t, uint32_t)
TANDEM_BELOW_LOW(fill_u32_below, uint32_t, int32_t)
TANDEM_BELOW_LOW(fill_u32_below, uint32_t, uint64_t)
TANDEM_BELOW_LOW(fill_u32_below, uint32_t, int64_t)
TANDEM_BELOW_LOW(fill_u64_below, uint64_t, uint64_t)
TANDEM_BELOW_LOW(fill_u64_below, uint64_t, int64_t)
#undef TANDEM_BELOW_LOW

/* Standard normals by Box-Muller. fill_normal_f64 is the flattened Rng::normal2 calls:
 * pair j, elements 2j (cos half) and 2j + 1 (sin half), comes from the Float64 draws 2j and
 * 2j + 1 of the fill that starts at the position aligned to 64 bits. The fill consumes
 * 2 ceil(n / 2) draws, so an odd n uses the cos half of its last pair and still advances past
 * both draws. fill_normal_f32 is the same on Rng::normalf2 with float arithmetic and Float32
 * draws, aligned to 32 bits. Device log and cos differ
 * from the host's in the last bits, so normals agree to a few ulps, not bit for bit. Not part
 * of the specification. */
inline uint64_t fill_normal_f64(const uint32_t key[4], uint64_t pos, uint32_t K, double *out,
                                size_t n, cudaStream_t stream = 0) {
    return detail::fill_normal(key, pos, K, out, n, stream);
}
inline uint64_t fill_normal_f32(const uint32_t key[4], uint64_t pos, uint32_t K, float *out,
                                size_t n, cudaStream_t stream = 0) {
    return detail::fill_normal_f32_impl(key, pos, K, out, n, stream);
}

/* A host handle for a stream: the fills above, with the position kept and advanced here. Fills
 * on one stream run in order, and each call returns the new position, so a generator can
 * issue fills back to back without a sync. */
struct generator {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;

    static generator from_key(const uint32_t k[4], uint64_t p = 0, uint32_t K = DEFAULT_K) {
        generator g;
        for (int w = 0; w < 4; w++) g.key[w] = k[w];
        g.pos = p;
        g.K = K ? K : DEFAULT_K;
        return g;
    }

    static generator seed(uint64_t seed_lo, uint64_t seed_hi = 0, uint32_t K = DEFAULT_K) {
        uint32_t k[4];
        seed_key(seed_lo, seed_hi, k);
        return from_key(k, 0, K);
    }

    uint64_t fill_u32(uint32_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u32(key, pos, K, out, n, s);
    }
    uint64_t fill_u64(uint64_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u64(key, pos, K, out, n, s);
    }
    uint64_t fill_f32(float *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_f32(key, pos, K, out, n, s);
    }
    uint64_t fill_f64(double *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_f64(key, pos, K, out, n, s);
    }
    uint64_t fill_bool(bool *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_bool(key, pos, K, out, n, s);
    }
    uint64_t fill_u8(uint8_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u8(key, pos, K, out, n, s);
    }
    uint64_t fill_u16(uint16_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u16(key, pos, K, out, n, s);
    }
    uint64_t fill_f16_bits(uint16_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_f16_bits(key, pos, K, out, n, s);
    }
    uint64_t fill_i8(int8_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_i8(key, pos, K, out, n, s);
    }
    uint64_t fill_i16(int16_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_i16(key, pos, K, out, n, s);
    }
    uint64_t fill_i32(int32_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_i32(key, pos, K, out, n, s);
    }
    uint64_t fill_i64(int64_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_i64(key, pos, K, out, n, s);
    }
    uint64_t fill_u32_below(uint32_t range, uint32_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u32_below(key, pos, K, range, out, n, s);
    }
    uint64_t fill_u64_below(uint64_t range, uint64_t *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u64_below(key, pos, K, range, out, n, s);
    }
    /* With a low bound, and for 32-bit ranges into wider outputs, see the free functions. */
    template <class O> uint64_t fill_u32_below(uint32_t range, O low, O *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u32_below(key, pos, K, range, low, out, n, s);
    }
    template <class O> uint64_t fill_u64_below(uint64_t range, O low, O *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_u64_below(key, pos, K, range, low, out, n, s);
    }
    uint64_t fill_normal_f64(double *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_normal_f64(key, pos, K, out, n, s);
    }
    uint64_t fill_normal_f32(float *out, size_t n, cudaStream_t s = 0) {
        return pos = tandem::fill_normal_f32(key, pos, K, out, n, s);
    }
};

} // namespace tandem
