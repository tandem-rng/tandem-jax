/* Tandem8x32: a noncryptographic pseudorandom number generator, fast on CPUs and GPUs alike.
 *
 * Reference C implementation of https://github.com/tandem-rng/spec.
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#ifndef TANDEM_H
#define TANDEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TANDEM_DEFAULT_K 32u

/* A generator is its transport form (key, bit position, chunk length K) plus a cache of the
 * eight chunk states of a 1024-bit row. Copy it freely: the cache is a pure function of the
 * transport form. Treat every field as private.
 *
 * The cache holds the state of row `ahead`, its exposed words in o[ahead & 1] in stream order
 * and its hidden words in h, word-major (h[word][lane]), when `cached` is set. base is the bit
 * position of the row the inline draws read, o[(base >> 10) & 1]: row ahead, or row ahead - 1
 * while the next row waits in the other slot. With nothing to read, base names a row before
 * pos, which draws never reach, since they only move forward. */
typedef struct {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;
    uint32_t cached;
    uint64_t ahead;
    /* Not next to pos: a draw stores pos, and a load of pos paired with base into one wider
     * load cannot take the stored value from the store buffer. */
    uint64_t base;
    uint32_t o[2][32];
    uint32_t h[4][8];
} tandem_rng;

/* Constructors. K is a power of two in [1, 65536]; pass 0 for the default of 32. */
tandem_rng tandem_from_key(const uint32_t key[4], uint64_t pos, uint32_t K);
tandem_rng tandem_seed(uint64_t seed_lo, uint64_t seed_hi, uint32_t K);

/* Transport form. */
void tandem_key(const tandem_rng *rng, uint32_t key[4]);
uint64_t tandem_position(const tandem_rng *rng);
uint32_t tandem_chunk_length(const tandem_rng *rng);
/* Move to bit position pos. Returns false and changes nothing when pos >= 2^63, the spec's
 * limit for a start position. */
bool tandem_set_position(tandem_rng *rng, uint64_t pos);

/* 128-bit words as two 64-bit halves, little-endian. */
typedef struct {
    uint64_t lo, hi;
} tandem_u128;

/* Scalar draws: align to the width, read, advance. */
bool tandem_next_bool(tandem_rng *rng);
uint8_t tandem_next_u8(tandem_rng *rng);
uint16_t tandem_next_u16(tandem_rng *rng);
tandem_u128 tandem_next_u128(tandem_rng *rng);

/* The 32- and 64-bit draws are inline, so that a loop keeps pos in a register: only the
 * refill is a call, and pos is stored after it. tandem_refill makes a row readable and steps
 * the cache one row ahead, so the next refill finds its row computed and no read waits on a
 * recent store. The library also exports each draw as a function. GNU89 inline semantics would
 * emit a definition in every file, so those builds call the exported functions. */
void tandem_refill(tandem_rng *rng, uint64_t row);
#if defined(__GNUC_GNU_INLINE__) && !defined(__cplusplus)
uint32_t tandem_next_u32(tandem_rng *rng);
uint64_t tandem_next_u64(tandem_rng *rng);
float tandem_next_f32(tandem_rng *rng);
double tandem_next_f64(tandem_rng *rng);
#else
/* pos - base is a multiple of the width below 1024 exactly when pos is aligned and readable. */
inline uint32_t tandem_next_u32(tandem_rng *rng) {
    uint64_t p = rng->pos;
    if ((p - rng->base) & ~(uint64_t)0x3e0u) {
        p = (p + 31u) & ~(uint64_t)31u;
        tandem_refill(rng, p >> 10);
    }
    rng->pos = p + 32u;
    return rng->o[(p >> 10) & 1u][(p >> 5) & 31u];
}

inline uint64_t tandem_next_u64(tandem_rng *rng) {
    uint64_t p = rng->pos;
    if ((p - rng->base) & ~(uint64_t)0x3c0u) {
        p = (p + 63u) & ~(uint64_t)63u;
        tandem_refill(rng, p >> 10);
    }
    rng->pos = p + 64u;
    const uint32_t *w = &rng->o[(p >> 10) & 1u][(p >> 5) & 31u];
    return w[0] | (uint64_t)w[1] << 32;
}

inline float tandem_next_f32(tandem_rng *rng) {
    return (float)(tandem_next_u32(rng) >> 8) * 5.9604644775390625e-8f; /* 2^-24 */
}

inline double tandem_next_f64(tandem_rng *rng) {
    /* 2^-53 */
    return (double)(tandem_next_u64(rng) >> 11) * 1.1102230246251565404236316680908203125e-16;
}
#endif
/* IEEE binary16 bit pattern of a uniform draw in [0, 1), the spec's Float16 mapping. */
uint16_t tandem_next_f16_bits(tandem_rng *rng);
/* A uniform Unicode scalar value, 64 stream bits per draw. */
uint32_t tandem_next_char(tandem_rng *rng);
/* Complex draws: out[0] is the real part, out[1] the imaginary part. */
void tandem_next_c32(tandem_rng *rng, float out[2]);
void tandem_next_c64(tandem_rng *rng, double out[2]);

/* Bounded draws, uniform on [0, n) by Lemire's multiply and reject over tandem_next_u32 or
 * tandem_next_u64. They match Rng::urand(range) and urand64(range) of tandem-cuda, and are not
 * part of the specification. A rejected draw is discarded, so the number of draws consumed
 * varies and a fill is not random access. n = 0 returns 0. */
uint32_t tandem_u32_below(tandem_rng *rng, uint32_t n);
uint64_t tandem_u64_below(tandem_rng *rng, uint64_t n);

/* Standard normals, Appendix A of the specification. Link with -lm.
 *
 * f64: a 1024-layer ziggurat on one 64-bit draw. A draw outside the inner rectangles, 0.43 % of
 * them, continues on a fallback stream keyed by its global draw index, which never moves this
 * generator. The values are bit exact on every compiler and target and across ports.
 *
 * f32: Box-Muller from two f32 uniforms a and b, r = sqrt(-2 ln(1 - a)) and the pair
 * (r cos 2 pi b, r sin 2 pi b), computed in float. tandem_normal_f32 returns the cos half and
 * consumes both uniforms, tandem_normal2_f32 returns both halves, cos first. They match
 * Rng::normalf of tandem-cuda bit for bit and other ports to a few ulps. */
double tandem_normal_f64(tandem_rng *rng);
float tandem_normal_f32(tandem_rng *rng);
void tandem_normal2_f32(tandem_rng *rng, float out[2]);

/* Fills: n aligned elements, the same values as n scalar draws. */
void tandem_fill_bool(tandem_rng *rng, bool *out, size_t n);
void tandem_fill_u8(tandem_rng *rng, uint8_t *out, size_t n);
void tandem_fill_u16(tandem_rng *rng, uint16_t *out, size_t n);
void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n);
void tandem_fill_u64(tandem_rng *rng, uint64_t *out, size_t n);
void tandem_fill_u128(tandem_rng *rng, tandem_u128 *out, size_t n);
void tandem_fill_f32(tandem_rng *rng, float *out, size_t n);
void tandem_fill_f64(tandem_rng *rng, double *out, size_t n);
void tandem_fill_f16_bits(tandem_rng *rng, uint16_t *out, size_t n);
void tandem_fill_char(tandem_rng *rng, uint32_t *out, size_t n);
/* n complex values as 2n interleaved components. */
void tandem_fill_c32(tandem_rng *rng, float *out, size_t n);
void tandem_fill_c64(tandem_rng *rng, double *out, size_t n);

/* len bounded draws, as the parallel fills of tandem-cuda: element i maps draw i of
 * tandem_fill_u32 or tandem_fill_u64 and the fill consumes exactly len draws. A rejected draw is
 * retried on a fallback generator split(g) of sub(0x424c573332 or 0x424c573634) of the fill's
 * generator, where g is the global draw index, the aligned start position over 32 or 64 plus i, so the values equal the scalar calls except where a draw is rejected, which has
 * probability (2^32 mod n) / 2^32, or the 64-bit analogue. */
void tandem_fill_u32_below(tandem_rng *rng, uint32_t *out, size_t len, uint32_t n);
void tandem_fill_u64_below(tandem_rng *rng, uint64_t *out, size_t len, uint64_t n);

/* Weighted choice, Appendix C of the specification: an index in [0, m) with probability
 * proportional to its weight, by an alias table in integers. tandem_choice_build fills the
 * caller's cut and alias arrays of m entries and points the table at them, with no draw. It
 * returns false and builds nothing unless 1 <= m < 2^32 and the weights are finite, not negative
 * and not all zero. A draw consumes one 64-bit draw. Element i of a fill comes from draw i of
 * tandem_fill_u64, so a fill equals n scalar draws and a fill cut anywhere equals the whole
 * fill. An empty fill aligns the position to 64. The table and the indices are exact across
 * ports. */
typedef struct {
    uint64_t capacity; /* S, the mass of one column */
    const uint64_t *cut;
    const uint32_t *alias;
    uint32_t m;
} tandem_choice_table;

bool tandem_choice_build(tandem_choice_table *table, const double *weights, size_t m,
                         uint64_t *cut, uint32_t *alias);
uint32_t tandem_choice(tandem_rng *rng, const tandem_choice_table *table);
void tandem_fill_choice(tandem_rng *rng, uint32_t *out, size_t n, const tandem_choice_table *table);

/* f64: element i from draw i of tandem_fill_u64, n draws in all, so a fill equals n scalar draws
 * and a fill cut anywhere equals the whole fill. An empty fill aligns the position to 64.
 * f32: the flattened tandem_normal2_f32 pairs, pair j from uniforms 2j and 2j + 1. An odd n keeps
 * the cos half of its last pair and still consumes both uniforms, 2 * ceil(n / 2) in all. */
void tandem_fill_normal_f64(tandem_rng *rng, double *out, size_t n);
void tandem_fill_normal_f32(tandem_rng *rng, float *out, size_t n);

/* Standard exponentials -ln(1 - u) from one uniform u each, f64 from f64 uniforms in double and
 * f32 from f32 uniforms in float, with the polynomial logarithm of the normals and no libm call.
 * They match tandem-cuda bit for bit on the host and the device and are not part of the
 * specification (Appendix A). Element i of a fill comes from uniform i of the plain fill, so a
 * fill equals n scalar draws and consumes n uniforms. */
double tandem_exponential_f64(tandem_rng *rng);
float tandem_exponential_f32(tandem_rng *rng);
void tandem_fill_exponential_f64(tandem_rng *rng, double *out, size_t n);
void tandem_fill_exponential_f32(tandem_rng *rng, float *out, size_t n);

#ifdef TANDEM_OPENMP_TARGET
/* Fills in device memory by an OpenMP target region, the same values as the host fills. out
 * points to memory of the given OpenMP device, for example from omp_target_alloc. The functions
 * are in tandem_target.c, built with the offload flags of your compiler. */
void tandem_fill_u32_target(tandem_rng *rng, uint32_t *out, size_t n, int device);
void tandem_fill_u64_target(tandem_rng *rng, uint64_t *out, size_t n, int device);
void tandem_fill_f32_target(tandem_rng *rng, float *out, size_t n, int device);
void tandem_fill_f64_target(tandem_rng *rng, double *out, size_t n, int device);
#endif

/* Random access: element i of the fill that would start here, without advancing. */
uint32_t tandem_at_u32(const tandem_rng *rng, uint64_t i);
uint64_t tandem_at_u64(const tandem_rng *rng, uint64_t i);
float tandem_at_f32(const tandem_rng *rng, uint64_t i);
double tandem_at_f64(const tandem_rng *rng, uint64_t i);

/* Derived generators. Children start at position 0 with the parent's K. */
tandem_rng tandem_split(const tandem_rng *rng, uint64_t index);
tandem_rng tandem_sub(const tandem_rng *rng, uint64_t purpose);
/* Fork n children from the parent's current block and move the parent past it. */
void tandem_fork(tandem_rng *parent, tandem_rng *children, uint64_t n);

/* Building blocks, exposed for conformance tests and ports. */
void tandem_T(uint32_t o[4], uint32_t h[4]);
void tandem_F(uint32_t o[4], uint32_t h[4]);
void tandem_F_keyed(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                    uint32_t o[4], uint32_t h[4]);
/* Block B(c, j): the exposed half of chunk c after j + 1 steps. */
void tandem_block(const uint32_t key[4], uint64_t c, uint32_t j, uint32_t out[4]);

#ifdef __cplusplus
}
#endif

#endif /* TANDEM_H */
