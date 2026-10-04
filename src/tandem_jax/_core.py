"""Tandem8x32 building blocks on uint32 arrays, after the specification."""

import jax
import jax.numpy as jnp
from jax import lax

U32 = jnp.uint32
CLOCK_WEYL = 0x9E3779B9
DOMAIN_STREAM = 0x9E3779B9
DOMAIN_SPLIT = 0xBB67AE85
DOMAIN_FORK = 0xD2511F53
DOMAIN_FOLD = 0xCD9E8D57
DOMAIN_SEED = 0xA54FF53A
AUX_STREAM = 0x94D049BB
RC = (0xD17CC1B7, 0xA7220A94, 0xFE13ABE8, 0xFA9A6EE0, 0xEDB14ACC, 0x9E21C820, 0xFF28B1D5, 0xEF5DE2B0)


def _rotl(x, r):
    return (x << U32(r)) | (x >> U32(32 - r))


def _mul_wide(a, b):
    """Low and high words of the 64-bit product. Without x64 the product is built from
    16-bit halves, which also runs on backends without 64-bit integers."""
    if jax.config.jax_enable_x64:
        p = a.astype(jnp.uint64) * b.astype(jnp.uint64)
        return p.astype(U32), (p >> jnp.uint64(32)).astype(U32)
    mask = U32(0xFFFF)
    a0, a1, b0, b1 = a & mask, a >> U32(16), b & mask, b >> U32(16)
    ll, lh, hl, hh = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    mid = (ll >> U32(16)) + (lh & mask) + (hl & mask)
    lo = (ll & mask) | (mid << U32(16))
    hi = hh + (lh >> U32(16)) + (hl >> U32(16)) + (mid >> U32(16))
    return lo, hi


def T(o, h):
    """One step: mix the exposed half, clock the hidden half, feed o0 back into h0.
    `o` and `h` are sequences of four uint32 arrays of one shape."""
    lo0, hi0 = _mul_wide(o[0], h[0] | U32(1))
    lo1, hi1 = _mul_wide(o[2], h[1] | U32(1))
    n0 = o[1] ^ hi1 ^ lo1
    n1 = _rotl(lo1, 16) ^ h[2]
    n2 = o[3] ^ hi0 ^ lo0
    n3 = _rotl(lo0, 16) ^ h[3]
    h0 = h[0] ^ _rotl(h[1], 7)
    h1 = h[1] ^ _rotl(h[2], 13)
    h2 = h[2] ^ _rotl(h[3], 22)
    h3 = h[3] ^ _rotl(h0, 3)
    h0 = (h0 + U32(CLOCK_WEYL)) ^ n0
    return (n0, n1, n2, n3), (h0, h1, h2, h3)


def F(o, h):
    """Eight rounds of T with a round constant and a half swap."""
    for rc in RC:
        o, h = T(o, h)
        o = (o[0] ^ U32(rc), o[1], o[2], o[3])
        o, h = h, o
    return o, h


def F_keyed(key, counter, domain, aux):
    """F from the keyed input block. `key` is four uint32 arrays, `counter` two uint32
    arrays (low, high) of one shape, `domain` and `aux` ints or uint32 arrays."""
    lo, hi = counter
    o = (lo, hi, jnp.broadcast_to(jnp.asarray(domain, U32), lo.shape), jnp.broadcast_to(jnp.asarray(aux, U32), lo.shape))
    h = tuple(jnp.broadcast_to(k, lo.shape) for k in key)
    return F(o, h)


def whiten(seed_lo, seed_hi):
    """The key of a 128-bit integer seed given as two uint64 or four uint32 words."""
    words = _u64_pair_to_words(seed_lo, seed_hi)
    o = tuple(jnp.asarray(x, U32) for x in (0, 0, DOMAIN_SEED, 0))
    o, _ = F(o, words)
    return jnp.stack(o)


def _u64_pair_to_words(lo, hi):
    """Four uint32 words of a 128-bit integer given as two 64-bit halves. Each half is an
    array of uint64, or a Python int when x64 is off."""

    def split(x):
        if isinstance(x, int):
            return U32(x & 0xFFFFFFFF), U32((x >> 32) & 0xFFFFFFFF)
        x = jnp.asarray(x)
        if x.dtype == jnp.uint64:
            return x.astype(U32), (x >> jnp.uint64(32)).astype(U32)
        return x.astype(U32), U32(0)

    return (*split(lo), *split(hi))


def child_keys(key, counter_lo, counter_hi, domain, aux, hidden):
    """Child keys (..., 4): half `hidden` (0 exposed, 1 hidden) of F(key, counter, domain, aux)."""
    o, h = F_keyed(key, (counter_lo, counter_hi), domain, aux)
    o, h = jnp.stack(o, -1), jnp.stack(h, -1)
    return jnp.where(hidden[..., None] != 0, h, o)


def rows(key, group0, ngroups, K):
    """Stream words of `ngroups` groups from `group0` at chunk length `K`, as a flat array
    in stream order: row by row, eight 128-bit blocks per row. The array may extend past the
    last group."""
    c = jnp.asarray(8 * group0, jnp.uint64 if jax.config.jax_enable_x64 else U32)
    c = c + jnp.arange(8 * ngroups, dtype=c.dtype)
    if c.dtype == jnp.uint64:
        counter = (c.astype(U32), (c >> jnp.uint64(32)).astype(U32))
    else:
        counter = (c, jnp.zeros_like(c))
    o, h = F_keyed(key, counter, DOMAIN_STREAM, AUX_STREAM)

    def step(state, _):
        o, h = T(*state)
        return (o, h), jnp.stack(o, -1)

    _, blocks = lax.scan(step, (o, h), None, length=K)
    # (K, 8 * ngroups, 4): step j, chunk 8g + lane, word. Row order is (g, j, lane, word).
    blocks = blocks.reshape(K, ngroups, 8, 4).transpose(1, 0, 2, 3)
    return blocks.reshape(-1)


def words_from(key, bit_position, n_words, K=32):
    """`n_words` uint32 stream words from a 32-bit aligned `bit_position` (static `n_words`)."""
    from . import _ffi

    def xla():
        word0 = bit_position >> 5
        row0 = word0 >> 5
        group0 = row0 // K
        nrows = (n_words + 31) // 32 + 1
        ngroups = (nrows + K - 1) // K + 1
        flat = rows(key, group0, ngroups, K)
        return lax.dynamic_slice(flat, (word0 - group0 * (32 * K),), (n_words,))

    supported = n_words > 0 and K >= _ffi.TILE_STEPS
    return _ffi.on_cuda(supported, lambda: _ffi.fill(jnp.asarray(key, U32), bit_position, n_words, U32, K, "stream"), xla)
