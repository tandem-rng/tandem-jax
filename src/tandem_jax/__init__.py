"""Tandem8x32 for JAX: a `jax.random` key implementation and positioned stream draws.

`key(seed, chunk_length=32)` is a typed JAX key whose `jax.random.bits`, `split` and
`fold_in` follow the specification: bits are the stream from position 0, `split` is the spec's split by index,
`fold_in` the spec's purpose derivation. `stream(key, position, n, dtype)` reads the stream
at a bit position with the spec's own float mappings, and `uniform` wraps it in the call shape
of `jax.random.uniform`. `jax.random.uniform` applies JAX's own mapping (52 random bits for
float64, 23 for float32), so it does not equal the spec's Float64 and Float32 draws; use
`uniform` or `stream` for those.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.random import define_prng_impl

from . import _core
from ._core import DOMAIN_FOLD, DOMAIN_FORK, DOMAIN_SPLIT, U32, F, F_keyed, T, whiten

__all__ = ["impl", "impl_for", "key", "key_data", "split", "sub", "stream", "uniform", "stream_normal", "normal", "stream_randint", "randint", "T", "F", "F_keyed", "whiten", "block", "fork", "fork_words"]

K = 32
"""The chunk length of the default key implementation: the canonical Tandem8x32-K32."""

_CHUNK_LENGTH = {}
"""Chunk length by implementation tag, so stream reads follow the key's own variant."""


def _seed(seed):
    seed = jnp.asarray(seed)
    if seed.dtype == jnp.int64 or seed.dtype == jnp.uint64:
        seed = seed.astype(jnp.uint64)
    else:
        seed = seed.astype(U32)
    return whiten(seed, 0)


def _words(x):
    """Low and high uint32 words of an unsigned integer array of any width."""
    x = jnp.asarray(x)
    if x.dtype.itemsize == 8:
        x = x.astype(jnp.uint64)
        return x.astype(U32), (x >> jnp.uint64(32)).astype(U32)
    return x.astype(U32), jnp.zeros(x.shape, U32)


def _split_words(key, index):
    """Child key words (..., 4) of the spec's split at an unsigned `index` array."""
    if jnp.asarray(index).dtype.itemsize == 8:
        index = jnp.asarray(index, jnp.uint64)
        hidden, half = (index & jnp.uint64(1)).astype(U32), index >> jnp.uint64(1)
    else:
        index = jnp.asarray(index, U32)
        hidden, half = index & U32(1), index >> U32(1)
    lo, hi = _words(half)
    return _core.child_keys(tuple(key), lo, hi, DOMAIN_SPLIT, 0, hidden)


def _split(key, shape):
    n = math.prod(shape)
    # Child indices past 2^32 need 64-bit arithmetic. Without x64 such a count cannot be
    # indexed, so refuse rather than wrap around and repeat keys.
    if n > 2**32 and not jax.config.jax_enable_x64:
        raise ValueError(f"splitting into {n} keys needs jax_enable_x64, the index exceeds 32 bits")
    i = jnp.arange(n, dtype=jnp.uint64 if n > 2**32 else U32)
    return _split_words(key, i).reshape(*shape, 4)


def _fold_in(key, data):
    lo, hi = _words(data)
    o, _ = F_keyed(tuple(key), (lo, hi), DOMAIN_FOLD, 0)
    return jnp.stack(o)


def _elements(words, bit_width, n):
    """`n` aligned `bit_width`-bit draws from stream words starting at a word boundary."""
    if bit_width == 32:
        return words[:n]
    if bit_width == 64:
        w = words[: 2 * n].reshape(n, 2).astype(jnp.uint64)
        return w[:, 0] | (w[:, 1] << jnp.uint64(32))
    per = 32 // bit_width
    w = words[: (n + per - 1) // per]
    shifts = U32(bit_width) * jnp.arange(per, dtype=U32)
    parts = (w[:, None] >> shifts) & U32((1 << bit_width) - 1)
    # Bits are held in uint8, there is no 1-bit integer type.
    return parts.reshape(-1)[:n].astype(jnp.dtype(f"uint{max(bit_width, 8)}"))


@functools.cache
def impl_for(chunk_length):
    """The key implementation of Tandem8x32-K`chunk_length`, a power of two from 1 to 65536.
    `impl_for(32)` is `impl`. The variants differ only in how `random_bits` and `stream`
    lay out the stream: split, fold_in and seeding do not depend on K, and children keep
    the parent's implementation."""
    K = int(chunk_length)
    if not 1 <= K <= 65536 or K & (K - 1):
        raise ValueError(f"chunk length must be a power of two from 1 to 65536, got {chunk_length}")

    def random_bits(key, bit_width, shape):
        n = math.prod(shape)
        n_words = (n * bit_width + 31) // 32
        words = _core.words_from(tuple(key), 0, n_words, K)
        return _elements(words, bit_width, n).reshape(shape)

    name, tag = ("tandem8x32", "tdm") if K == 32 else (f"tandem8x32-K{K}", f"tdm{K}")
    _CHUNK_LENGTH[tag] = K
    return define_prng_impl(
        key_shape=(4,), seed=_seed, split=_split, random_bits=random_bits, fold_in=_fold_in, name=name, tag=tag
    )


impl = impl_for(K)


def key(seed, chunk_length=K):
    """A typed key from an integer seed through the specification's seed whitening, so
    `key(42)` is the generator of Julia `Tandem8x32(42)` and C `tandem_seed(42, 0, 32)`.
    `chunk_length` selects the variant Tandem8x32-K`chunk_length`."""
    return jax.random.key(seed, impl=impl_for(chunk_length))


def split(k, index):
    """The spec's split child `index` of a key: `half(index & 1)` of `F(key, index >> 1,
    DOMAIN_SPLIT, 0)`. `index` is a scalar or array of any unsigned width, traced or not,
    and the result is a typed key of the same shape and variant. `jax.random.split(k, n)` gives
    children `0..n-1`."""
    return jax.random.wrap_key_data(_split_words(key_data(k), index), impl=impl_for(_chunk_length(k)))


def _is_typed(k):
    return jnp.issubdtype(jnp.asarray(k).dtype, jax.dtypes.prng_key)


def _chunk_length(k):
    return _CHUNK_LENGTH.get(str(jax.random.key_impl(k)), K) if _is_typed(k) else K


def sub(k, purpose):
    """The spec's purpose child of a key for an unsigned `purpose` of up to 64 bits, scalar
    or array. `jax.random.fold_in` hands the implementation a 32-bit value, so it covers
    purposes below 2^32 only."""
    lo, hi = _words(purpose)
    o, _ = F_keyed(tuple(key_data(k)), (lo, hi), DOMAIN_FOLD, 0)
    return jax.random.wrap_key_data(jnp.stack(o, -1), impl=impl_for(_chunk_length(k)))


def key_data(k):
    """The four key words of a typed key, or a (4,) uint32 array passed through."""
    if _is_typed(k):
        return jax.random.key_data(k)
    return jnp.asarray(k, U32)


_WIDTH = {
    jnp.dtype("bool"): 1,
    jnp.dtype("int8"): 8,
    jnp.dtype("int16"): 16,
    jnp.dtype("int32"): 32,
    jnp.dtype("int64"): 64,
    jnp.dtype("uint8"): 8,
    jnp.dtype("uint16"): 16,
    jnp.dtype("uint32"): 32,
    jnp.dtype("uint64"): 64,
    jnp.dtype("float16"): 16,
    jnp.dtype("float32"): 32,
    jnp.dtype("float64"): 64,
}


def _to_float(raw, dtype):
    if dtype == jnp.dtype("float64"):
        return (raw >> jnp.uint64(11)).astype(dtype) * dtype.type(2.0**-53)
    if dtype == jnp.dtype("float32"):
        return (raw >> U32(8)).astype(dtype) * dtype.type(2.0**-24)
    # Float16: the integer fits in 11 bits, so the convert and the scaling are exact.
    return (raw >> jnp.uint16(5)).astype(jnp.float32).astype(dtype) * dtype.type(2.0**-11)


def stream(k, position, n, dtype=jnp.float64, chunk_length=None):
    """`n` draws of `dtype` from stream bit `position`, with the spec's alignment and
    mappings: bool is one bit, signed integers reinterpret the unsigned draw, floats use
    the spec's scaling, and complex draws alternate real and imaginary components of the
    matching float type. Returns the draws and the position after them. `chunk_length`
    defaults to the variant of a typed key and to 32 for raw key words."""
    if chunk_length is None:
        chunk_length = _chunk_length(k)
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        part = jnp.dtype("float32" if dtype == jnp.complex64 else "float64")
        parts, pos = stream(k, position, 2 * n, part, chunk_length)
        return lax.complex(parts[0::2], parts[1::2]), pos
    if dtype not in _WIDTH:
        raise TypeError(f"stream does not support dtype {dtype}")
    w = _WIDTH[dtype]
    position = jnp.asarray(position, jnp.uint64 if jax.config.jax_enable_x64 else U32)
    aligned = (position + (w - 1)) & ~jnp.asarray(w - 1, position.dtype)
    # Elements narrower than a word may start inside the first word; read from its start.
    n_words = (n * w + 31) // 32 + (1 if w < 32 else 0)
    words = _core.words_from(key_data(k), aligned & ~jnp.asarray(31, position.dtype), n_words, chunk_length)
    if w < 32:
        offset = ((aligned & 31) // w).astype(jnp.int32)
        per = 32 // w
        raw = lax.dynamic_slice(_elements(words, w, n_words * per), (offset,), (n,))
    else:
        raw = _elements(words, w, n)
    if jnp.issubdtype(dtype, jnp.floating):
        out = _to_float(raw, dtype)
    elif dtype == jnp.bool_:
        out = raw != 0
    elif jnp.issubdtype(dtype, jnp.signedinteger):
        out = lax.bitcast_convert_type(raw, dtype)
    else:
        out = raw
    return out, aligned + w * n


def uniform(k, shape=(), dtype=None, position=0, *, minval=0.0, maxval=1.0):
    """Uniform draws of `shape` with the spec's float mappings, `(raw >> 11) * 2**-53` for
    float64, `(raw >> 8) * 2**-24` for float32 and `(raw >> 5) * 2**-11` for float16, read
    from stream bit `position`. Same call shape as `jax.random.uniform`, which applies
    JAX's own mapping and so returns different values. `dtype` defaults to the JAX float
    default. The result lies in [minval, maxval) up to rounding, as in JAX. Returns the
    array only, so use `stream` when the next position is needed."""
    dtype = jnp.dtype(dtype if dtype is not None else jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"uniform needs a floating dtype, got {dtype}")
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    x = stream(k, position, math.prod(shape), dtype)[0].reshape(shape)
    minval, maxval = jnp.asarray(minval, dtype), jnp.asarray(maxval, dtype)
    return jnp.maximum(minval, x * (maxval - minval) + minval)


def block(k, c, j):
    """Block B(c, j): the exposed half of chunk `c` after `j + 1` steps, as (4,) uint32."""
    key = tuple(key_data(k))
    c = int(c)
    o, h = F_keyed(key, (U32(c & 0xFFFFFFFF), U32(c >> 32)), _core.DOMAIN_STREAM, _core.AUX_STREAM)
    for _ in range(int(j) + 1):
        o, h = T(o, h)
    return jnp.stack(o)


def fork_words(k, position, n):
    """The `n` fork children of a key at a bit position as (n, 4) uint32 words, and the
    parent's new position. `position` may be traced, `n` is static."""
    if n > 2**32 and not jax.config.jax_enable_x64:
        raise ValueError(f"forking {n} keys needs jax_enable_x64, the index exceeds 32 bits")
    position = jnp.asarray(position, jnp.uint64 if jax.config.jax_enable_x64 else U32)
    b = position >> position.dtype.type(7)
    lo, hi = _words(b)
    i = jnp.arange(n, dtype=jnp.uint64 if n > 2**32 else U32)
    hidden, half = (i & i.dtype.type(1)).astype(U32), (i >> i.dtype.type(1)).astype(U32)
    ones = jnp.ones(n, U32)
    kids = _core.child_keys(tuple(key_data(k)), lo * ones, hi * ones, DOMAIN_FORK, half, hidden)
    return kids, (b + position.dtype.type(1)) << position.dtype.type(7)


def fork(k, position, n):
    """The `n` fork children of a key at a bit position as a typed key array of shape
    (n,), and the parent's new position. Works under `jit` with a traced position."""
    kids, new_position = fork_words(k, position, n)
    return jax.random.wrap_key_data(kids, impl=impl_for(_chunk_length(k))), new_position


# Appendix A of the specification: draws derived from the uniform stream. Not normative, but
# every port follows it so that they agree.

PURPOSE_BELOW32 = 0x424C573332
PURPOSE_BELOW64 = 0x424C573634
# A rejection has probability below range / 2^w, so few draws are rejected. Locating them
# with one nonzero over all elements costs several passes over the output. Counting per block
# of _BLOCK elements and gathering the few blocks that reject is one reduction instead.
_RETRY_MAX = 64
_BLOCK = 1024

_TWO_PI = {  # 2 pi as a high and low part, so the reduced angle is good to the last bit
    jnp.dtype("float64"): (6.283185307179586, 2.4492935982947064e-16),
    jnp.dtype("float32"): (6.2831855, -1.7484555e-7),
}


def _default_float():
    return jnp.dtype("float64" if jax.config.jax_enable_x64 else "float32")


def _position(position):
    return jnp.asarray(position, jnp.uint64 if jax.config.jax_enable_x64 else U32)


def stream_normal(k, position, n, dtype=None):
    """`n` standard normals from stream bit `position` by Box-Muller, and the position after
    them. Pair `j` is elements `2j` and `2j + 1`, `(r cos 2 pi b, r sin 2 pi b)` with
    `r = sqrt(-2 log(1 - a))`, from uniform draws `2j` (`a`) and `2j + 1` (`b`). The fill
    consumes `2 ceil(n / 2)` draws and an odd `n` drops the last sin half. float32 is computed
    in float32. `n = 0` leaves the position unchanged."""
    dtype = jnp.dtype(dtype if dtype is not None else _default_float())
    if dtype not in _TWO_PI:
        raise TypeError(f"normal needs float32 or float64, got {dtype}")
    if n == 0:
        return jnp.zeros(0, dtype), _position(position)
    u, pos = stream(k, position, 2 * ((n + 1) // 2), dtype)
    a, b = u[0::2], u[1::2]
    r = jnp.sqrt(-2 * jnp.log(1 - a))
    # Reduce to a quadrant and an angle in [-1/8, 1/8] of a turn first: b - q / 4 is exact, so
    # no precision is lost to the large angle.
    q = (b * 4 + 0.5).astype(jnp.int32)
    f = b - q.astype(dtype) * 0.25
    hi, lo = _TWO_PI[dtype]
    th = f * dtype.type(hi) + f * dtype.type(lo)
    c0, s0 = jnp.cos(th), jnp.sin(th)
    q = q & 3
    c = jnp.where(q == 0, c0, jnp.where(q == 1, -s0, jnp.where(q == 2, -c0, s0)))
    s = jnp.where(q == 0, s0, jnp.where(q == 1, c0, jnp.where(q == 2, -s0, -c0)))
    return jnp.stack([r * c, r * s], axis=-1).reshape(-1)[:n], pos


def normal(k, shape=(), dtype=None, position=0):
    """Standard normals of `shape` from stream bit `position`, as `stream_normal` returns
    them. Same call shape as `jax.random.normal`, with different values. Returns the array only."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return stream_normal(k, position, math.prod(shape), dtype)[0].reshape(shape)


def _mulhilo64(x, r):
    """Low and high 64-bit halves of the 128-bit product of two uint64 arrays."""
    mul = _core._mul_wide
    (x0, x1), (r0, r1) = _words(x), _words(r)
    ll0, ll1 = mul(x0, r0)
    lh0, lh1 = mul(x0, r1)
    hl0, hl1 = mul(x1, r0)
    hh0, hh1 = mul(x1, r1)
    u64 = jnp.uint64
    mid = ll1.astype(u64) + lh0.astype(u64) + hl0.astype(u64)
    lo = ll0.astype(u64) | (mid << u64(32))
    hi = (hh0.astype(u64) | (hh1.astype(u64) << u64(32))) + lh1.astype(u64) + hl1.astype(u64) + (mid >> u64(32))
    return lo, hi


def _lemire(x, r, t, w):
    """High word of the product and whether Lemire's method rejects the draw `x`."""
    lo, hi = _core._mul_wide(x, r) if w == 32 else _mulhilo64(x, r)
    return jnp.where(r == 0, 0, hi), (lo < t) & (r != 0)


def _threshold(r, w):
    zero, one = (U32(0), U32(1)) if w == 32 else (jnp.uint64(0), jnp.uint64(1))
    return (zero - r) % jnp.where(r == 0, one, r)


def _fallback_draw(child, d, w, K):
    """Draw `d` of the streams of the child keys `child` (four arrays), from position 0."""
    word = d if w == 32 else d * 2
    block, offset = word >> 2, word & 3
    row, lane = block >> 3, block & 7
    shift = K.bit_length() - 1
    chunk = (row >> shift) * 8 + lane
    shape = child[0].shape
    o, h = _core.F_keyed(child, (jnp.full(shape, chunk, U32), jnp.zeros(shape, U32)), _core.DOMAIN_STREAM, _core.AUX_STREAM)
    o, h = lax.fori_loop(jnp.int32(0), ((row & (K - 1)) + 1).astype(jnp.int32), lambda _, s: _core.T(*s), (o, h))
    words = jnp.stack(o)
    if w == 32:
        return words[offset]
    return words[offset].astype(jnp.uint64) | (words[offset + 1].astype(jnp.uint64) << jnp.uint64(32))


def _retry(key, idx, pending, r, t, w, K):
    """Bounded draws of the elements `idx` that took the fallback: element `i` retries on
    split(i) of sub(PURPOSE) of the fill's key at position 0, draw after draw until Lemire accepts."""
    purpose = PURPOSE_BELOW32 if w == 32 else PURPOSE_BELOW64
    lo, hi = U32(purpose & 0xFFFFFFFF), U32(purpose >> 32)
    o, _ = F_keyed(key, (lo, hi), DOMAIN_FOLD, 0)
    child = _split_words(o, idx)
    child = tuple(child[..., i] for i in range(4))

    def body(s):
        d, val, pend = s
        v, rej = _lemire(_fallback_draw(child, d, w, K), r, t, w)
        return d + U32(1), jnp.where(pend, v, val), pend & rej

    init = (U32(0), jnp.zeros(idx.shape, jnp.uint32 if w == 32 else jnp.uint64), pending)
    return lax.while_loop(lambda s: s[2].any(), body, init)[1]


def stream_randint(k, position, n, minval, maxval, dtype=None):
    """`n` integers uniform on [minval, maxval) from stream bit `position`, and the position
    after them, by Lemire's method as in Appendix A. Element `i` uses uniform draw `i`, 32 bits
    for dtypes up to 32 bits and 64 bits otherwise, and a rejected draw retries on a fallback
    stream. `n` draws are consumed. `maxval <= minval` gives `minval`. `n = 0` leaves the
    position unchanged. `minval` and `maxval` are scalars or arrays of length `n`."""
    dtype = jnp.dtype(dtype if dtype is not None else jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
    if not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"randint needs an integer dtype, got {dtype}")
    w = 64 if dtype.itemsize == 8 else 32
    wide = jnp.dtype(f"uint{w}")
    lo_s, hi_s = jnp.asarray(minval, dtype), jnp.asarray(maxval, dtype)
    lo_u, hi_u = lo_s.astype(wide), hi_s.astype(wide)
    r = jnp.where(hi_s > lo_s, hi_u - lo_u, wide.type(0))
    if n == 0:
        return jnp.zeros(0, dtype), _position(position)
    draws, pos = stream(k, position, n, wide)
    t = _threshold(r, w)
    plain, rej = _lemire(draws, r, t, w)
    chunk, key = _chunk_length(k), tuple(key_data(k))
    index = jnp.uint64 if n > 2**32 else U32
    if n > 2**32 and not jax.config.jax_enable_x64:
        raise ValueError(f"{n} elements need jax_enable_x64, the index exceeds 32 bits")
    pick = (lambda a, i: a[i]) if r.ndim else (lambda a, i: a)
    blocks = -(-n // _BLOCK)
    per_block = jnp.pad(rej, (0, blocks * _BLOCK - n)).reshape(blocks, _BLOCK)
    in_block = per_block.sum(1, dtype=jnp.int32)
    count = in_block.sum()
    cap = min(_RETRY_MAX, n)

    def few(_):
        # At most cap elements reject, so at most cap blocks do.
        rows = jnp.nonzero(in_block, size=cap, fill_value=0)[0]
        hit = per_block[rows] & (jnp.arange(cap) < (in_block > 0).sum())[:, None]
        flat = jnp.nonzero(hit.reshape(-1), size=cap, fill_value=0)[0]
        idx = (rows[flat // _BLOCK] * _BLOCK + flat % _BLOCK).astype(index)
        active = jnp.arange(cap) < count
        fixed = _retry(key, idx, active, pick(r, idx), pick(t, idx), w, chunk)
        return plain.at[jnp.where(active, idx, n)].set(fixed, mode="drop")

    def many(_):
        fixed = _retry(key, jnp.arange(n, dtype=index), rej, r, t, w, chunk)
        return jnp.where(rej, fixed, plain)

    branches = [lambda _: plain, few] + ([many] if n > _RETRY_MAX else [])
    val = lax.switch((count > 0).astype(jnp.int32) + (count > _RETRY_MAX).astype(jnp.int32), branches, None)
    return (lo_u + val).astype(dtype), pos


def randint(k, shape, minval, maxval, dtype=None, position=0):
    """Integers of `shape` uniform on [minval, maxval) from stream bit `position`, as
    `stream_randint` returns them. Same call shape as `jax.random.randint`, with different
    values. The bounds may be arrays that broadcast to `shape`. Returns the array only."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    flat = lambda a: a if jnp.ndim(a) == 0 else jnp.broadcast_to(a, shape).reshape(-1)
    return stream_randint(k, position, math.prod(shape), flat(minval), flat(maxval), dtype)[0].reshape(shape)
