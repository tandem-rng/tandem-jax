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
import importlib.resources
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.random import define_prng_impl

from . import _core, _ffi
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
    position = _position(position)
    aligned = _align(position, w)
    key = key_data(k)

    def xla():
        # Elements narrower than a word may start inside the first word; read from its start.
        n_words = (n * w + 31) // 32 + (1 if w < 32 else 0)
        words = _core.words_from(key, aligned & ~jnp.asarray(31, position.dtype), n_words, chunk_length)
        if w < 32:
            offset = ((aligned & 31) // w).astype(jnp.int32)
            per = 32 // w
            raw = lax.dynamic_slice(_elements(words, w, n_words * per), (offset,), (n,))
        else:
            raw = _elements(words, w, n)
        if jnp.issubdtype(dtype, jnp.floating):
            return _to_float(raw, dtype)
        if dtype == jnp.bool_:
            return raw != 0
        if jnp.issubdtype(dtype, jnp.signedinteger):
            return lax.bitcast_convert_type(raw, dtype)
        return raw

    # The CUDA kernels write 32- and 64-bit draws as the dtype; narrower ones come from the word
    # fill. tandem-c fills every width but single bits.
    cuda = w >= 32 and n > 0 and chunk_length >= _ffi.TILE_STEPS
    out = _ffi.native(cuda, w >= 8 and n > 0, lambda: _ffi.fill(key, aligned, n, dtype, chunk_length, "stream"), xla)
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
    # On [0, 1) the map below is the identity, and XLA would still spend a pass over x on it.
    if isinstance(minval, (int, float)) and isinstance(maxval, (int, float)) and (minval, maxval) == (0, 1):
        return x
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

# 2 pi as a high and low part, so the reduced angle of the float32 normals is good to the last bit.
_TWO_PI_F32 = (6.2831855, -1.7484555e-7)


def _default_float():
    return jnp.dtype("float64" if jax.config.jax_enable_x64 else "float32")


def _position(position):
    return jnp.asarray(position, jnp.uint64 if jax.config.jax_enable_x64 else U32)


def _align(position, w):
    return (position + (w - 1)) & ~jnp.asarray(w - 1, position.dtype)


def stream_normal(k, position, n, dtype=None):
    """`n` standard normals from stream bit `position`, and the position after them.

    float64: the 1024-layer ziggurat of Appendix A, element `i` from 64-bit draw `i`, `n` draws
    in all. A draw that misses the inner rectangles continues on its own fallback stream, keyed
    by its index in the key's stream. `n = 0` aligns the position to 64 bits.

    float32: Box-Muller in float32. Pair `j` is elements `2j` and `2j + 1`,
    `(r cos 2 pi b, r sin 2 pi b)` with `r = sqrt(-2 log(1 - a))`, from uniform draws `2j` (`a`)
    and `2j + 1` (`b`). The fill consumes `2 ceil(n / 2)` draws and an odd `n` drops the last sin
    half. `n = 0` leaves the position unchanged."""
    dtype = jnp.dtype(dtype if dtype is not None else _default_float())
    if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise TypeError(f"normal needs float32 or float64, got {dtype}")
    position = _position(position)
    if n == 0:
        return jnp.zeros(0, dtype), _align(position, 64) if dtype.itemsize == 8 else position
    return _normal(key_data(k), position, n, dtype, _chunk_length(k))


@functools.partial(jax.jit, static_argnames=("n", "dtype", "chunk"))
def _normal(key, position, n, dtype, chunk):
    call = lambda: _ffi.fill(key, position, n, dtype, chunk, "normal")
    if dtype.itemsize == 8:
        pos = _align(position, 64) + 64 * n
        return _ffi.native(True, True, call, lambda: _ziggurat(key, position, n, chunk)), pos
    draws = 2 * ((n + 1) // 2)
    pos = _align(position, 32) + 32 * draws
    return _ffi.native(True, True, call, lambda: _box_muller(key, position, draws, chunk)[:n]), pos


def _box_muller(key, position, draws, chunk):
    # Traced under _normal's jit, so the uniforms, log, sqrt, sin and cos fuse into one pass.
    u, _ = stream(key, position, draws, jnp.float32, chunk)
    a, b = u[0::2], u[1::2]
    r = jnp.sqrt(-2 * jnp.log(1 - a))
    # Reduce to a quadrant and an angle in [-1/8, 1/8] of a turn first: b - q / 4 is exact, so
    # no precision is lost to the large angle.
    q = (b * 4 + 0.5).astype(jnp.int32)
    f = b - q.astype(jnp.float32) * 0.25
    hi, lo = _TWO_PI_F32
    th = f * jnp.float32(hi) + f * jnp.float32(lo)
    c0, s0 = jnp.cos(th), jnp.sin(th)
    q = q & 3
    c = jnp.where(q == 0, c0, jnp.where(q == 1, -s0, jnp.where(q == 2, -c0, s0)))
    s = jnp.where(q == 0, s0, jnp.where(q == 1, c0, jnp.where(q == 2, -s0, -c0)))
    return jnp.stack([r * c, r * s], axis=-1).reshape(-1)


# The float64 ziggurat of Appendix A. Its values are exact across ports only when every operation
# rounds as the spec says, so the reference logarithm's fused multiply-adds are emulated exactly
# and no other product may fuse into a sum.

PURPOSE_NORMAL64 = 0x4E524D3634
_LN2_HI, _LN2_LO = 1.3862943607382476, 3.816429394731813e-10  # 2 ln 2 split, nk * hi is exact
_LOG_C = (0.08312363319426472, 0.09070001083303751, 0.11111433317907482, 0.14285712049336274,
          0.2000000000566491, 0.33333333333331017)  # c6 down to c1


@functools.cache
def _zig_tables():
    """W for a clear sign bit at i and -W at 1024 + i, so the low 11 bits of a draw index it, then
    K, Y and R, from the spec's tables/normal_f64_zig1024.json."""
    d = json.loads(importlib.resources.files(__package__).joinpath("normal_f64_zig1024.json").read_text())
    w = np.array([float.fromhex(v) for v in d["W"]])
    y = np.array([float.fromhex(v) for v in d["Y"]])
    return np.concatenate([w, -w]), np.array(d["K"], np.uint64), y, float.fromhex(d["R"])


def _alone(p):
    """The product `p` rounded on its own. XLA's CPU backend fuses a product into the sum that
    reads it, which rounds once where the spec rounds twice. A select hides the product from
    that fusion, and no product here is NaN."""
    return jnp.where(jnp.isnan(p), 0.0, p)


def _opaque(c, like):
    """The constant `c` as a value XLA cannot see, since its simplifier rewrites the error-free
    sums below when one operand is a literal. `like` is never NaN."""
    return jnp.where(jnp.isnan(like), 0.0, c)


def _two_sum(a, b):
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


def _halves(a):
    """`a` as a 26-bit high part, rounded on the bits, and the exact rest of at most 26 bits."""
    bits = lax.bitcast_convert_type(a, jnp.uint64)
    hi = lax.bitcast_convert_type((bits + jnp.uint64(1 << 26)) & ~jnp.uint64((1 << 27) - 1), jnp.float64)
    return hi, a - hi


def _two_prod(a, b):
    """Dekker's product: `p + e == a * b` exactly. The partial products are exact, so a fused
    multiply-add over them gives the same bits."""
    a, b = jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64)
    p = _alone(a * b)
    ah, al = _halves(a)
    bh, bl = _halves(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _fma(a, b, c):
    """`a * b + c` rounded once, by Boldo and Melquiond's emulation with a sum rounded to odd
    (IEEE Trans. Computers 57, 2008). It needs no underflow, which the logarithm's operands avoid."""
    b, c = _opaque(b, a), _opaque(c, a)
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    s, e = _two_sum(tl, ul)
    bits = lax.bitcast_convert_type(s, jnp.uint64)
    # Round to odd: an inexact sum with an even last bit moves one ulp toward the exact one.
    step = jnp.where((e > 0) == (s > 0), jnp.uint64(1), jnp.uint64(2**64 - 1))
    bits = jnp.where((e != 0) & ((bits & jnp.uint64(1)) == 0), bits + step, bits)
    return th + lax.bitcast_convert_type(bits, jnp.float64)


def _neg2_log(x):
    """The spec's reference logarithm, -2 ln x for float64 x in (0, 1]."""
    u64 = jnp.uint64
    ix = lax.bitcast_convert_type(x, u64) + u64(0x00095F6200000000)
    nk = (1023 - (ix >> u64(52)).astype(jnp.int64)).astype(jnp.float64)
    m = lax.bitcast_convert_type((ix & u64(0x000FFFFFFFFFFFFF)) + u64(0x3FE6A09E00000000), jnp.float64)
    s = (m - 1.0) / (m + 1.0)
    z = _alone(s * s)
    p = _LOG_C[0]
    for c in _LOG_C[1:] + (1.0,):
        p = _fma(z, p, c)
    return _fma(nk, _LN2_LO, _fma(nk, _LN2_HI, _alone((s * -4.0) * p)))


def _zig_fast(r):
    """The fast path of a 64-bit draw `r`: its value and whether it missed the inner rectangles."""
    W, K, _, _ = _zig_tables()
    i = (r & jnp.uint64(2047)).astype(jnp.int32)
    ra = r >> jnp.uint64(11)
    return ra.astype(jnp.float64) * jnp.asarray(W)[i], ra >= jnp.asarray(K)[i & 1023]


_WEDGE, _REDRAW, _TAIL_A, _TAIL_B = range(4)
_ZIG_SHARE = 128
"""The slow path takes up to one element in this many, 1.8 times the expected misses."""
_ZIG_GROUP = 16
"""Misses are gathered from groups of this many elements, so that one pass of the gather runs
over a sixteenth of the fill and the other over an eighth."""


def _zig_slow(key, g, r, pending, K):
    """The misses with draws `r` at global draw indices `g`, each on its fallback stream split(g)
    of sub(PURPOSE_NORMAL64). Every pending element takes one fallback draw a round, as the
    phase it is in needs: a wedge test, a fresh draw, or one of the two tail draws."""
    _, _, Y, R = _zig_tables()
    Y = jnp.asarray(Y)
    lo, hi = U32(PURPOSE_NORMAL64 & 0xFFFFFFFF), U32(PURPOSE_NORMAL64 >> 32)
    o, _ = F_keyed(key, (lo, hi), DOMAIN_FOLD, 0)
    child = _split_words(o, g)
    child = tuple(child[..., w] for w in range(4))
    x, _ = _zig_fast(r)
    layer = lambda r: (r & jnp.uint64(1023)).astype(jnp.int32)
    phase = jnp.where(layer(r) == 0, _TAIL_A, _WEDGE)
    zero = jnp.zeros(g.shape)

    def body(st):
        d, blk, phase, r, x, a, out, pend = st
        # Draws 2b and 2b + 1 share stream block b, so an odd draw reuses the block of the last.
        blk = lax.cond((d & 1) == 0, lambda: _fallback_block(child, d >> 1, K), lambda: blk)
        odd = (d & 1) == 1
        v = jnp.where(odd, blk[2], blk[0]).astype(jnp.uint64) | (jnp.where(odd, blk[3], blk[1]).astype(jnp.uint64) << jnp.uint64(32))
        u = (v >> jnp.uint64(11)).astype(jnp.float64) * 2.0**-53
        i = layer(r)
        R_ = _opaque(R, u)
        y = Y[i] + _alone(u * (Y[i + 1] - Y[i]))
        lg = 0.5 * _neg2_log(jnp.where(phase == _WEDGE, y, 1.0 - u))  # -ln
        x2, miss2 = _zig_fast(v)
        tail = jnp.where((r >> jnp.uint64(10)) & jnp.uint64(1) == 1, -(R_ + a), R_ + a)
        done = jnp.select(
            [phase == _WEDGE, phase == _REDRAW, phase == _TAIL_B],
            [-lg < -0.5 * (x * x), ~miss2, lg + lg >= a * a],
            False,
        )
        val = jnp.select([phase == _WEDGE, phase == _REDRAW], [x, x2], tail)
        after = jnp.select(
            [phase == _WEDGE, phase == _REDRAW, phase == _TAIL_A],
            [_REDRAW, jnp.where(layer(v) == 0, _TAIL_A, _WEDGE), _TAIL_B],
            _TAIL_A,
        )
        redraw = pend & (phase == _REDRAW)
        return (
            d + U32(1),
            blk,
            jnp.where(pend, after, phase),
            jnp.where(redraw, v, r),
            jnp.where(redraw, x2, x),
            jnp.where(pend & (phase == _TAIL_A), lg / R_, a),
            jnp.where(pend & done, val, out),
            pend & ~done,
        )

    init = (U32(0), jnp.zeros((4,) + g.shape, U32), phase, r, x, zero, zero, pending)
    return lax.while_loop(lambda st: st[-1].any(), body, init)[6]


def _ziggurat(key, position, n, chunk):
    r, pos = stream(key, position, n, jnp.uint64, chunk)
    x, miss = _zig_fast(r)
    g0 = pos // jnp.uint64(64) - jnp.uint64(n)
    key = tuple(key)
    # 0.43 % of the draws miss. The slow path runs on up to cap of them, gathered in two passes:
    # the groups that hold a miss, then the misses of those groups, of which there are no more
    # than misses. When more miss than cap, it runs on every element.
    cap = min(n, max(_RETRY_MAX, n // _ZIG_SHARE))
    G = _ZIG_GROUP
    groups = -(-n // G)
    by_group = jnp.pad(miss, (0, groups * G - n)).reshape(groups, G)
    hit = by_group.any(1)
    count, ngroups = miss.sum(), hit.sum()
    gcap = min(groups, cap)

    def few(_):
        rows = jnp.nonzero(hit, size=gcap, fill_value=0)[0]
        sel = by_group[rows] & (jnp.arange(gcap) < ngroups)[:, None]
        flat = jnp.nonzero(sel.reshape(-1), size=cap, fill_value=0)[0]
        idx = rows[flat // G] * G + flat % G
        active = jnp.arange(cap) < count
        fixed = _zig_slow(key, g0 + idx.astype(jnp.uint64), r[idx], active, chunk)
        return x.at[jnp.where(active, idx, n)].set(fixed, mode="drop")

    def many(_):
        fixed = _zig_slow(key, g0 + jnp.arange(n, dtype=jnp.uint64), r, miss, chunk)
        return jnp.where(miss, fixed, x)

    branches = [lambda _: x, few] + ([many] if n > cap else [])
    return lax.switch((count > 0).astype(jnp.int32) + (count > cap).astype(jnp.int32), branches, None)


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


def _fallback_block(child, block, K):
    """Stream block `block` (four words) of the child keys `child` (four arrays), from position 0."""
    row, lane = block >> 3, block & 7
    shift = K.bit_length() - 1
    chunk = (row >> shift) * 8 + lane
    shape = child[0].shape
    o, h = _core.F_keyed(child, (jnp.full(shape, chunk, U32), jnp.zeros(shape, U32)), _core.DOMAIN_STREAM, _core.AUX_STREAM)
    o, h = lax.fori_loop(jnp.int32(0), ((row & (K - 1)) + 1).astype(jnp.int32), lambda _, s: _core.T(*s), (o, h))
    return jnp.stack(o)


def _fallback_draw(child, d, w, K):
    """Draw `d` of the streams of the child keys `child` (four arrays), from position 0."""
    word = d if w == 32 else d * 2
    offset = word & 3
    words = _fallback_block(child, word >> 2, K)
    if w == 32:
        return words[offset]
    return words[offset].astype(jnp.uint64) | (words[offset + 1].astype(jnp.uint64) << jnp.uint64(32))


def _retry(key, g, pending, r, t, w, K):
    """Bounded draws of the elements with global draw index `g` that took the fallback: each
    retries on split(g) of sub(PURPOSE) of the fill's key at position 0, draw after draw until
    Lemire accepts."""
    purpose = PURPOSE_BELOW32 if w == 32 else PURPOSE_BELOW64
    lo, hi = U32(purpose & 0xFFFFFFFF), U32(purpose >> 32)
    o, _ = F_keyed(key, (lo, hi), DOMAIN_FOLD, 0)
    child = _split_words(o, g)
    child = tuple(child[..., i] for i in range(4))

    def body(s):
        d, val, pend = s
        v, rej = _lemire(_fallback_draw(child, d, w, K), r, t, w)
        return d + U32(1), jnp.where(pend, v, val), pend & rej

    init = (U32(0), jnp.zeros(g.shape, jnp.uint32 if w == 32 else jnp.uint64), pending)
    return lax.while_loop(lambda s: s[2].any(), body, init)[1]


def _bounded(k, position, n, lo, r, full, w):
    """`n` draws of width `w` on [0, r) plus `lo`, in the unsigned type of `lo`, and the position
    after. `full` marks the range 2^bits of `r`'s type, which `r` holds as 0."""
    wide = jnp.dtype(f"uint{w}")
    if r.dtype.itemsize * 8 > w:
        # A range of 2^32 reaches 32-bit draws only from a 64-bit type: every draw is a value.
        full = r == 2**32
    elif r.dtype.itemsize * 8 < w:
        r = r.astype(wide) + (jnp.asarray(full, wide) << wide.type(32))
        full = False
    r = r.astype(wide)
    draws, pos = stream(k, position, n, wide)
    t = _threshold(r, w)
    plain, rej = _lemire(draws, r, t, w)
    chunk, key = _chunk_length(k), tuple(key_data(k))
    # The fallback is keyed by the draw's index in the key's stream, so a fill cut at any
    # element boundary equals the whole fill.
    g0 = pos // pos.dtype.type(w) - pos.dtype.type(n)
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
        idx = (rows[flat // _BLOCK] * _BLOCK + flat % _BLOCK).astype(pos.dtype)
        active = jnp.arange(cap) < count
        fixed = _retry(key, g0 + idx, active, pick(r, idx), pick(t, idx), w, chunk)
        return plain.at[jnp.where(active, idx, n)].set(fixed, mode="drop")

    def many(_):
        fixed = _retry(key, g0 + jnp.arange(n, dtype=pos.dtype), rej, r, t, w, chunk)
        return jnp.where(rej, fixed, plain)

    branches = [lambda _: plain, few] + ([many] if n > _RETRY_MAX else [])
    val = lax.switch((count > 0).astype(jnp.int32) + (count > _RETRY_MAX).astype(jnp.int32), branches, None)
    if full is not False:
        val = jnp.where(full, draws, val)
    return lo + val.astype(lo.dtype), pos


def _bound(v, dtype):
    """A bound in `dtype`, and whether it lay above the dtype's range. A Python int outside the
    range clips to it, as in `jax.random.randint`."""
    if isinstance(v, int):
        info = jnp.iinfo(dtype)
        return jnp.asarray(min(max(v, info.min), info.max), dtype), v > info.max
    return jnp.asarray(v, dtype), False


def stream_randint(k, position, n, minval, maxval, dtype=None, width=None):
    """`n` integers uniform on [minval, maxval) from stream bit `position`, and the position
    after them, by Lemire's method as in Appendix A. Element `i` uses uniform draw `i`. The
    draw width follows the range, 32 bits when it is at most 2^32 and 64 otherwise, so `dtype`
    does not change the values. A rejected draw retries on a fallback stream keyed by the draw's
    index in the key's stream. `n` draws are consumed. `maxval <= minval` gives `minval`. A
    Python int `maxval` above the dtype's range includes its largest value, as in
    `jax.random.randint`. `n = 0` leaves the position unchanged. `minval` and `maxval` are
    scalars or arrays of length `n`; the width then follows the largest range. `width` of 32 or
    64 names the draw width instead, as the `u32` and `u64` fills of the C and CUDA ports do, and
    32 needs ranges of at most 2^32."""
    dtype = jnp.dtype(dtype if dtype is not None else jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
    if not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"randint needs an integer dtype, got {dtype}")
    if width not in (None, 32, 64):
        raise ValueError(f"width must be 32 or 64, got {width}")
    wide = jnp.dtype("uint64" if dtype.itemsize == 8 else "uint32")
    (lo_s, _), (hi_s, over) = _bound(minval, dtype), _bound(maxval, dtype)
    lo = lo_s.astype(wide)
    if over:
        r = hi_s.astype(wide) - lo + wide.type(1)
        # Only the whole range of a 32- or 64-bit dtype wraps r to 0.
        full = (r == 0) if dtype.itemsize == wide.itemsize else False
    else:
        r, full = jnp.where(hi_s > lo_s, hi_s.astype(wide) - lo, wide.type(0)), False
    if width == 32 and wide.itemsize == 8:
        try:
            too_wide = bool(jnp.any((r > wide.type(2**32)) | full))
        except jax.errors.ConcretizationTypeError:
            too_wide = False
        if too_wide:
            raise ValueError("width 32 needs ranges of at most 2^32")
    if n == 0:
        return jnp.zeros(0, dtype), _position(position)

    def xla():
        if width is not None or wide.itemsize == 4:
            val, pos = _bounded(k, position, n, lo, r, full, width or 32)
        else:
            wide_range = jnp.any((r > wide.type(2**32)) | full)
            try:
                val, pos = _bounded(k, position, n, lo, r, full, 64 if bool(wide_range) else 32)
            except jax.errors.TracerBoolConversionError:
                val, pos = lax.cond(
                    wide_range,
                    lambda: _bounded(k, position, n, lo, r, full, 64),
                    lambda: _bounded(k, position, n, lo, r, full, 32),
                )
        return val.astype(dtype), pos

    def native():
        # Width 0 leaves the choice to the kernel, which reads the range, so no conditional runs.
        w = width or (32 if wide.itemsize == 4 else 0)
        # The kernels store 32- or 64-bit outputs at least as wide as the draw.
        size = max(w, 32)
        out = dtype if dtype.itemsize * 8 >= size else jnp.dtype(("int" if dtype.kind == "i" else "uint") + str(size))
        p = _position(position)
        val = _ffi.fill(key_data(k), p, n, out, _chunk_length(k), "below", r, lo, w).astype(dtype)
        if w:
            return val, _align(p, w) + w * n
        return val, jnp.where(r > wide.type(2**32), _align(p, 64) + 64 * n, _align(p, 32) + 32 * n)

    # Per-element bounds take the XLA path, the native fills take one range per fill. They read
    # a range of 0 as empty, so a dtype's whole range takes the XLA path too.
    supported = jnp.ndim(lo) == 0 and jnp.ndim(r) == 0 and full is False
    return _ffi.native(supported and _chunk_length(k) >= _ffi.TILE_STEPS, supported, native, xla)


def randint(k, shape, minval, maxval, dtype=None, position=0, width=None):
    """Integers of `shape` uniform on [minval, maxval) from stream bit `position`, as
    `stream_randint` returns them. Same call shape as `jax.random.randint`, with different
    values. The bounds may be arrays that broadcast to `shape`. Returns the array only."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    flat = lambda a: a if jnp.ndim(a) == 0 else jnp.broadcast_to(a, shape).reshape(-1)
    return stream_randint(k, position, math.prod(shape), flat(minval), flat(maxval), dtype, width)[0].reshape(shape)
