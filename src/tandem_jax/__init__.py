"""Tandem8x32 for JAX: a `jax.random` key implementation and positioned stream draws.

`key(seed)` is a typed JAX key whose `jax.random.bits`, `split` and `fold_in` follow the
specification: bits are the stream from position 0, `split` is the spec's split by index,
`fold_in` the spec's purpose derivation. `stream(key, position, n, dtype)` reads the stream
at a bit position with the spec's own float mappings, and `uniform` wraps it in the call shape
of `jax.random.uniform`. `jax.random.uniform` applies JAX's own mapping (52 random bits for
float64, 23 for float32), so it does not equal the spec's Float64 and Float32 draws; use
`uniform` or `stream` for those.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.random import define_prng_impl

from . import _core
from ._core import DOMAIN_FOLD, DOMAIN_FORK, DOMAIN_SPLIT, U32, F, F_keyed, T, whiten

__all__ = ["impl", "key", "key_data", "split", "stream", "uniform", "T", "F", "F_keyed", "whiten", "block", "fork", "fork_words"]

K = 32
"""The chunk length of the key implementation: the canonical Tandem8x32-K32."""


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


def _random_bits(key, bit_width, shape):
    n = math.prod(shape)
    n_words = (n * bit_width + 31) // 32
    words = _core.words_from(tuple(key), 0, n_words, K)
    return _elements(words, bit_width, n).reshape(shape)


impl = define_prng_impl(
    key_shape=(4,),
    seed=_seed,
    split=_split,
    random_bits=_random_bits,
    fold_in=_fold_in,
    name="tandem8x32",
    tag="tdm",
)


def key(seed):
    """A typed key from an integer seed through the specification's seed whitening, so
    `key(42)` is the generator of Julia `Tandem8x32(42)` and C `tandem_seed(42, 0, 32)`."""
    return jax.random.key(seed, impl=impl)


def split(k, index):
    """The spec's split child `index` of a key: `half(index & 1)` of `F(key, index >> 1,
    DOMAIN_SPLIT, 0)`. `index` is a scalar or array of any unsigned width, traced or not,
    and the result is a typed key of the same shape. `jax.random.split(k, n)` gives
    children `0..n-1`."""
    return jax.random.wrap_key_data(_split_words(key_data(k), index), impl=impl)


def key_data(k):
    """The four key words of a typed key, or a (4,) uint32 array passed through."""
    if jnp.issubdtype(jnp.asarray(k).dtype, jax.dtypes.prng_key):
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


def stream(k, position, n, dtype=jnp.float64, chunk_length=K):
    """`n` draws of `dtype` from stream bit `position`, with the spec's alignment and
    mappings: bool is one bit, signed integers reinterpret the unsigned draw, floats use
    the spec's scaling, and complex draws alternate real and imaginary components of the
    matching float type. Returns the draws and the position after them."""
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
    return jax.random.wrap_key_data(kids, impl=impl), new_position
