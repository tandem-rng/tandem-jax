"""Tandem8x32 for JAX: a `jax.random` key implementation and positioned stream draws.

`key(seed)` is a typed JAX key whose `jax.random.bits`, `split` and `fold_in` follow the
specification: bits are the stream from position 0, `split` is the spec's split by index,
`fold_in` the spec's purpose derivation. `stream(key, position, n, dtype)` reads the stream
at a bit position with the spec's own float mappings. `jax.random.uniform` applies JAX's own
mapping (52 random bits for float64, 23 for float32), so it does not equal the spec's
Float64 and Float32 draws; use `stream` for those.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.random import define_prng_impl

from . import _core
from ._core import DOMAIN_FOLD, DOMAIN_FORK, DOMAIN_SPLIT, U32, F, F_keyed, T, whiten

__all__ = ["impl", "key", "key_data", "stream", "T", "F", "F_keyed", "whiten", "block", "fork"]

K = 32
"""The chunk length of the key implementation: the canonical Tandem8x32-K32."""


def _seed(seed):
    seed = jnp.asarray(seed)
    if seed.dtype == jnp.int64 or seed.dtype == jnp.uint64:
        seed = seed.astype(jnp.uint64)
    else:
        seed = seed.astype(U32)
    return whiten(seed, 0)


def _split(key, shape):
    n = math.prod(shape)
    i = jnp.arange(n, dtype=U32)
    children = _core.child_keys(tuple(key), i >> U32(1), jnp.zeros_like(i), DOMAIN_SPLIT, 0, i & U32(1))
    return children.reshape(*shape, 4)


def _fold_in(key, data):
    data = jnp.asarray(data)
    if data.dtype.itemsize == 8:
        data = data.astype(jnp.uint64)
        lo, hi = data.astype(U32), (data >> jnp.uint64(32)).astype(U32)
    else:
        lo, hi = data.astype(U32), U32(0)
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
    return parts.reshape(-1)[:n].astype(jnp.dtype(f"uint{bit_width}"))


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


def key_data(k):
    """The four key words of a typed key, or a (4,) uint32 array passed through."""
    if jnp.issubdtype(jnp.asarray(k).dtype, jax.dtypes.prng_key):
        return jax.random.key_data(k)
    return jnp.asarray(k, U32)


_WIDTH = {
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
    float mappings. Returns the draws and the position after them."""
    dtype = jnp.dtype(dtype)
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
    out = _to_float(raw, dtype) if jnp.issubdtype(dtype, jnp.floating) else raw
    return out, aligned + w * n


def block(k, c, j):
    """Block B(c, j): the exposed half of chunk `c` after `j + 1` steps, as (4,) uint32."""
    key = tuple(key_data(k))
    c = int(c)
    o, h = F_keyed(key, (U32(c & 0xFFFFFFFF), U32(c >> 32)), _core.DOMAIN_STREAM, _core.AUX_STREAM)
    for _ in range(int(j) + 1):
        o, h = T(o, h)
    return jnp.stack(o)


def fork(k, position, n):
    """The `n` fork children of a key at a bit position, (n, 4), and the parent's new position."""
    b = int(position) >> 7
    i = jnp.arange(n, dtype=U32)
    counter_lo, counter_hi = jnp.full_like(i, b & 0xFFFFFFFF), jnp.full_like(i, b >> 32)
    kids = _core.child_keys(tuple(key_data(k)), counter_lo, counter_hi, DOMAIN_FORK, i >> U32(1), i & U32(1))
    return kids, (b + 1) << 7
