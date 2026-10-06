# API

## Use

```python
import jax, jax.numpy as jnp
import tandem_jax as tj

key = tj.key(42)                                   # the spec's generator for seed 42
k1, k2 = jax.random.split(key)                     # the spec's split by index
step_key = jax.random.fold_in(key, 7)              # the spec's purpose 7
z = jax.random.normal(k1, (1000,))                 # any jax.random function
bits = jax.random.bits(key, (16,), jnp.uint32)     # the stream words from position 0

key8 = tj.key(42, chunk_length=8)                  # the Tandem8x32-K8 variant
kid = tj.split(key, 2**40 + 5)                     # the spec's split child for any unsigned 64-bit index
u = tj.uniform(key, (1000,), jnp.float32)          # like jax.random.uniform, with the spec's mapping
z = tj.normal(key, (1000,), jnp.float32)           # Box-Muller normals, Appendix A
r = tj.randint(key, (1000,), 0, 6, jnp.int32)      # Lemire bounded integers, Appendix A
e = tj.exponential(key, (1000,), jnp.float32)      # exponentials -ln(1 - u), Appendix A
x, pos = tj.stream(key, 0, 2**20, jnp.float64)     # the spec's Float64 draws, and the position after
y, pos = tj.stream(key, pos, 100, jnp.uint8)       # continue at that position, aligned per the spec
kids, pos = tj.fork(key, pos, 4)                   # the spec's fork at the current block, typed keys, jit-safe
```

## Reference

`tj.key(seed)` whitens the integer seed as the specification requires. `jax.random.split`
gives spec children `split(0), split(1), ...`, and `jax.random.fold_in(key, u)` gives the spec's
purpose child `sub(u)`. `jax.random.bits` reads the stream from position 0 with the spec's
alignment, so `bits(key, shape, uint32)` are the stream words and `uint8`, `uint16` and
`uint64` requests are the spec's narrower and wider draws.

The key implementation `tj.impl` is the canonical `Tandem8x32-K32`. `tj.impl_for(K)` gives the
implementation of `Tandem8x32-K<K>` for a power of two from 1 to 65536, and
`tj.key(seed, chunk_length=K)` makes a typed key of that variant, with `K = 32` by default. Children from `split`, `fork`
and `stream` follow the parent's variant, and `stream` takes `chunk_length` for raw key words.

`tj.split(key, index)` is the spec's split child for an unsigned index of any width, scalar
or array, traced or not, and returns typed keys. `jax.random.split(key, n)` gives children
`0..n-1`. A count above 2^32 needs `jax_enable_x64`, and without it `split` raises
`ValueError` instead of repeating keys.

`tj.sub(key, purpose)` is the purpose child for a purpose of up to 64 bits. `jax.random.fold_in`
passes the implementation a 32-bit value, so it covers purposes below 2^32 only.

`tj.fork(key, position, n)` returns `n` typed keys and the parent's new position. The position
may be a traced `uint64`, so it runs under `jit`. `tj.fork_words` returns the same children
as an `(n, 4)` array of `uint32` key words.

`jax.random.uniform` applies JAX's own bit mapping: 52 random bits for `float64` and 23 for
`float32`. The specification keeps 53 and 24, as `(raw >> 11) * 2**-53` and
`(raw >> 8) * 2**-24`, so `jax.random.uniform` does not equal the specification's Float64 draws.
`tj.uniform(key, shape, dtype, position=0, *, minval, maxval)` is the drop-in with the
specification's mapping, including `float16` as `(raw >> 5) * 2**-11`. It reads from the stream
at `position` and returns the array only.
`tj.stream(key, position, n, dtype)` applies the spec's mappings for `float16`, `float32`
and `float64`, returns `bool` as single bits, returns the signed integer types by
reinterpreting the unsigned draw in two's complement, returns the unsigned types as the raw
aligned words, and returns `complex64` and `complex128` as alternating real and imaginary
`float32` or `float64` draws.

`tj.normal(key, shape, dtype, position=0)` and `tj.randint(key, shape, minval, maxval, dtype,
position=0, width=None)` follow Appendix A of the specification, so every port returns the same
values. `stream_normal(key, position, n, dtype)` and `stream_randint(key, position, n, minval,
maxval, dtype, width=None)` return the draws and the position after them, which is aligned to
the draw width plus `n` draws for `float64` normals and bounded integers and `2 * ceil(n / 2)`
draws for `float32` normals. For `n = 0` a `float64` normal fill aligns the position to 64 bits
and the others leave it unchanged. `maxval <= minval` gives `minval`. As in
`jax.random.randint`, Python int bounds outside the dtype clip to it, and a `maxval` above it
includes the largest value, so `randint(key, n, 0, 256, jnp.uint8)` gives every byte.
`width=32` needs ranges of at most 2^32. Bounded integers, uniforms and `float64` normals are
exact across ports. `float32` normals agree to the tolerance of Appendix A. Neither equals
`jax.random.normal` or `jax.random.randint`.

`tj.exponential(key, shape, dtype, position=0)` and `stream_exponential(key, position, n, dtype)`
give `-ln(1 - u)` of one uniform draw each, as Appendix A defines them, exact across ports in
`float64` and `float32`. `n = 0` leaves the position unchanged.

`tj.choice_table(weights)` builds the integer alias table of
[Appendix C](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-c-weighted-choice-non-normative)
on the host from finite, nonnegative weights, not all zero. `tj.choice(key, shape, table,
position=0)` and `stream_choice(key, position, n, table)` draw `uint32` indices with probability
proportional to the weights. Element `i` maps `uint64` draw `i` by integer operations only and
never retries, so the indices are exact across ports and `n = 0` aligns the position to 64 bits.
Unlike `jax.random.choice`, they draw with replacement from a prebuilt table. They run on the XLA
path on every backend and need `jax_enable_x64`.

A fill from a Python integer position whose end would reach 2^64 raises `ValueError`. Array and
traced positions are not checked.

64-bit types need `jax.config.update("jax_enable_x64", True)`.

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the position of
their first element, or draw from `split(task)`, reproduce a serial run for any decomposition,
as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
