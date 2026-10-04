<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .jax"></p>

# tandem-jax

JAX key implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. Pure Python over
`jax.numpy` and `jax.lax`, so it runs on every XLA backend. It produces the stream the
specification defines, bit for bit.

## Use

```python
import jax, jax.numpy as jnp
import tandem_jax as tj

key = tj.key(42)                                   # the spec's generator for seed 42      
k1, k2 = jax.random.split(key)                     # the spec's split by index
step_key = jax.random.fold_in(key, 7)              # the spec's purpose 7
z = jax.random.normal(k1, (1000,))                 # any jax.random function
bits = jax.random.bits(key, (16,), jnp.uint32)     # the stream words from position 0

kid = tj.split(key, 2**40 + 5)                     # the spec's split child for any unsigned 64-bit index
x, pos = tj.stream(key, 0, 2**20, jnp.float64)     # the spec's Float64 draws, and the position after
y, pos = tj.stream(key, pos, 100, jnp.uint8)       # continue at that position, aligned per the spec
kids, pos = tj.fork(key, pos, 4)                   # the spec's fork at the current block, typed keys, jit-safe
```

`tj.split(key, index)` is the spec's split child for an unsigned index of any width, scalar
or array, traced or not, and returns typed keys. `jax.random.split(key, n)` gives children
`0..n-1`. A count above 2^32 needs `jax_enable_x64`, and without it `split` raises
`ValueError` instead of repeating keys.

`tj.fork(key, position, n)` returns `n` typed keys and the parent's new position. The position
may be a traced `uint64`, so it runs under `jit`. `tj.fork_words` returns the same children
as an `(n, 4)` array of `uint32` key words.

`tj.key(seed)` whitens the integer seed as the specification requires. `jax.random.split`
gives spec children `split(0), split(1), ...`, and `jax.random.fold_in(key, u)` gives the spec's
purpose child `sub(u)`. `jax.random.bits` reads the stream from position 0 with the spec's
alignment, so `bits(key, shape, uint32)` are the stream words and `uint8`, `uint16` and
`uint64` requests are the spec's narrower and wider draws.

`jax.random.uniform` applies JAX's own bit mapping: 52 random bits for `float64` and 23 for
`float32`. The specification keeps 53 and 24, as `(raw >> 11) * 2**-53` and
`(raw >> 8) * 2**-24`, so `uniform` does not equal the specification's Float64 draws.
`tj.stream(key, position, n, dtype)` applies the spec's mappings for `float16`, `float32`
and `float64`, and returns the raw aligned words for the unsigned integer types. The key
implementation is the canonical `Tandem8x32-K32`. `stream` takes `chunk_length` for the
other variants.

64-bit types need `jax.config.update("jax_enable_x64", True)`. Without it the 32-bit
multiplies are built from 16-bit halves, which also serves backends without 64-bit integers.

## Install

```sh
pip install .
```

For development, `pixi install` creates an environment with the package installed editable,
and `pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file, with a drift check in CI), compares positioned reads and
`jax.random.bits` with reference stream dumps in `tests/data`,
fork children at traced positions, split children for indices up to 2^64 - 1 against a direct evaluation of F,
and runs the key implementation under `jit` and `vmap`.

## Speed

Apple M4, XLA CPU backend, `pixi run bench`, 2^24 Float64 draws, minimum of seven runs after
a warm-up:

| | GiB/s |
|---|---|
| `tandem_jax.stream(key, 0, n, float64)` | 6.7 |
| `jax.random.uniform(tandem key, float64)` | 6.6 |
| `jax.random.uniform(threefry key, float64)` | 5.3 |
| `jax.random.uniform(rbg key, float64)` | 9.9 |

XLA runs the elementwise step over all chunks at once and uses several threads. The rbg
row is XLA's built-in generator op.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
