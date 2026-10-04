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

key8 = tj.key(42, chunk_length=8)                  # the Tandem8x32-K8 variant
kid = tj.split(key, 2**40 + 5)                     # the spec's split child for any unsigned 64-bit index
u = tj.uniform(key, (1000,), jnp.float32)          # like jax.random.uniform, with the spec's mapping
z = tj.normal(key, (1000,), jnp.float32)           # Box-Muller normals, Appendix A
r = tj.randint(key, (1000,), 0, 6, jnp.int32)      # Lemire bounded integers, Appendix A
x, pos = tj.stream(key, 0, 2**20, jnp.float64)     # the spec's Float64 draws, and the position after
y, pos = tj.stream(key, pos, 100, jnp.uint8)       # continue at that position, aligned per the spec
kids, pos = tj.fork(key, pos, 4)                   # the spec's fork at the current block, typed keys, jit-safe
```

`tj.split(key, index)` is the spec's split child for an unsigned index of any width, scalar
or array, traced or not, and returns typed keys. `jax.random.split(key, n)` gives children
`0..n-1`. A count above 2^32 needs `jax_enable_x64`, and without it `split` raises
`ValueError` instead of repeating keys.

`tj.normal(key, shape, dtype, position=0)` and `tj.randint(key, shape, minval, maxval, dtype,
position=0, width=None)` follow Appendix A of the specification, so every port returns the same
values. `stream_normal(key, position, n, dtype)` and `stream_randint(key, position, n, minval,
maxval, dtype, width=None)` return the draws and the position after them, which is aligned to
the draw width plus `2 * ceil(n / 2)` draws for normals and `n` draws for bounded integers, and
unchanged for `n = 0`. Normals are Box-Muller pairs: elements `2j` and `2j + 1` are
`r cos 2 pi b` and `r sin 2 pi b` from uniform draws `2j` and `2j + 1`, computed in `float32` or
`float64` as the dtype says, in one fused `jit`. Bounded integers use Lemire's method on draw `i`
for element `i`. The draw width follows the range, 32 bits up to 2^32 and 64 bits above, so the
dtype does not change the values, and `width` names it as the `u32` and `u64` fills of the C and
CUDA ports do. A rejected draw retries on the fallback stream `split(g)` of `sub(0x424c573332)`
(`0x424c573634` for 64 bits) from position 0, with `g` the index of the draw in the key's stream,
so a fill cut at any element boundary equals the whole fill. The retry loop runs only when a
draw was rejected. `maxval <= minval` gives `minval`. Bounded integers and uniforms are exact
across ports. Normals agree to the tolerance of Appendix A. Neither equals `jax.random.normal`
or `jax.random.randint`.

`tj.sub(key, purpose)` is the purpose child for a purpose of up to 64 bits. `jax.random.fold_in`
passes the implementation a 32-bit value, so it covers purposes below 2^32 only.

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
`(raw >> 8) * 2**-24`, so `jax.random.uniform` does not equal the specification's Float64 draws.
`tj.uniform(key, shape, dtype, position=0, *, minval, maxval)` is the drop-in with the
specification's mapping, including `float16` as `(raw >> 5) * 2**-11`. It reads from the stream
at `position` and returns the array only.
`tj.stream(key, position, n, dtype)` applies the spec's mappings for `float16`, `float32`
and `float64`, returns `bool` as single bits, returns the signed integer types by
reinterpreting the unsigned draw in two's complement, returns the unsigned types as the raw
aligned words, and returns `complex64` and `complex128` as alternating real and imaginary
`float32` or `float64` draws. The key
implementation `tj.impl` is the canonical `Tandem8x32-K32`.
`tj.impl_for(K)` gives the implementation of `Tandem8x32-K<K>` for a power of two from 1 to
65536, and `tj.key(seed, chunk_length=K)` makes a typed key of that variant. Children from
`split`, `fork` and `stream` follow the parent's variant, and `stream` takes `chunk_length`
for raw key words.

64-bit types need `jax.config.update("jax_enable_x64", True)`. Without it the 32-bit
multiplies are built from 16-bit halves, which also serves backends without 64-bit integers.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Install

```sh
pip install .
```

For development, `pixi install` creates an environment with the package installed editable,
and `pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file, with a drift check in CI), checks `normal` and `randint` against the C and CUDA fixtures in `tests/cross_derived.json` (written by `tools/convert_c_fixtures.py`, rejections included, fills from positions 0, 1 and 12345), checks that a bounded fill cut at any element equals the whole fill and that `int32` and `int64` agree for a small range, checks `split`, `fork` and `sub` against fixed values from the C reference (`tests/cross_port.json`, written by `tools/gen_split_fixture.c`), checks the K = 8 variant, `uniform` and the `bool` and complex stream dtypes against the dumps, compares positioned reads and
`jax.random.bits` with reference stream dumps in `tests/data`,
fork children at traced positions, split children for indices up to 2^64 - 1 against a direct evaluation of F,
and runs the key implementation under `jit` and `vmap`.

## Speed

NVIDIA A100 (one GPU of two, idle), CUDA 12 `jaxlib` 0.11.2 with driver 570, `jax_enable_x64`,
`pixi run bench`, jitted, minimum of seven runs after a warm-up, GiB/s of output.
`jax.random` uses the default threefry2x32 key.

| draw | log2 n | tandem_jax | threefry |
|---|---|---|---|
| uniform float32 | 24 | 151 | 167 |
| uniform float64 | 24 | 177 | 317 |
| normal float32 | 24 | 108 | 151 |
| normal float64 | 24 | 101 | 115 |
| randint int32 in [0, 1000) | 24 | 76 | 109 |
| randint int64 in [0, 1000) | 24 | 127 | 236 |
| uniform float32 | 27 | 272 | 348 |
| uniform float64 | 27 | 273 | 589 |
| normal float32 | 27 | 164 | 245 |
| normal float64 | 27 | 139 | 110 |
| randint int32 in [0, 1000) | 27 | 165 | 172 |
| randint int64 in [0, 1000) | 27 | 243 | 317 |

On a GPU backend the stream rows come from one Pallas kernel (`src/tandem_jax/_gpu.py`) that
keeps the chunk state in registers and writes each block once. Without it XLA writes the state of
every step to memory, and the same draws run two to four times slower. JAX marks the Pallas
Triton backend as deprecated, so a future JAX release may need the kernel ported.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
