<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .jax"></p>

# tandem-jax

[![CI](https://github.com/tandem-rng/tandem-jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-jax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-jax/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

JAX key implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. It is pure Python over `jax.numpy` and `jax.lax`, with optional
fills from [tandem-c](https://github.com/tandem-rng/tandem-c) on the CPU and
kernels from [tandem-cuda](https://github.com/tandem-rng/tandem-cuda) on NVIDIA GPUs. It produces the
stream the specification defines, bit for bit, fast on CPU and GPU.

## Install

```sh
pip install .
```

Needs Python 3.11 and `jax>=0.10`. For development, `pixi install`, then `pixi run test`.
64-bit types need `jax.config.update("jax_enable_x64", True)`.

For the CPU, build the XLA FFI extension in `cpu/` against the installed `jaxlib`. It needs a C
and C++17 compiler and CMake 3.18 or later.

```sh
pip install scikit-build-core
pip install --no-build-isolation ./cpu
```

`cpu/tandem` is tandem-c commit `121db59`.

For NVIDIA GPUs, build the XLA FFI extension in `cuda/`. It needs `nvcc` matching your
`jax[cuda]` major version, and CMake 3.24 or later.

```sh
pip install "jax[cuda12]" scikit-build-core
pip install --no-build-isolation ./cuda
# one architecture only: -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80
```

The headers in `cuda/include` are tandem-cuda commit `76eddae`.

## Use

```python
import jax, jax.numpy as jnp
import tandem_jax as tj

key = tj.key(42)                                   # the spec's generator for seed 42
k1, k2 = jax.random.split(key)                     # the spec's split by index
step_key = jax.random.fold_in(key, 7)              # the spec's purpose 7
z = jax.random.normal(k1, (1000,))                 # any jax.random function

u = tj.uniform(key, (1000,), jnp.float32)          # the spec's mapping
z = tj.normal(key, (1000,), jnp.float32)           # Box-Muller normals
r = tj.randint(key, (1000,), 0, 6, jnp.int32)      # Lemire bounded integers
x, pos = tj.stream(key, 0, 2**20, jnp.float64)     # Float64 draws, and the position after
kids, pos = tj.fork(key, pos, 4)                   # the spec's fork, typed keys, jit-safe
```

## What it provides

- `tj.key(seed, chunk_length=K)`: a typed key. `K = 32` by default, any power of two to 65536.
  `tj.impl` is `Tandem8x32-K32` and `tj.impl_for(K)` gives the variant.
- `jax.random.split`, `fold_in`, `bits`: the spec's `split(0..n-1)`, `sub(u)` for `u < 2^32`,
  and stream words from position 0.
- `tj.split(key, index)`, `tj.sub(key, purpose)`: any unsigned 64-bit index or purpose.
  A count above 2^32 needs `jax_enable_x64`.
- `tj.fork(key, position, n)`, `tj.fork_words`: fork children as typed keys or `(n, 4)` words.
- `tj.stream(key, position, n, dtype)`: draws for `bool`, signed and unsigned integers,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, and the position after.
- `tj.uniform(key, shape, dtype, position=0)`: the spec's mapping, unlike `jax.random.uniform`.
- `tj.randint`, `tj.stream_randint`: bounded integers, same values in every port.
- `tj.normal`, `tj.stream_normal`: the ziggurat for `float64`, bit exact in every port, and
  Box-Muller for `float32`, same to the tolerance of Appendix A.
- Neither `tj.normal` nor `tj.randint` equals the `jax.random` function of that name.
- CPU: with `cpu/` installed, `stream`, `uniform`, `normal`, `randint`, and `jax.random` on a
  Tandem key run on the tandem-c fills over XLA's thread pool, under `jit` and `vmap`.
- CUDA: the same functions run on the kernels on a GPU, under `jit` and `vmap`.
- Parallel use: ranks, threads, or devices that start at the position of their first element, or
  draw from `split(task)`, reproduce a serial run. See
  [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).

## Tests

`pixi run test` runs `tests/test_tandem.py`. It checks:

- Every specification vector (`tests/vectors.json`) and the stream dumps in `tests/data`.
- `normal` and `randint` against C and CUDA fixtures in `tests/cross_derived.json`, written by
  `tools/convert_c_fixtures.py`.
- `split`, `fork`, and `sub` against `tests/cross_port.json`, written by
  `tools/gen_split_fixture.c`.
- Fills cut at any element equal the whole fill, under `jit` and `vmap`.
- Normal moments to fourth order and a KS test on 10^7 draws, and chi-square uniformity of
  bounded integers at a range that rejects a quarter of the draws.
- With the CPU extension, the fills equal the XLA path, and normals equal tandem-c bit for bit.
- With the CUDA extension, the kernels equal the XLA path on the CPU device.

## Speed

Apple M4 Pro, XLA CPU backend, `jax_enable_x64`, `pixi run bench 22`, jitted, minimum of five
runs, GiB/s of output, 2^22 elements. `jax.random` uses the default threefry2x32 key. The `cpu/`
column has the extension installed, which splits each fill over XLA's thread pool.

| draw | tandem_jax | tandem_jax with `cpu/` | threefry |
|---|---|---|---|
| uniform float32 | 5.2 | 94 | 3.0 |
| uniform float64 | 4.8 | 102 | 5.5 |
| normal float32 | 3.6 | 23 | 2.6 |
| normal float64 | 1.2 | 26 | 3.0 |
| randint int32 in [0, 1000) | 3.0 | 47 | 1.7 |
| randint int64 in [0, 1000) | 5.3 | 88 | 3.0 |

NVIDIA A100, CUDA 12, `jax_enable_x64`, kernels built for `sm_80`, `python tools/bench.py`,
best of seven runs, GiB/s of output. `jax.random` uses the default threefry2x32 key.

| draw | log2 n | tandem_jax | threefry |
|---|---|---|---|
| uniform float32 | 24 | 730 | 523 |
| uniform float64 | 24 | 1125 | 973 |
| normal float32 | 24 | 841 | 418 |
| normal float64 | 24 | 468 | 226 |
| randint int32 in [0, 1000) | 24 | 919 | 298 |
| randint int64 in [0, 1000) | 24 | 1041 | 537 |
| uniform float32 | 27 | 1304 | 656 |
| uniform float64 | 27 | 1345 | 991 |
| normal float32 | 27 | 1193 | 427 |
| normal float64 | 27 | 953 | 220 |
| randint int32 in [0, 1000) | 27 | 1216 | 329 |
| randint int64 in [0, 1000) | 27 | 1277 | 567 |

See [API](docs/api.md), [design](docs/design.md), [tests](docs/tests.md) and
[speed](docs/speed.md) for the longer notes.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

[Documentation](https://tandem-rng.github.io/tandem-jax/) · [Apache 2.0 license](LICENSE)
