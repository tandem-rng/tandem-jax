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

Needs Python 3.11 and `jax>=0.10`. `cpu/tandem` is tandem-c commit `121db59`. The headers in
`cuda/include` are tandem-cuda commit `bab9870`.

```sh
pip install .
pip install --no-build-isolation ./cpu     # optional tandem-c fills on the CPU
pip install --no-build-isolation ./cuda    # optional CUDA kernels
```

```python
import jax, jax.numpy as jnp
import tandem_jax as tj

key = tj.key(42)                                   # the spec's generator for seed 42
k1, k2 = jax.random.split(key)                     # the spec's split by index
z = jax.random.normal(k1, (1000,))                 # any jax.random function
x, pos = tj.stream(key, 0, 2**20, jnp.float64)     # Float64 draws, and the position after
r = tj.randint(key, (1000,), 0, 6, jnp.int32)      # Lemire bounded integers
e = tj.exponential(key, (1000,))                   # exponentials -ln(1 - u), Appendix A
kids, pos = tj.fork(key, pos, 4)                   # the spec's fork, typed keys, jit-safe
```

See the [documentation](https://tandem-rng.github.io/tandem-jax/) for the build requirements of
`cpu/` and `cuda/`, the API, tests and speed.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-jax/) · [Apache 2.0 license](LICENSE)
