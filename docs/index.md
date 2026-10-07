# tandem-jax

JAX key implementation of Tandem8x32. It is pure Python over `jax.numpy` and `jax.lax`, with
optional fills from [tandem-c](https://github.com/tandem-rng/tandem-c) on the CPU and kernels
from [tandem-cuda](https://github.com/tandem-rng/tandem-cuda) on NVIDIA GPUs. It produces the
stream of the [specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for
bit.

- [API](api.md): typed keys, `jax.random` on a Tandem key, the `tj` functions and parallel use.
- [Design](design.md): the CPU and CUDA fills, bounded integers and normals.
- [Tests](tests.md): what the suite checks, where the fixtures come from, and what CI runs.
- [Speed](speed.md): the M4 Pro CPU and A100 figures against threefry.

## Install

```sh
pip install .
```

Needs Python 3.11 and `jax>=0.10`. For development, `pixi install` creates an environment with
the package installed editable, and `pixi run test` runs the tests.

### CPU fills

The package in `cpu/` builds the tandem-c fills as an XLA FFI handler for the CPU. It needs a C
and C++17 compiler and CMake 3.18 or later. Build it against the installed `jaxlib`, since a
handler built on newer FFI headers fails to register and stops the CPU backend from starting:

```sh
pip install scikit-build-core
pip install --no-build-isolation ./cpu
```

`cpu/tandem` holds `tandem.c`, `tandem.h` and `tandem_normal_tables.h` of tandem-c commit
`1c75956`, with the ziggurat `float64` normals. Its source is compiled with `-ffp-contract=off`,
as tandem-c's Makefile does, which keeps the normals bit exact on every compiler.

### CUDA kernels

The package in `cuda/` builds the tandem-cuda fills as an XLA FFI handler. It needs the CUDA
toolkit (`nvcc`, of the major version of your `jax[cuda]` install and no newer than the driver
supports) and CMake 3.24 or later. Build it against the installed `jaxlib`:

```sh
pip install "jax[cuda12]" scikit-build-core
pip install --no-build-isolation ./cuda
```

The build targets every major GPU architecture. Add
`-C cmake.define.CMAKE_CUDA_ARCHITECTURES=80` (an A100, for example) to build for one only.

The headers in `cuda/include` are those of tandem-cuda commit `2693c63`. The `float64` normal
kernels in `cuda/tandem_ffi.cu` follow those of commit `0ff5f18`: the octet table pass of `3aac1fb`
changes speed only, and XLA's scratch allocator already keeps the miss lists.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
