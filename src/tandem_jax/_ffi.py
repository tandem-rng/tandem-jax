"""Fills by native code as XLA FFI calls: the tandem-cuda kernels on CUDA devices when the
`tandem_jax_cuda` extension is installed, the tandem-c fills on the CPU when `tandem_jax_cpu` is.
Everywhere else the XLA path runs."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

U32 = jnp.uint32
TILE_STEPS = 8
"""The CUDA fill kernels step a chunk this many blocks at a time, so they need K >= 8."""

try:
    import tandem_jax_cuda
except ImportError:
    tandem_jax_cuda = None
else:
    jax.ffi.register_ffi_target("tandem_fill", tandem_jax_cuda.fill_handler(), platform="CUDA")

try:
    import tandem_jax_cpu
except ImportError:
    tandem_jax_cpu = None
else:
    jax.ffi.register_ffi_target("tandem_fill", tandem_jax_cpu.fill_handler(), platform="cpu")


def _words(x):
    x = jnp.asarray(x)
    if x.dtype.itemsize == 8:
        x = x.astype(jnp.uint64)
        return [x.astype(U32), (x >> jnp.uint64(32)).astype(U32)]
    return [x.astype(U32), jnp.zeros((), U32)]


def fill(key, position, n, dtype, chunk, kind, bound=0, low=0, width=0):
    """`n` elements of `dtype` at stream bit `position` from the fill `kind`: "stream" for the
    spec's draws, "below" for bounded integers `low + [0, bound)` of draw width `width` (0 takes it
    from the bound), "normal" for normals, "exponential" for exponentials."""
    prm = jnp.concatenate([jnp.asarray(key, U32), jnp.stack(_words(position) + _words(bound) + _words(low))])
    call = jax.ffi.ffi_call("tandem_fill", jax.ShapeDtypeStruct((n,), dtype), vmap_method="broadcast_all")
    return call(prm, n=np.int64(n), chunk=np.int64(chunk), kind=kind, width=np.int32(width))


def native(cuda, cpu, call, default):
    """`call()` when lowered for a CUDA device with its extension loaded and `cuda` true, or for
    the CPU with its extension loaded and `cpu` true, else `default()`. The choice is made at
    lowering, so the compiled code holds one branch only."""
    branches = {}
    if cuda and tandem_jax_cuda is not None:
        branches["cuda"] = call
    if cpu and tandem_jax_cpu is not None:
        branches["cpu"] = call
    if not branches:
        return default()
    return lax.platform_dependent(**branches, default=default)
