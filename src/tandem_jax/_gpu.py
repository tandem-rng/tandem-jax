"""Stream rows on GPU as one Pallas kernel.

Under XLA the K steps of a chunk are a loop that writes the state and the output of every
step to memory, or a long unrolled graph that XLA splits into many kernels with transposes
between them. The kernel keeps the eight state words of a chunk in registers, applies T K
times and writes each block once, in stream order.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from . import _core

U32 = jnp.uint32
GROUPS_PER_PROGRAM = 16  # 8 lanes per group, so 128 chunks per program


def available():
    return jax.default_backend() in ("gpu", "cuda")


def rows(key, c0, ngroups, K):
    """The words of `ngroups` groups from chunk `c0` (a uint64 or uint32 scalar), as
    `_core.rows` returns them, plus trailing words of the last program's padding."""
    G = min(GROUPS_PER_PROGRAM, 1 << (ngroups.bit_length() - 1))
    programs = -(-ngroups // G)
    L = 8 * G
    block = G * K * 32

    def kernel(prm_ref, out_ref):
        k = tuple(prm_ref[i] for i in range(4))
        q = lax.iota(U32, L)
        lo = prm_ref[4] + pl.program_id(0).astype(U32) * U32(L) + q
        hi = prm_ref[5] + (lo < prm_ref[4]).astype(U32)
        o, h = _core.F_keyed(k, (lo, hi), _core.DOMAIN_STREAM, _core.AUX_STREAM)
        base = ((q >> 3) * U32(K * 32) + (q & 7) * U32(4)).astype(jnp.int32)

        def step(j, s):
            o, h = _core.T(*s)
            for w in range(4):
                out_ref[base + j * 32 + w] = o[w]
            return o, h

        lax.fori_loop(0, K, step, (o, h))

    c0 = jnp.asarray(c0, jnp.uint64 if jax.config.jax_enable_x64 else U32)
    lo, hi = (c0.astype(U32), (c0 >> jnp.uint64(32)).astype(U32)) if c0.dtype == jnp.uint64 else (c0, U32(0))
    prm = jnp.stack([*(jnp.asarray(w, U32) for w in key), lo, hi])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((programs * block,), U32),
        grid=(programs,),
        in_specs=[pl.BlockSpec((6,), lambda i: (0,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(prm)
