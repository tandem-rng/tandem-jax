"""Throughput of uniform, normal and bounded-integer draws, jitted, best of seven after a
warm-up: tandem_jax against jax.random with the default threefry2x32 key.

    python tools/bench.py [log2 sizes, default 24 27]
"""

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import tandem_jax as tj  # noqa: E402

sizes = [int(a) for a in sys.argv[1:]] or [24, 27]


def best(fn, nbytes, runs=5):
    fn().block_until_ready()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return nbytes / min(ts) / 2**30


tk, jk = tj.key(42), jax.random.key(0)
print("backend", jax.default_backend(), jax.devices()[0], "load", os.getloadavg())
print(f"{'draw':26s} {'log2 n':>6s} {'tandem':>8s} {'threefry':>9s}  GiB/s of output")
for lg in sizes:
    n = 2**lg
    rows = [
        ("uniform float32", 4, lambda: tj.uniform(tk, (n,), jnp.float32), lambda: jax.random.uniform(jk, (n,), jnp.float32)),
        ("uniform float64", 8, lambda: tj.uniform(tk, (n,), jnp.float64), lambda: jax.random.uniform(jk, (n,), jnp.float64)),
        ("normal float32", 4, lambda: tj.normal(tk, (n,), jnp.float32), lambda: jax.random.normal(jk, (n,), jnp.float32)),
        ("normal float64", 8, lambda: tj.normal(tk, (n,), jnp.float64), lambda: jax.random.normal(jk, (n,), jnp.float64)),
        ("randint int32 in [0, 1000)", 4, lambda: tj.randint(tk, (n,), 0, 1000, jnp.int32), lambda: jax.random.randint(jk, (n,), 0, 1000, jnp.int32)),
        ("randint int64 in [0, 1000)", 8, lambda: tj.randint(tk, (n,), 0, 1000, jnp.int64), lambda: jax.random.randint(jk, (n,), 0, 1000, jnp.int64)),
    ]
    for name, size, ours, theirs in rows:
        print(f"{name:26s} {lg:6d} {best(jax.jit(ours), n * size):8.2f} {best(jax.jit(theirs), n * size):9.2f}", flush=True)
