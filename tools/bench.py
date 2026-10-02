"""Throughput of 2**24 Float64 draws on the XLA CPU backend: tandem_jax.stream, then
jax.random.uniform with the default threefry and with rbg keys."""

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import tandem_jax as tj  # noqa: E402

N = 2**24
BYTES = N * 8


def best(fn, runs=7):
    fn().block_until_ready()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return BYTES / min(ts) / 2**30


k = tj.key(42)
rows = [
    ("tandem_jax.stream(key, 0, n, float64)", best(jax.jit(lambda: tj.stream(k, 0, N, jnp.float64)[0]))),
    ("jax.random.uniform(tandem key, float64)", best(jax.jit(lambda: jax.random.uniform(k, (N,), jnp.float64)))),
    ("jax.random.uniform(threefry key, float64)", best(jax.jit(lambda: jax.random.uniform(jax.random.key(0), (N,), jnp.float64)))),
    ("jax.random.uniform(rbg key, float64)", best(jax.jit(lambda: jax.random.uniform(jax.random.key(0, impl="rbg"), (N,), jnp.float64)))),
]
for name, gibs in rows:
    print(f"{name:44s} {gibs:6.2f} GiB/s")
print("backend", jax.default_backend(), "load", os.getloadavg())
