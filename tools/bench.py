"""Throughput of uniform, normal, exponential and bounded-integer draws, jitted: tandem_jax on the
XLA path and through its native extension, against jax.random with the default threefry2x32 key,
in GiB/s of output.

Each function first warms up for half a second, so the clocks settle. On a CPU a run is one call,
best of five. On a GPU a run times ten calls issued back to back, as a loop of draws would, best of
seven: one call there is short enough that its dispatch would dominate the time.

    python tools/bench.py [log2 sizes, default 24 27]
"""

import contextlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import tandem_jax as tj  # noqa: E402

sizes = [int(a) for a in sys.argv[1:]] or [24, 27]
GPU = jax.default_backend() == "gpu"
RUNS, CALLS, WARM = (7, 10, 0.5) if GPU else (5, 1, 0.5)
NATIVE = (tj._ffi.tandem_jax_cpu, tj._ffi.tandem_jax_cuda)


@contextlib.contextmanager
def xla_path():
    """Trace with no extension loaded, so tandem_jax lowers to its XLA path. The draws jit
    internally, so the caches are cleared on both sides to keep either lowering from leaking."""
    jax.clear_caches()
    tj._ffi.tandem_jax_cpu = tj._ffi.tandem_jax_cuda = None
    try:
        yield
    finally:
        tj._ffi.tandem_jax_cpu, tj._ffi.tandem_jax_cuda = NATIVE
        jax.clear_caches()


def best(draw, nbytes):
    # A fresh function per call, so jit traces again rather than reusing a cached lowering.
    fn = jax.jit(lambda: draw())
    jax.block_until_ready(fn())
    end = time.perf_counter() + WARM
    while time.perf_counter() < end:
        jax.block_until_ready(fn())
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for _ in range(CALLS - 1):
            fn()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t0) / CALLS)
    return nbytes / min(ts) / 2**30


tk, jk = tj.key(42), jax.random.key(0)
native = any(NATIVE)
print("backend", jax.default_backend(), jax.devices()[0])
print("extensions", [m.__name__ for m in NATIVE if m])
print(f"{'draw':26s} {'log2 n':>6s} {'XLA path':>9s} {'native':>8s} {'threefry':>9s}  GiB/s of output")
for lg in sizes:
    n = 2**lg
    f32, f64, i32, i64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
    rows = [
        ("uniform float32", 4, lambda: tj.uniform(tk, (n,), f32), lambda: jax.random.uniform(jk, (n,), f32)),
        ("uniform float64", 8, lambda: tj.uniform(tk, (n,), f64), lambda: jax.random.uniform(jk, (n,), f64)),
        ("normal float32", 4, lambda: tj.normal(tk, (n,), f32), lambda: jax.random.normal(jk, (n,), f32)),
        ("normal float64", 8, lambda: tj.normal(tk, (n,), f64), lambda: jax.random.normal(jk, (n,), f64)),
        ("exponential float32", 4, lambda: tj.exponential(tk, (n,), f32), lambda: jax.random.exponential(jk, (n,), f32)),
        ("exponential float64", 8, lambda: tj.exponential(tk, (n,), f64), lambda: jax.random.exponential(jk, (n,), f64)),
        ("randint int32 in [0, 1000)", 4, lambda: tj.randint(tk, (n,), 0, 1000, i32), lambda: jax.random.randint(jk, (n,), 0, 1000, i32)),
        ("randint int64 in [0, 1000)", 8, lambda: tj.randint(tk, (n,), 0, 1000, i64), lambda: jax.random.randint(jk, (n,), 0, 1000, i64)),
    ]
    for name, size, ours, theirs in rows:
        with xla_path():
            xla = best(ours, n * size)
        ext = f"{best(ours, n * size):8.2f}" if native else f"{'-':>8s}"
        print(f"{name:26s} {lg:6d} {xla:9.2f} {ext} {best(theirs, n * size):9.2f}", flush=True)
