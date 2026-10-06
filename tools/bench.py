"""Throughput of uniform, normal, exponential and bounded-integer draws, jitted: tandem_jax on the
XLA path and through its native extension, against jax.random with the default threefry2x32 key,
in GiB/s of output. On a GPU a cuRAND Philox4x32-10 column follows, by the same method.

Each cell runs in its own process: after heavy XLA work, such as the XLA path's float64 draws,
every large write in that process slows, XLA's own included, which would make the figures depend
on the order of the cells. Each function first warms up for half a second, so the clocks settle.
On a CPU a run is one call, best of five. On a GPU a run times ten calls issued back to back, as a
loop of draws would, best of seven: one call there is short enough that its dispatch would
dominate the time.

    python tools/bench.py [log2 sizes, default 24 27]
"""

import os
import subprocess
import sys
import time

# The parent only spawns the cells. Without preallocation it leaves them the GPU's memory.
if sys.argv[1:2] != ["--cell"]:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import tandem_jax as tj  # noqa: E402

GPU = jax.default_backend() == "gpu"
RUNS, CALLS, WARM = (7, 10, 0.5) if GPU else (5, 1, 0.5)
COLUMNS = ("xla", "native", "threefry") + (("curand",) if GPU else ())


def rows(n):
    tk, jk = tj.key(42), jax.random.key(0)
    f32, f64, i32, i64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
    return [
        ("uniform float32", 4, lambda: tj.uniform(tk, (n,), f32), lambda: jax.random.uniform(jk, (n,), f32)),
        ("uniform float64", 8, lambda: tj.uniform(tk, (n,), f64), lambda: jax.random.uniform(jk, (n,), f64)),
        ("normal float32", 4, lambda: tj.normal(tk, (n,), f32), lambda: jax.random.normal(jk, (n,), f32)),
        ("normal float64", 8, lambda: tj.normal(tk, (n,), f64), lambda: jax.random.normal(jk, (n,), f64)),
        ("exponential float32", 4, lambda: tj.exponential(tk, (n,), f32), lambda: jax.random.exponential(jk, (n,), f32)),
        ("exponential float64", 8, lambda: tj.exponential(tk, (n,), f64), lambda: jax.random.exponential(jk, (n,), f64)),
        ("randint int32 in [0, 1000)", 4, lambda: tj.randint(tk, (n,), 0, 1000, i32), lambda: jax.random.randint(jk, (n,), 0, 1000, i32)),
        ("randint int64 in [0, 1000)", 8, lambda: tj.randint(tk, (n,), 0, 1000, i64), lambda: jax.random.randint(jk, (n,), 0, 1000, i64)),
    ]


def best(draw, nbytes):
    fn = jax.jit(draw)
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


# The cuRAND Philox4x32-10 host call that stands for each row, and how many of its values one
# element takes. cuRAND has no exponential and no bounded or 64-bit integer output, so those rows
# take the nearest call: the uniform the exponential reads, or 32-bit words into the same bytes.
CURAND = [
    ("curandGenerateUniform", 1), ("curandGenerateUniformDouble", 1),
    ("curandGenerateNormal", 1), ("curandGenerateNormalDouble", 1),
    ("curandGenerateUniform", 1), ("curandGenerateUniformDouble", 1),
    ("curandGenerate", 1), ("curandGenerate", 2),
]


def curand_best(i, n, nbytes):
    """cuRAND into a JAX device buffer by the method of best(), through ctypes in XLA's context."""
    import ctypes
    import os

    import nvidia.cuda_runtime
    import nvidia.curand

    cr = ctypes.CDLL(os.path.join(nvidia.curand.__path__[0], "lib", "libcurand.so.10"))
    rt = ctypes.CDLL(os.path.join(nvidia.cuda_runtime.__path__[0], "lib", "libcudart.so.12"))
    buf = jnp.zeros((nbytes // 4,), jnp.uint32)
    jax.block_until_ready(buf)
    out = ctypes.c_void_p(buf.unsafe_buffer_pointer())
    gen = ctypes.c_void_p()
    assert cr.curandCreateGenerator(ctypes.byref(gen), 161) == 0  # CURAND_RNG_PSEUDO_PHILOX4_32_10
    assert cr.curandSetPseudoRandomGeneratorSeed(gen, ctypes.c_ulonglong(42)) == 0
    call, per = CURAND[i]
    f, m = getattr(cr, call), ctypes.c_size_t(n * per)
    args = {"curandGenerateNormal": (ctypes.c_float(0), ctypes.c_float(1)),
            "curandGenerateNormalDouble": (ctypes.c_double(0), ctypes.c_double(1))}.get(call, ())

    def run(calls):
        for _ in range(calls):
            assert f(gen, out, m, *args) == 0
        assert rt.cudaDeviceSynchronize() == 0

    end = time.perf_counter() + WARM
    while time.perf_counter() < end:
        run(1)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run(CALLS)
        ts.append((time.perf_counter() - t0) / CALLS)
    cr.curandDestroyGenerator(gen)
    return nbytes / min(ts) / 2**30


def cell(lg, i, column):
    """GiB/s of row `i` at 2^lg elements in `column`, in this process."""
    if column == "xla":
        tj._ffi.tandem_jax_cpu = tj._ffi.tandem_jax_cuda = None
    name, size, ours, theirs = rows(2**lg)[i]
    if column == "curand":
        return curand_best(i, 2**lg, 2**lg * size)
    return best(theirs if column == "threefry" else ours, 2**lg * size)


if sys.argv[1:2] == ["--cell"]:
    print(cell(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
    sys.exit()

native = tj._ffi.tandem_jax_cpu is not None or tj._ffi.tandem_jax_cuda is not None
print("backend", jax.default_backend(), jax.devices()[0])
print("extensions", [m.__name__ for m in (tj._ffi.tandem_jax_cpu, tj._ffi.tandem_jax_cuda) if m])
print(f"{'draw':26s} {'log2 n':>6s} {'XLA path':>9s} {'native':>8s} {'threefry':>9s}"
      + (f" {'cuRAND':>8s}" if GPU else "") + "  GiB/s of output")
for lg in [int(a) for a in sys.argv[1:]] or [24, 27]:
    for i, (name, *_) in enumerate(rows(2**lg)):
        out = []
        for column in COLUMNS:
            if column == "native" and not native:
                out.append(float("nan"))
                continue
            run = subprocess.run([sys.executable, __file__, "--cell", str(lg), str(i), column],
                                 capture_output=True, text=True, check=True)
            out.append(float(run.stdout.split()[-1]))
        print(f"{name:26s} {lg:6d} {out[0]:9.2f} {out[1]:8.2f} {out[2]:9.2f}"
              + (f" {out[3]:8.2f} {CURAND[i][0]}" if GPU else ""), flush=True)
