# Speed

`pixi run bench 22` produces the CPU figures and `python tools/bench.py` the GPU figures.
`jax.random` uses the default threefry2x32 key in both.

## CPU

Apple M4 Pro, XLA CPU backend, `jax_enable_x64`, `pixi run bench 22`, jitted, minimum of five
runs after a warm-up, GiB/s of output, 2^22 elements. The `cpu/` column has the extension
installed, which splits each fill over XLA's thread pool.

| draw | tandem_jax | tandem_jax with `cpu/` | threefry |
|---|---|---|---|
| uniform float32 | 5.2 | 94 | 3.0 |
| uniform float64 | 4.8 | 102 | 5.5 |
| normal float32 | 3.6 | 23 | 2.6 |
| normal float64 | 3.9 | 26 | 3.0 |
| randint int32 in [0, 1000) | 3.0 | 47 | 1.7 |
| randint int64 in [0, 1000) | 5.3 | 88 | 3.0 |

With `cpu/` built to fill in one part, the uniforms ran at 16.0 (`float32`) and 16.6 (`float64`)
GiB/s against 16.6 and 16.1 for tandem-c's own fills in the same window, and the `float32` normals
at 5.3 against 5.4. The thread pool gives the rest of the `cpu/` column, and moves with the load
on the other cores: `float32` normals ranged from 21 to 35 GiB/s over two runs. tandem-c's own
ziggurat fill writes 7.5 GiB/s of `float64` normals on one core. The XLA path's ziggurat writes
3.7 to 3.9 GiB/s over three runs, against 1.1 for its gather of the misses by `nonzero` and 2.3
to 2.6 for threefry in the same window, and 3.3 for its Box-Muller before. The stream takes about
two thirds of that time and the misses the rest.

## GPU

NVIDIA A100 (one GPU of two, idle), CUDA 12 `jaxlib` 0.11.2 with driver 570, `jax_enable_x64`,
the CUDA kernels built for `sm_80`, `python tools/bench.py`. Each function is jitted and warmed
up for half a second, a run times ten calls issued back to back, and the table gives the best of
seven runs in GiB/s of output.

| draw | log2 n | tandem_jax | threefry |
|---|---|---|---|
| uniform float32 | 24 | 730 | 523 |
| uniform float64 | 24 | 1125 | 973 |
| normal float32 | 24 | 841 | 418 |
| normal float64 | 24 | 768 | 226 |
| randint int32 in [0, 1000) | 24 | 919 | 298 |
| randint int64 in [0, 1000) | 24 | 1041 | 537 |
| uniform float32 | 27 | 1304 | 656 |
| uniform float64 | 27 | 1345 | 991 |
| normal float32 | 27 | 1193 | 427 |
| normal float64 | 27 | 1036 | 220 |
| randint int32 in [0, 1000) | 27 | 1216 | 329 |
| randint int64 in [0, 1000) | 27 | 1277 | 567 |

The kernels alone, as the JAX profiler times them at 2^27, write 1280 to 1390 GiB/s, the rates of
tandem-cuda's own benchmark. At 2^24 the launch and dispatch cost of each call is a larger share,
and uniform `float32` there ranged from 717 to 984 GiB/s over three runs. The other cells moved by a
few percent. The `float64` normals are the ziggurat's two kernels, a table pass and a pass over
the misses, as in tandem-cuda, which writes 788 to 825 GiB/s at 2^24. The kernels take the same
time here. The miss list comes from XLA's scratch allocator. A stream-ordered allocation in each
call held 2^24 at 468 GiB/s, because XLA's event syncs let the pool release the list between calls.
Without the extension a GPU runs the XLA path, at 93 to 158 GiB/s for these draws at 2^27. Its
`float64` normals write 70 GiB/s at 2^27 and 45 at 2^24, against 45 and 41 before the XLA path
found its misses by a binary search. Those four figures come from tandem-cuda's steady-state
method: each fill runs for two seconds, then the median of 21 calls, and two runs agreed within 5 %.
