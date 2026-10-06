# Speed

`pixi run bench 22` produces the CPU figures and `python tools/bench.py` the GPU figures.
`jax.random` uses the default threefry2x32 key in both.

## CPU

Apple M4 Pro, XLA CPU backend, `jax_enable_x64`, `pixi run bench 22`, jitted, minimum of five
runs after a half-second warm-up, GiB/s of output, 2^22 elements, median of three passes in one
session. The `cpu/` column has the extension installed, which splits each fill over XLA's thread
pool. Each cell runs in its own process. After heavy XLA work, every large write in a process
slows, XLA's own `jnp.full` from 200 to about 40 GiB/s, and in one process the `cpu/` `float64`
uniforms read 32.

| draw | tandem_jax | tandem_jax with `cpu/` | threefry |
|---|---|---|---|
| uniform float32 | 6.0 | 93 | 3.6 |
| uniform float64 | 5.4 | 102 | 6.0 |
| normal float32 | 3.6 | 32 | 2.8 |
| normal float64 | 3.9 | 58 | 2.8 |
| exponential float32 | 1.7 | 48 | 2.7 |
| exponential float64 | 1.1 | 46 | 3.1 |
| randint int32 in [0, 1000) | 3.2 | 46 | 1.7 |
| randint int64 in [0, 1000) | 6.4 | 87 | 3.0 |

The XLA path costs the same per 32-bit word at every width: its `float64` uniforms run at the
rate of 2n `uint32` words. Threefry hashes once per element and returns 64 bits for `float64`, so
it leads there and Tandem leads at `float32`. Without the extension the exponentials trail
threefry too.

With `cpu/` built to fill in one part, the uniforms ran at 16.0 (`float32`) and 16.6 (`float64`)
GiB/s against 16.6 and 16.1 for tandem-c's own fills in the same window, and the `float32` normals
at 5.3 against 5.4. The thread pool gives the rest of the `cpu/` column, and moves with the load
on the other cores: `float32` normals ranged from 21 to 35 GiB/s over two runs. tandem-c's own
ziggurat fill writes 7.5 GiB/s of `float64` normals on one core. The XLA path's ziggurat writes
3.7 to 3.9 GiB/s over three runs, against 1.1 for its gather of the misses by `nonzero` and 2.3
to 2.6 for threefry in the same window, and 3.3 for its Box-Muller before. The stream takes about
two thirds of that time and the misses the rest.

## GPU

NVIDIA A100 (GPU 0 of two, idle), CUDA 12 `jaxlib` 0.11.2 with driver 570, `jax_enable_x64`,
the CUDA kernels built for `sm_80`, `python tools/bench.py`, one session. Each cell runs in its
own process. Each function is jitted and warmed up for half a second, a run times ten calls
issued back to back, and the table gives the best of seven runs in GiB/s of output. The XLA
path column is tandem_jax without the extension. The third-party columns come from the same
session, by the same method: cuRAND Philox4x32-10 of cuRAND 10.3.10 through its host API on a JAX
buffer, then `jax.random` with threefry. cuRAND has no exponentials and no bounded or 64-bit
integer output, so those rows give the nearest call, marked "nearest": the uniform the
exponential reads, or `curandGenerate` into the same bytes.

| draw | log2 n | tandem_jax | XLA path | cuRAND Philox4x32-10 | cuRAND call | threefry |
|---|---|---|---|---|---|---|
| uniform float32 | 24 | 745 | 88 | 1144 | `curandGenerateUniform` | 516 |
| uniform float64 | 24 | 1054 | 122 | 787 | `curandGenerateUniformDouble` | 863 |
| normal float32 | 24 | 660 | 76 | 863 | `curandGenerateNormal` | 422 |
| normal float64 | 24 | 765 | 50 | 568 | `curandGenerateNormalDouble` | 227 |
| exponential float32 | 24 | 639 | 55 | 1143 | `curandGenerateUniform`, nearest | 510 |
| exponential float64 | 24 | 861 | 63 | 787 | `curandGenerateUniformDouble`, nearest | 454 |
| randint int32 in [0, 1000) | 24 | 641 | 55 | 1150 | `curandGenerate`, nearest | 300 |
| randint int64 in [0, 1000) | 24 | 1000 | 102 | 1240 | `curandGenerate`, nearest | 543 |
| uniform float32 | 27 | 1302 | 110 | 1268 | `curandGenerateUniform` | 657 |
| uniform float64 | 27 | 1348 | 109 | 803 | `curandGenerateUniformDouble` | 1022 |
| normal float32 | 27 | 1203 | 93 | 887 | `curandGenerateNormal` | 431 |
| normal float64 | 27 | 1031 | 72 | 587 | `curandGenerateNormalDouble` | 223 |
| exponential float32 | 27 | 1099 | 65 | 1269 | `curandGenerateUniform`, nearest | 535 |
| exponential float64 | 27 | 893 | 59 | 803 | `curandGenerateUniformDouble`, nearest | 445 |
| randint int32 in [0, 1000) | 27 | 1216 | 93 | 1293 | `curandGenerate`, nearest | 331 |
| randint int64 in [0, 1000) | 27 | 1286 | 156 | 1302 | `curandGenerate`, nearest | 570 |

The kernels alone, as the JAX profiler times them at 2^27, write 1280 to 1390 GiB/s, the rates of
tandem-cuda's own benchmark. At 2^24 the launch and dispatch cost of each call is a larger share:
a second run of the session moved those tandem_jax cells by up to 7 %, the 2^27 cells by under
2 %. cuRAND leads at 2^24 because its host calls skip JAX's dispatch, and on the exponential and
randint rows because its nearest call does less work: uniforms without the logarithm, 32-bit words
without bounding. The extension vendors tandem-cuda 2693c63. Against bab9870 in the same session,
its folded exponential took the 2^27 exponentials from 957 to 1099 GiB/s in `float32` and from 868
to 893 in `float64`, and the `float32` normals moved from 1190 to 1203.

The `float64` normals are the ziggurat's two kernels, a table pass and a pass over the misses, as in tandem-cuda, which writes 788 to 825 GiB/s at 2^24.
The kernels take the same time here. The miss list comes from XLA's scratch allocator. A
stream-ordered allocation in each call held 2^24 at 468 GiB/s, because XLA's event syncs let the
pool release the list between calls. The XLA path's `float64` normals wrote 45 and 41 GiB/s at
2^27 and 2^24 before it found its misses by a binary search.
