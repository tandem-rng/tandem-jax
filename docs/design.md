# Design

## Fills

### XLA path

Without `jax_enable_x64` the 32-bit multiplies are built from 16-bit halves, which also serves
backends without 64-bit integers.

### CPU fills

The package in `cpu/` builds the tandem-c fills as an XLA FFI handler for the CPU, under the same
target name and operands as the CUDA one. `tandem_jax` loads the extension on import when it is
installed. On the CPU, `stream` for every dtype but `bool`, `uniform`, `normal`, `randint` with
scalar bounds, their `stream_*` forms and `jax.random.bits` then run on the tandem-c fills, at
any chunk length. Fills of 256 KiB or more split into parts at stream row boundaries, one per
thread of XLA's intra-op pool, and the call completes when the last part does. `jit` and `vmap`
work: a batch writes its rows back to back. Stream draws, uniforms, bounded integers and
`float64` normals equal the XLA path bit for bit. `float32` normals equal tandem-c bit for bit
and the XLA path within the tolerance of Appendix A, since XLA's `log`, `sin` and `cos` are not
tandem-c's polynomials.

### CUDA kernels

`tandem_jax` loads the extension on import when it is installed. On a CUDA device, `stream`,
`uniform`, `normal`, `randint`, their `stream_*` forms and the `jax.random` functions on a Tandem
key then run on the kernels. `lax.platform_dependent` makes the choice when XLA lowers the code,
so a CPU device in the same program keeps the XLA path. The kernels write 32- and 64-bit draws
directly. Narrower dtypes come from the 32-bit word fill and one XLA pass. They need a chunk
length of 8 or more, and `randint` with per-element bounds keeps the XLA path. `jit` and `vmap`
work: a batch of keys or positions runs as one kernel launch. The values are the XLA path's: bit
for bit for the stream, uniforms, bounded integers and `float64` normals, and within the
tolerance of Appendix A for `float32` normals, which use the fast `__sincosf` of tandem-cuda,
within 16 ulps + 1e-6. The `float64` normals take the ziggurat in one kernel that continues each
miss where it finds it, as tandem-cuda's short fills do.

## Bounded integers

Bounded integers use Lemire's method on draw `i` for element `i`. The draw width follows the
range, 32 bits up to 2^32 and 64 bits above, so the dtype does not change the values, and
`width` names it as the `u32` and `u64` fills of the C and CUDA ports do. A rejected draw
retries on the fallback stream `split(g)` of `sub(0x424c573332)` (`0x424c573634` for 64 bits)
from position 0, with `g` the index of the draw in the key's stream, so a fill cut at any
element boundary equals the whole fill. The retry loop runs only when a draw was rejected.

## Normals

`float64` normals are the 1024-layer ziggurat of Appendix A: element `i` from 64-bit draw `i`
and the spec's tables, which `src/tandem_jax` holds as the spec's file
`normal_f64_zig1024.json`. A draw that misses the inner rectangles, 0.43 % of them, continues on
the fallback stream `split(g)` of `sub(0x4e524d3634)`, with `g` the index of the draw in the
key's stream. XLA fuses a product into the sum that reads it, so the XLA path emulates the
reference logarithm's fused multiply-adds exactly and keeps every other product out of a sum.

The XLA path writes each miss as NaN and packs the misses into 32-bit words. A binary search over
the running count of the words then finds miss `k`, where `nonzero` over the fill took longer
than the fill. The slow path takes a batch of the expected misses plus six standard deviations,
and more misses take further batches. Two of its rounds settle all but a few dozen misses, and
the rest finish on a gathered set of 256. XLA fuses an unrolled F into each consumer of its
eight words, so the slow path runs F as a loop.

`float32` normals are Box-Muller pairs: elements `2j` and `2j + 1` are `r cos 2 pi b` and
`r sin 2 pi b` from uniform draws `2j` and `2j + 1`, computed in `float32` in one fused `jit`.
