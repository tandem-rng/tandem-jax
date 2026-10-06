# Tests

```sh
pixi run test     # tests/test_tandem.py and tests/test_conformance.py
```

## Suite

`tests/test_conformance.py` reads `tests/conformance`, copies of the spec's `conformance/*.json`
at commit `f420545`, and checks every item of the spec's `conformance/CHECKLIST.md`, on the native
fills and on the XLA path:

- every bounded, normal, exponential and weighted choice case, values and end positions, whole,
  cut at elements 1, 7, 20, 21 and `n - 1` (even ones for `float32` normals), and one element at
  a time,
- the scalar bounded cases, by Lemire's loop over the port's plain draws,
- the fallback index of rejected and missed draws, the width that follows the range, the empty
  fills, odd `n` and the pairs of the `float32` normals,
- the choice tables, rejected weights and `choice` under `jit`,
- the SHA-256 of the stream dumps and of the port's fills of them, and of the normal and
  exponential dumps,
- complex draws across a block, random access across rows and chunks, a draw at 2^63 - 1, and
  fills that would reach 2^64.

`tests/test_tandem.py`:

- checks every vector of the specification (`tests/vectors.json`),
- checks that a bounded fill cut at any element equals the whole fill and that `int32` and
  `int64` agree for a small range,
- checks `split`, `fork` and `sub` against fixed values from the C reference
  (`tests/cross_port.json`),
- checks the K = 8 variant, `uniform` and the `bool` and complex stream dtypes against the dumps,
- compares positioned reads and `jax.random.bits` with reference stream dumps in `tests/data`,
- checks the ziggurat tables against the spec file's SHA-256,
- checks that the XLA path's `float64` normals keep their values when the misses run in batches
  of one or three, or finish on a gathered set of one or four,
- checks fork children at traced positions, and split children for indices up to 2^64 - 1
  against a direct evaluation of F,
- checks normal moments to fourth order and a KS test on 10^7 draws, and chi-square uniformity
  of bounded integers at a range that rejects a quarter of the draws,
- and runs the key implementation under `jit` and `vmap`.

On a CUDA device with the extension, the whole suite runs on the kernels, and three more tests
check that the kernels equal the XLA path on the CPU device for every fill at odd, unaligned and
2^33-bit starts and over many thread blocks, that a `vmap` batch equals its rows, and that a GPU
lowering calls the kernels.

With the CPU extension, the whole suite runs on the tandem-c fills, and four more tests check
that the fills equal the XLA path for every fill at odd and 2^33-bit starts, at chunk lengths 1
and 32 and at sizes that split over the thread pool, that the `float32` normals equal tandem-c's
pairs bit for bit, that a `vmap` batch equals its rows, and that a CPU lowering calls the fills
and does not without the extension.

## Fixtures

- `tests/vectors.json` and `tests/conformance/*.json` are copies of the spec repository's files,
  with a drift check in CI.
- `tests/cross_port.json` is written by `tools/gen_split_fixture.c`.

## CI

- CI runs the suite on Linux and macOS, with Python 3.11 and 3.14, with and without the CPU
  extension.
- A separate job checks that `tests/vectors.json` equals the spec repository's `vectors.json` and
  `tests/conformance` its `conformance` directory at commit `f420545`.
