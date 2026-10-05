# Tests

```sh
pixi run test     # tests/test_tandem.py
```

## Suite

`tests/test_tandem.py`:

- checks every vector of the specification (`tests/vectors.json`),
- checks `normal` and `randint` against the C and CUDA fixtures in `tests/cross_derived.json`,
  rejections included, fills from positions 0, 1 and 12345, from both C and CUDA,
- checks that a bounded fill cut at any element equals the whole fill and that `int32` and
  `int64` agree for a small range,
- checks `split`, `fork` and `sub` against fixed values from the C reference
  (`tests/cross_port.json`),
- checks the K = 8 variant, `uniform` and the `bool` and complex stream dtypes against the dumps,
- compares positioned reads and `jax.random.bits` with reference stream dumps in `tests/data`,
- checks the `float64` normals bit for bit against tandem-c's `tests/cross_normal.h` at commit
  `121db59` (rows with wedge, redraw and tail misses) and tandem-cuda's
  `tests/cross_fill_normal.h` at `76eddae`,
- checks the ziggurat tables against the spec file's SHA-256,
- checks fork children at traced positions, and split children for indices up to 2^64 - 1
  against a direct evaluation of F,
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

- `tests/vectors.json` is a copy of the spec repository's file, with a drift check in CI.
- `tests/cross_derived.json` is written by `tools/convert_c_fixtures.py`.
- `tests/cross_port.json` is written by `tools/gen_split_fixture.c`.
- tandem-c's `tests/cross_normal.h` at commit `121db59` has the SHA-256
  `3cd7c8f9178711255718288eb712eaccb33a1726d2a185f412f13590398ad3ac`.

## CI

- CI runs the suite on Linux and macOS, with Python 3.11 and 3.14, with and without the CPU
  extension.
- A separate job checks that `tests/vectors.json` equals the spec repository's `vectors.json`.
