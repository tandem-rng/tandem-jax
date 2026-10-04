"""Spec vectors, Julia stream dumps, and the registered key implementation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tandem_jax as tj  # noqa: E402

HERE = Path(__file__).parent
V = json.loads((HERE / "vectors.json").read_text())
KEY = jnp.array([int(w, 16) for w in V["key"]], jnp.uint32)
KEY1234 = jnp.array([1, 2, 3, 4], jnp.uint32)


def words(ws):
    return np.array([int(w, 16) for w in ws], np.uint32)


def dump(name, dtype):
    return np.fromfile(HERE / "data" / name, dtype=dtype)


def typed(key_words):
    return jax.random.wrap_key_data(key_words, impl=tj.impl)


def test_T():
    for t in V["T"]:
        o, h = tj.T(tuple(jnp.uint32(x) for x in words(t["o"])), tuple(jnp.uint32(x) for x in words(t["h"])))
        assert np.array_equal(np.array(o), words(t["o_out"]))
        assert np.array_equal(np.array(h), words(t["h_out"]))


def test_F():
    for f in V["F"]:
        o, h = tj.F_keyed(tuple(KEY), (jnp.uint32(f["counter"]), jnp.uint32(0)), tj._core.DOMAIN_STREAM, tj._core.AUX_STREAM)
        assert np.array_equal(np.array(o), words(f["o"]))
        assert np.array_equal(np.array(h), words(f["h"]))


def test_stream_words():
    got, pos = tj.stream(KEY, 0, 64, jnp.uint32)
    assert pos == 64 * 32
    for s in V["stream_words"]:
        i = s["first_word"]
        assert np.array_equal(np.array(got[i : i + 4]), words(s["words"]))
        assert np.array_equal(np.array(tj.block(KEY, 8 * (s["row"] // 32) + s["lane"], s["row"] % 32)), words(s["words"]))


def test_draws_from_position_0():
    d = V["draws_from_position_0"]
    f64, _ = tj.stream(KEY, 0, 17, jnp.float64)
    f32, _ = tj.stream(KEY, 0, 3, jnp.float32)
    for i, x in d["Float64"].items():
        assert float(f64[int(i)]) == x
    for i, x in d["Float32"].items():
        assert float(f32[int(i)]) == np.float32(x)
    u8, _ = tj.stream(KEY, 0, 17, jnp.uint8)
    for i, b in d["Bool"].items():
        assert int((u8[int(i) // 8] >> (int(i) % 8)) & 1) == b


def test_derived_keys():
    k = V["derived_keys"]
    kids = jax.random.key_data(jax.random.split(typed(KEY), 2))
    assert np.array_equal(np.array(kids[0]), words(k["split_child_0"]))
    assert np.array_equal(np.array(kids[1]), words(k["split_child_1"]))
    assert np.array_equal(np.array(jax.random.key_data(jax.random.fold_in(typed(KEY), 7))), words(k["purpose_7"]))
    forks, pos = tj.fork_words(KEY, 0, 3)
    assert np.array_equal(np.array(forks[0]), words(k["fork_child_0_at_block_0"]))
    assert pos == 128


def test_seed_whitening():
    s = V["seed_whitening"]
    k = tj.key(s["seed"])
    assert np.array_equal(np.array(jax.random.key_data(k)), words(s["key"]))
    f64, _ = tj.stream(k, 0, 17, jnp.float64)
    for i, x in s["Float64"].items():
        assert float(f64[int(i)]) == x
    u32, _ = tj.stream(k, 0, 1, jnp.uint32)
    assert int(u32[0]) == int(s["UInt32"]["0"], 16)


@pytest.mark.parametrize(
    "name, key, K, dtype",
    [
        ("k1234_K32_u32.bin", KEY1234, 32, jnp.uint32),
        ("k1234_K32_u64.bin", KEY1234, 32, jnp.uint64),
        ("k1234_K8_u32.bin", KEY1234, 8, jnp.uint32),
        ("seed42_K32_f64.bin", 42, 32, jnp.float64),
        ("seed42_K32_f32.bin", 42, 32, jnp.float32),
        ("seed42_K32_u8.bin", 42, 32, jnp.uint8),
        ("seed42_K32_f16bits.bin", 42, 32, jnp.float16),
    ],
)
def test_dumps(name, key, K, dtype):
    want = dump(name, np.dtype(dtype))
    k = tj.key(key) if isinstance(key, int) else key
    got, pos = tj.stream(k, 0, len(want), dtype, chunk_length=K)
    assert np.array_equal(np.array(got).view(want.dtype), want)
    assert int(pos) == len(want) * want.dtype.itemsize * 8
    # Positioned reads agree with the dump at offsets, including mid-row and mid-word starts.
    for start in (1, 7, 33, 100, 257):
        part, _ = tj.stream(k, start * want.dtype.itemsize * 8, 300, dtype, chunk_length=K)
        assert np.array_equal(np.array(part).view(want.dtype), want[start : start + 300])


def test_bool_dump():
    want = dump("seed42_K32_bool.bin", np.uint8)
    bits = np.unpackbits(np.array(jax.random.bits(tj.key(42), (len(want) // 8,), jnp.uint8)), bitorder="little")
    assert np.array_equal(bits, want)


def test_random_bits_widths():
    k = tj.key(42)
    u32 = np.array(jax.random.bits(k, (64,), jnp.uint32))
    assert np.array_equal(u32, np.array(tj.stream(k, 0, 64, jnp.uint32)[0]))
    assert np.array_equal(np.array(jax.random.bits(k, (32,), jnp.uint64)).view(np.uint32), u32)
    assert np.array_equal(np.array(jax.random.bits(k, (128,), jnp.uint16)).view(np.uint32), u32)
    assert np.array_equal(np.array(jax.random.bits(k, (256,), jnp.uint8)).view(np.uint32), u32)


def test_alignment():
    k = tj.key(42)
    u8, p = tj.stream(k, 0, 1, jnp.uint8)
    assert p == 8
    u64, p = tj.stream(k, p, 1, jnp.uint64)
    assert p == 128
    want, _ = tj.stream(k, 0, 2, jnp.uint64)
    assert int(u64[0]) == int(want[1])


def test_jax_uniform_is_not_the_spec_mapping():
    # JAX keeps 52 random bits for float64. The spec keeps 53. Document, do not force.
    k = tj.key(42)
    u = jax.random.uniform(k, (16,), jnp.float64)
    spec, _ = tj.stream(k, 0, 16, jnp.float64)
    raw = np.array(jax.random.bits(k, (16,), jnp.uint64))
    assert np.array_equal(np.array(u), (raw >> 12) * 2.0**-52)
    assert not np.array_equal(np.array(u), np.array(spec))


def test_jit_and_vmap():
    k = tj.key(42)
    f = jax.jit(lambda k: jax.random.bits(k, (8,), jnp.uint32))
    assert np.array_equal(np.array(f(k)), np.array(jax.random.bits(k, (8,), jnp.uint32)))
    keys = jax.random.split(k, 4)
    batched = jax.vmap(lambda k: jax.random.bits(k, (8,), jnp.uint32))(keys)
    for i in range(4):
        assert np.array_equal(np.array(batched[i]), np.array(jax.random.bits(keys[i], (8,), jnp.uint32)))
    s = jax.jit(lambda k, p: tj.stream(k, p, 100, jnp.float64)[0])
    assert np.array_equal(np.array(s(k, 640)), np.array(tj.stream(k, 640, 100, jnp.float64)[0]))


def _split_oracle(key_words, i):
    o, h = tj.F_keyed(
        tuple(key_words), (jnp.uint32((i >> 1) & 0xFFFFFFFF), jnp.uint32(i >> 33)), tj._core.DOMAIN_SPLIT, 0
    )
    return np.array(h if i & 1 else o)


def test_split_by_index():
    k = typed(KEY)
    kids = tj.split(k, jnp.arange(2, dtype=jnp.uint64))
    assert np.array_equal(np.array(jax.random.key_data(kids[0])), words(V["derived_keys"]["split_child_0"]))
    assert np.array_equal(np.array(jax.random.key_data(kids[1])), words(V["derived_keys"]["split_child_1"]))
    for i in (0, 1, 2**32 - 1, 2**32, 2**32 + 5, 2**40 + 7, 2**63 + 1, 2**64 - 1, 2**64 - 2):
        got = jax.random.key_data(tj.split(k, jnp.uint64(i)))
        assert np.array_equal(np.array(got), _split_oracle(KEY, i)), i


def test_split_jit_array_and_raw_key():
    f = jax.jit(tj.split)
    idx = jnp.array([3, 2**33, 2**62 + 1], jnp.uint64)
    got = jax.random.key_data(f(KEY, idx))
    assert got.shape == (3, 4)
    for g, i in zip(np.array(got), (3, 2**33, 2**62 + 1)):
        assert np.array_equal(g, _split_oracle(KEY, i))
    # jax.random.split children are split(0..n-1).
    n = 9
    a = jax.random.key_data(jax.random.split(typed(KEY), n))
    b = jax.random.key_data(tj.split(KEY, jnp.arange(n, dtype=jnp.uint32)))
    assert np.array_equal(np.array(a), np.array(b))
    # A typed key from split is usable by jax.random.
    jax.random.bits(tj.split(KEY, 5), (2,), jnp.uint32)


def test_split_count_above_2_32():
    # Lowering runs the impl's split on abstract values, so nothing of that size is allocated.
    big = jax.jit(lambda k: jax.random.split(k, 2**32 + 2))
    assert "ui64" in big.lower(typed(KEY)).as_text()
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="x64"):
            jax.jit(lambda k: jax.random.split(k, 2**32 + 3)).lower(tj.key(1))


def _fork_oracle(key_words, position, i):
    b = position >> 7
    o, h = tj.F_keyed(
        tuple(key_words), (jnp.uint32(b & 0xFFFFFFFF), jnp.uint32(b >> 32)), tj._core.DOMAIN_FORK, (i >> 1) & 0xFFFFFFFF
    )
    return np.array(h if i & 1 else o)


def test_fork_typed_jit_and_traced_position():
    k = typed(KEY)
    kids, pos = tj.fork(k, 0, 3)
    assert jnp.issubdtype(kids.dtype, jax.dtypes.prng_key) and kids.shape == (3,)
    assert np.array_equal(np.array(jax.random.key_data(kids[0])), words(V["derived_keys"]["fork_child_0_at_block_0"]))
    assert int(pos) == 128
    f = jax.jit(lambda k, p: tj.fork(k, p, 5))
    for p in (0, 127, 128, 5000, 2**40 + 129, 2**62):
        kids, new = f(k, jnp.uint64(p))
        assert int(new) == ((p >> 7) + 1) << 7
        for i, c in enumerate(np.array(jax.random.key_data(kids))):
            assert np.array_equal(c, _fork_oracle(KEY, p, i)), (p, i)
    # Typed children feed jax.random directly, and a raw key is accepted too.
    jax.random.bits(kids[0], (2,), jnp.uint32)
    assert np.array_equal(np.array(tj.fork_words(KEY, 5000, 5)[0]), np.array(jax.random.key_data(f(k, 5000)[0])))
    assert tj.fork(k, 0, 0)[0].shape == (0,) and int(tj.fork(k, 0, 0)[1]) == 128


@pytest.mark.parametrize("name, dtype", [("seed42_K32_f64.bin", jnp.float64), ("seed42_K32_f32.bin", jnp.float32)])
def test_uniform_matches_dumps(name, dtype):
    want = dump(name, np.dtype(dtype))
    k = tj.key(42)
    got = tj.uniform(k, (len(want),), dtype)
    assert got.dtype == np.dtype(dtype) and np.array_equal(np.array(got), want)
    w = np.dtype(dtype).itemsize * 8
    assert np.array_equal(np.array(tj.uniform(k, (4, 5), dtype, 7 * w)), want[7:27].reshape(4, 5))
    assert np.array_equal(np.array(tj.uniform(k, 50, dtype, 13 * w)), want[13:63])
    f = jax.jit(lambda k, p: tj.uniform(k, (50,), dtype, p))
    assert np.array_equal(np.array(f(k, jnp.uint64(13 * w))), want[13:63])


def test_uniform_dtypes_range_and_jax_difference():
    k = tj.key(42)
    assert tj.uniform(k).shape == () and tj.uniform(k, (3,)).dtype == jnp.float64
    h = tj.uniform(k, (1000,), jnp.float16)
    assert h.dtype == jnp.float16 and float(h.min()) >= 0 and float(h.max()) < 1
    x = tj.uniform(k, (1000,), jnp.float64, minval=-2.0, maxval=3.0)
    assert float(x.min()) >= -2 and float(x.max()) < 3
    base = tj.uniform(k, (1000,), jnp.float64)
    assert np.array_equal(np.array(x), np.array(jnp.maximum(-2.0, base * 5.0 - 2.0)))
    assert not np.array_equal(np.array(jax.random.uniform(k, (16,), jnp.float64)), np.array(tj.uniform(k, (16,), jnp.float64)))
    with pytest.raises(ValueError):
        tj.uniform(k, (2,), jnp.uint32)


def test_bool_stream_matches_dump():
    want = dump("seed42_K32_bool.bin", np.uint8).astype(bool)
    k = tj.key(42)
    got, pos = tj.stream(k, 0, len(want), jnp.bool_)
    assert got.dtype == jnp.bool_ and int(pos) == len(want)
    assert np.array_equal(np.array(got), want)
    for start in (1, 5, 31, 32, 33, 1000):
        part, _ = tj.stream(k, start, 300, jnp.bool_)
        assert np.array_equal(np.array(part), want[start : start + 300])
    # Bool draws are single bits: no alignment, and they interleave with wider draws.
    _, pos = tj.stream(k, 3, 2, jnp.bool_)
    assert int(pos) == 5
    assert int(tj.stream(k, pos, 1, jnp.uint8)[1]) == 16


@pytest.mark.parametrize("signed, unsigned", [("int8", "uint8"), ("int16", "uint16"), ("int32", "uint32"), ("int64", "uint64")])
def test_signed_ints_reinterpret(signed, unsigned):
    k = tj.key(42)
    u = np.array(tj.stream(k, 5, 200, getattr(jnp, unsigned))[0])
    s, pos = tj.stream(k, 5, 200, getattr(jnp, signed))
    assert s.dtype == np.dtype(signed) and np.array_equal(np.array(s), u.view(signed))
    assert (u.view(signed) < 0).any()
    assert int(pos) == int(tj.stream(k, 5, 200, getattr(jnp, unsigned))[1])


@pytest.mark.parametrize("name, dtype", [("seed42_K32_c32.bin", jnp.complex64), ("seed42_K32_c64.bin", jnp.complex128)])
def test_complex_stream_matches_dump(name, dtype):
    want = dump(name, np.dtype(dtype))
    k = tj.key(42)
    got, pos = tj.stream(k, 0, len(want), dtype)
    assert got.dtype == np.dtype(dtype) and np.array_equal(np.array(got), want)
    w = np.dtype(dtype).itemsize * 4
    assert int(pos) == len(want) * 2 * w
    part, _ = tj.stream(k, 7 * 2 * w, 100, dtype)
    assert np.array_equal(np.array(part), want[7:107])
    # A complex draw is a real fill of twice the length, aligned to the component width.
    _, after_byte = tj.stream(k, 8, 3, dtype)
    assert int(after_byte) == ((8 + w - 1) & ~(w - 1)) + 6 * w


def test_chunk_length_variants():
    want = dump("k1234_K8_u32.bin", np.uint32)
    k8 = jax.random.wrap_key_data(KEY1234, impl=tj.impl_for(8))
    assert tj.impl_for(8) is tj.impl_for(8) and tj.impl_for(32) is tj.impl
    assert np.array_equal(np.array(jax.random.bits(k8, (len(want),), jnp.uint32)), want)
    # A typed key carries its variant into stream, including under jit, and into its children.
    assert np.array_equal(np.array(tj.stream(k8, 0, len(want), jnp.uint32)[0]), want)
    assert np.array_equal(np.array(jax.jit(lambda k: tj.stream(k, 0, 1000, jnp.uint32)[0])(k8)), want[:1000])
    assert not np.array_equal(np.array(tj.stream(typed(KEY1234), 0, 1000, jnp.uint32)[0]), want[:1000])
    child = jax.random.split(k8)[0]
    assert np.array_equal(
        np.array(jax.random.bits(child, (500,), jnp.uint32)),
        np.array(tj.stream(jax.random.key_data(child), 0, 500, jnp.uint32, chunk_length=8)[0]),
    )
    assert tj.fork(k8, 0, 2)[0].dtype == k8.dtype and tj.split(k8, 3).dtype == k8.dtype
    # Every variant shares seeding, split and fold_in.
    k1, k2 = tj.key(42, 8), tj.key(42)
    assert np.array_equal(np.array(jax.random.key_data(k1)), np.array(jax.random.key_data(k2)))
    for c in (1, 2, 64, 1024, 65536):
        a = jax.random.bits(tj.key(42, c), (3000,), jnp.uint32)
        assert np.array_equal(np.array(a), np.array(tj.stream(jax.random.key_data(tj.key(42, c)), 0, 3000, jnp.uint32, chunk_length=c)[0]))
    for bad in (0, 3, 131072, -8):
        with pytest.raises(ValueError, match="power of two"):
            tj.impl_for(bad)


def test_cross_port_fixture_from_c_reference():
    # tests/cross_port.json is written by tools/gen_split_fixture.c from the C reference.
    X = json.loads((HERE / "cross_port.json").read_text())
    k = tj.key(X["seed"])
    assert np.array_equal(np.array(jax.random.key_data(k)), words(X["key"]))
    for i, want in X["split"].items():
        got = jax.random.key_data(tj.split(k, jnp.uint64(int(i))))
        assert np.array_equal(np.array(got), words(want)), i
    for p, want in X["fork"].items():
        kids, new = jax.jit(lambda k, p: tj.fork(k, p, 3))(k, jnp.uint64(int(p)))
        assert int(new) == want["new_position"], p
        for g, w in zip(np.array(jax.random.key_data(kids)), want["children"]):
            assert np.array_equal(g, words(w)), p
    for u, want in X["sub"].items():
        got = jax.random.key_data(tj.sub(k, jnp.uint64(int(u))))
        if int(u) < 2**32:
            assert np.array_equal(np.array(jax.random.key_data(jax.random.fold_in(k, int(u)))), words(want))
        assert np.array_equal(np.array(got), words(want)), u


D = json.loads((HERE / "cross_derived.json").read_text())
ULP32 = 2.0**-23


def close(got, want, dtype, ulps32=8):
    got, want = np.array(got), np.array(want, dtype)
    tol = 1e-12 * np.abs(want) if dtype == np.float64 else ulps32 * ULP32 * np.abs(want) + 1e-6
    assert (np.abs(got - want) <= tol).all(), np.abs(got - want).max()


def test_normal_pairs_match_c_reference():
    k = tj.key(42)
    for dtype, name, end in ((jnp.float64, "pairs64", "pairs64_end_pos"), (jnp.float32, "pairs32", "pairs32_end_pos")):
        z, pos = tj.stream_normal(k, 1, len(D[name]), dtype)
        assert z.dtype == np.dtype(dtype) and int(pos) == D[end]
        close(z, D[name], np.dtype(dtype).type)


def test_normal_fills_match_cuda_fixtures():
    k = jax.random.wrap_key_data(jnp.array([0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82], jnp.uint32), impl=tj.impl)
    for name, dtype, w in (("device_normal64", jnp.float64, 64), ("device_normal32", jnp.float32, 32)):
        for c in D[name]:
            z, pos = tj.stream_normal(k, c["pos"], c["n"], dtype)
            close(z, c["out"], np.dtype(dtype).type)
            start = (c["pos"] + w - 1) // w * w
            assert int(pos) == start + w * 2 * ((c["n"] + 1) // 2)
            assert np.array_equal(np.array(tj.normal(k, c["n"], dtype, c["pos"])), np.array(z))


def test_normal_edge_cases():
    k = tj.key(42)
    z, pos = tj.stream_normal(k, 5, 0, jnp.float64)
    assert z.shape == (0,) and int(pos) == 5
    odd, pos_odd = tj.stream_normal(k, 0, 5, jnp.float64)
    even, pos_even = tj.stream_normal(k, 0, 6, jnp.float64)
    assert np.array_equal(np.array(odd), np.array(even[:5])) and int(pos_odd) == int(pos_even) == 6 * 64
    big = tj.normal(k, (200_000,), jnp.float64)
    assert abs(float(big.mean())) < 0.01 and abs(float(big.std()) - 1) < 0.01
    f = jax.jit(lambda k, p: tj.stream_normal(k, p, 100, jnp.float32)[0])
    # Fusion may change the last bit of a float32 sin or cos.
    close(f(k, jnp.uint64(640)), np.array(tj.stream_normal(k, 640, 100, jnp.float32)[0]), np.float32, ulps32=2)
    with pytest.raises(TypeError):
        tj.normal(k, (2,), jnp.float16)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_normal_moments_and_ks(dtype):
    # Bounds are five standard errors for the moments and the 0.1% KS critical value.
    n = 10**7
    z = np.array(tj.normal(tj.key(2026), (n,), dtype, 77), np.float64)
    m, v = z.mean(), z.var()
    skew, kurt = ((z - m) ** 3).mean() / v**1.5, ((z - m) ** 4).mean() / v**2 - 3
    assert abs(m) < 5 / n**0.5 and abs(v - 1) < 5 * (2 / n) ** 0.5
    assert abs(skew) < 5 * (6 / n) ** 0.5 and abs(kurt) < 5 * (24 / n) ** 0.5
    cdf = np.array(jax.scipy.special.ndtr(jnp.sort(jnp.asarray(z))))
    i = np.arange(1, n + 1)
    assert max((i / n - cdf).max(), (cdf - (i - 1) / n).max()) < 1.95 / n**0.5


def test_randint_uniform_through_rejections():
    # Range 3 * 2^30 rejects a quarter of the draws, so the fallback supplies a quarter of the
    # values. Chi-square over 1000 equal bins, bound at five standard deviations.
    n, r, bins = 10**6, 3 * 2**30, 1000
    x = np.array(tj.stream_randint(tj.key(9), 12345, n, 0, r, jnp.uint32)[0]).astype(np.uint64)
    counts = np.bincount((x * bins // r).astype(np.int64), minlength=bins)
    chi2 = ((counts - n / bins) ** 2 / (n / bins)).sum()
    assert abs(chi2 - (bins - 1)) < 5 * (2 * (bins - 1)) ** 0.5


@pytest.mark.parametrize("name, dtype", [("fill_below32", jnp.uint32), ("fill_below64", jnp.uint64)])
def test_randint_matches_c_fills(name, dtype):
    k, w = tj.key(42), 32 if dtype == jnp.uint32 else 64
    for c in D[name]:
        got, pos = tj.stream_randint(k, c["start"], len(c["out"]), 0, c["range"], dtype, w)
        assert np.array_equal(np.array(got), np.array(c["out"], dtype)), (c["start"], c["range"])
        assert int(pos) == c["end_pos"]
        assert np.array_equal(np.array(tj.randint(k, (len(c["out"]),), 0, c["range"], dtype, c["start"], w)), np.array(got))
    assert {c["start"] for c in D[name]} >= {0, 1, 12345}
    # The fallback key depends on the start, so the fixtures at 12345 must contain rejections.
    big = [c for c in D[name] if c["start"] == 12345 and c["range"] > 2**31]
    plain = lambda c: (np.array(tj.stream(k, 12345, 64, dtype)[0]).astype(object) * c["range"]) >> w
    assert any(list(plain(c)) != c["out"] for c in big)


@pytest.mark.parametrize("name, w", [("scalar_below32", 32), ("scalar_below64", 64)])
def test_scalar_bounded_fixtures_match_sequential_lemire(name, w):
    # A scalar draw rejects by taking the next draw of the main stream, which stream_randint
    # does not do, so the fixture is checked against Lemire's loop over the plain draws.
    k = tj.key(42)
    draws = [int(x) for x in np.array(tj.stream(k, 1, 600, jnp.dtype(f"uint{w}"))[0])]
    for c in D[name]:
        r, out, used = c["range"], [], 0
        t = ((1 << w) - r) % r
        while len(out) < len(c["out"]):
            m = draws[used] * r
            used += 1
            if m % (1 << w) >= t:
                out.append(m >> w)
        assert out == c["out"], r
        # The fixture starts at bit 1, which aligns up to one draw width.
        assert w + w * used == c["end_pos"], r


@pytest.mark.parametrize("name, dtype", [("device_below32", jnp.uint32), ("device_below64", jnp.uint64)])
def test_randint_matches_cuda_fills_with_rejections(name, dtype):
    k = jax.random.wrap_key_data(jnp.array([0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82], jnp.uint32), impl=tj.impl)
    assert max(c["rejected"] for c in D[name]) > 30
    for c in D[name]:
        want, w = np.array(c["out"], dtype), 32 if dtype == jnp.uint32 else 64
        assert np.array_equal(np.array(tj.stream_randint(k, 0, 64, 0, c["range"], dtype, w)[0]), want), c["range"]
        assert np.array_equal(np.array(tj.randint(k, (64,), 0, c["range"], dtype, width=w)), want)


def test_randint_bounds_dtypes_and_heavy_rejection():
    k = tj.key(42)
    x = tj.randint(k, (1000,), -5, 7, jnp.int8)
    assert x.dtype == jnp.int8 and int(x.min()) >= -5 and int(x.max()) < 7 and len(np.unique(np.array(x))) == 12
    assert np.array_equal(np.array(tj.randint(k, (50,), 3, 3, jnp.int32)), np.full(50, 3))
    assert np.array_equal(np.array(tj.randint(k, (50,), 9, 3, jnp.int32)), np.full(50, 9))
    lo, hi = jnp.arange(300, dtype=jnp.int32), jnp.arange(300, dtype=jnp.int32) + 1
    assert np.array_equal(np.array(tj.randint(k, (300,), lo, hi, jnp.int32)), np.arange(300))
    s, pos = tj.stream_randint(k, 7, 0, 0, 10, jnp.int32)
    assert s.shape == (0,) and int(pos) == 7
    # A range of 2^31 + 1 rejects about half the draws. 4000 elements exceed the compact retry
    # block, so the full path runs, and its first 64 elements must equal the compact path's.
    r, n = 2**31 + 1, 4000
    full = np.array(tj.stream_randint(k, 0, n, 0, r, jnp.uint32)[0])
    assert (full < r).all()
    plain = (np.array(tj.stream(k, 0, n, jnp.uint32)[0]).astype(np.uint64) * r) >> 32
    assert 0.3 < np.mean(plain == full) < 0.7
    assert np.array_equal(np.array(tj.stream_randint(k, 0, 64, 0, r, jnp.uint32)[0]), full[:64])
    j = jax.jit(lambda k, p: tj.stream_randint(k, p, 100, -50, 50, jnp.int64)[0])
    assert np.array_equal(np.array(j(k, jnp.uint64(64))), np.array(tj.stream_randint(k, 64, 100, -50, 50, jnp.int64)[0]))


def test_randint_rejections_across_blocks(monkeypatch):
    # About 20 rejections spread over 100 blocks: the compact path must place each one by its
    # element index. Forcing the full path gives the reference.
    k, n, r = tj.key(7), 100_000, 1_000_000
    plain = (np.array(tj.stream(k, 0, n, jnp.uint32)[0]).astype(np.uint64) * r) >> 32
    got = np.array(tj.stream_randint(k, 0, n, 0, r, jnp.uint32)[0])
    changed = np.flatnonzero(plain != got)
    assert 5 < len(changed) <= 64 and changed.max() - changed.min() > 10_000
    monkeypatch.setattr(tj, "_RETRY_MAX", 1)
    assert np.array_equal(np.array(tj.stream_randint(k, 0, n, 0, r, jnp.uint32)[0]), got)


def test_randint_cut_equals_whole_at_rejections():
    # Range 2^31 + 1 rejects about half the draws, so the cuts straddle many rejected draws.
    k, r, n = tj.key(3), 2**31 + 1, 300
    for start in (0, 1, 12345):
        whole, end = tj.stream_randint(k, start, n, 0, r, jnp.int64)
        plain = (np.array(tj.stream(k, start, n, jnp.uint32)[0]).astype(np.uint64) * r) >> 32
        assert 50 < np.sum(plain != np.array(whole)) < 250
        pos, parts = start, []
        for m in (1, 63, 101, 135):
            part, pos = tj.stream_randint(k, pos, m, 0, r, jnp.int64)
            parts.append(np.array(part))
        assert np.array_equal(np.concatenate(parts), np.array(whole)) and int(pos) == int(end)
    k64, r64 = tj.key(3), 2**63 + 1
    whole, end = tj.stream_randint(k64, 5, 200, 0, r64, jnp.uint64)
    a, mid = tj.stream_randint(k64, 5, 77, 0, r64, jnp.uint64)
    b, pos = tj.stream_randint(k64, mid, 123, 0, r64, jnp.uint64)
    assert np.array_equal(np.concatenate([np.array(a), np.array(b)]), np.array(whole)) and int(pos) == int(end)


def test_randint_width_follows_the_range_not_the_dtype():
    k = tj.key(11)
    for lo, hi in ((0, 1000), (-50, 2**31 - 1), (-(2**31), 2**31 - 1)):
        i32, p32 = tj.stream_randint(k, 7, 500, lo, hi, jnp.int32)
        i64, p64 = tj.stream_randint(k, 7, 500, lo, hi, jnp.int64)
        assert np.array_equal(np.array(i32), np.array(i64)) and int(p32) == int(p64) == 32 + 500 * 32
    # Above 2^32 the draws are 64 bits wide, below it they are 32 bits wide, under jit too.
    f = jax.jit(lambda k, hi: tj.stream_randint(k, 0, 8, 0, hi, jnp.int64))
    assert int(f(k, jnp.int64(2**32))[1]) == 8 * 32
    assert int(f(k, jnp.int64(2**32 + 1))[1]) == 8 * 64
    eager = tj.stream_randint(k, 0, 8, 0, 2**40, jnp.int64)
    assert np.array_equal(np.array(f(k, jnp.int64(2**40))[0]), np.array(eager[0]))
    # Range 2^32 is every 32-bit draw unchanged.
    full = tj.stream_randint(k, 0, 100, 0, 2**32, jnp.int64)[0]
    assert np.array_equal(np.array(full), np.array(tj.stream(k, 0, 100, jnp.uint32)[0]))
    s, p = tj.stream_randint(k, 3, 4, jnp.int64(-5), jnp.int64(-5), jnp.int64)
    assert np.array_equal(np.array(s), np.full(4, -5)) and int(p) == 32 + 4 * 32


def test_randint_clips_bounds_like_jax():
    k, n = tj.key(13), 4000
    # Bounds past the dtype clip, and a maxval above it includes the largest value.
    b = tj.randint(k, (n,), 0, 256, jnp.uint8)
    assert np.array_equal(np.array(b), np.array(tj.randint(k, (n,), 0, 256, jnp.int32)).astype(np.uint8))
    assert int(b.max()) == 255 and len(np.unique(np.array(b))) == 256
    s = tj.randint(k, (n,), -1000, 1000, jnp.int8)
    assert np.array_equal(np.array(s), np.array(tj.randint(k, (n,), -128, 128, jnp.int16)).astype(np.int8))
    assert np.array_equal(np.array(tj.randint(k, (9,), 300, 400, jnp.uint8)), np.full(9, 255))
    # The whole range of a 32- or 64-bit dtype is every draw of that width, offset by minval.
    u32 = np.array(tj.stream(k, 3, n, jnp.uint32)[0])
    i32 = tj.stream_randint(k, 3, n, -(2**31), 2**31, jnp.int32)
    assert np.array_equal(np.array(i32[0]), (u32 ^ np.uint32(2**31)).view(np.int32)) and int(i32[1]) == 32 + 32 * n
    assert np.array_equal(np.array(tj.randint(k, (n,), 0, 2**32, jnp.uint32, 3)), u32)
    assert np.array_equal(np.array(tj.randint(k, (n,), -(2**31), 2**31, jnp.int64, 3)), np.array(i32[0]))
    u64 = np.array(tj.stream(k, 3, n, jnp.uint64)[0])
    assert np.array_equal(np.array(tj.randint(k, (n,), 0, 2**64, jnp.uint64, 3)), u64)
    assert np.array_equal(np.array(tj.randint(k, (n,), -(2**63), 2**63, jnp.int64, 3)), (u64 ^ np.uint64(2**63)).view(np.int64))
    # Range 2^32 in 64-bit draws keeps the high half of each draw, whatever the dtype.
    w64 = np.array(tj.randint(k, (n,), 0, 2**32, jnp.uint32, 3, width=64))
    assert np.array_equal(w64, (u64 >> np.uint64(32)).astype(np.uint32))
    assert np.array_equal(np.array(tj.randint(k, (n,), 0, 2**32, jnp.int64, 3, width=64)), w64.astype(np.int64))
    f = jax.jit(lambda k, p: tj.randint(k, (n,), -(2**63), 2**63, jnp.int64, p))
    assert np.array_equal(np.array(f(k, jnp.uint64(3))), (u64 ^ np.uint64(2**63)).view(np.int64))
    for hi in (2**32 + 1, 2**64):
        with pytest.raises(ValueError, match="width 32"):
            tj.randint(k, (4,), 0, hi, jnp.uint64, width=32)


def test_randint_array_bounds_under_jit_and_vmap():
    keys = jax.random.split(tj.key(5), 4)
    f = lambda k, a, b: tj.randint(k, (300,), a, b, jnp.int64)
    # One row per width: each row takes its width from its own largest range.
    lo = jnp.stack([jnp.arange(300, dtype=jnp.int64) - 150, jnp.full(300, -7, jnp.int64)])
    hi = jnp.stack([lo[0] + 2**31 + 1, jnp.full(300, 2**35, jnp.int64)])
    batched = jax.jit(jax.vmap(f, (None, 0, 0)))(keys[1], lo, hi)
    for i in range(2):
        assert np.array_equal(np.array(batched[i]), np.array(f(keys[1], lo[i], hi[i])))
    his = jnp.array([10, 2**31 + 1, 2**33, 2**40 + 3], jnp.int64)
    rows = jax.jit(jax.vmap(f, (0, None, 0)))(keys, 0, his)
    for i in range(4):
        assert np.array_equal(np.array(rows[i]), np.array(f(keys[i], 0, int(his[i]))))


@pytest.mark.parametrize("name, dtype, w", [("device_below32_at", jnp.uint32, 32), ("device_below64_at", jnp.uint64, 64)])
def test_randint_matches_cuda_fills_at_nonzero_starts(name, dtype, w):
    k = jax.random.wrap_key_data(jnp.array([0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82], jnp.uint32), impl=tj.impl)
    assert sum(c["rejected"] for c in D[name]) > 50 and {c["start"] for c in D[name]} == {1, 12345}
    for c in D[name]:
        want = np.array(c["out"], dtype)
        assert np.array_equal(np.array(tj.stream_randint(k, c["start"], 64, 0, c["range"], dtype, w)[0]), want), c
        if (w == 32 or c["range"] > 2**32) and c["range"] < 2**63:
            # Width from the range: an int64 result type must not change a 32-bit draw.
            auto = tj.stream_randint(k, c["start"], 64, 0, c["range"], jnp.int64)[0]
            assert np.array_equal(np.array(auto).astype(np.uint64), want.astype(np.uint64)), c


CUDA = jax.default_backend() == "gpu" and tj._ffi.tandem_jax_cuda is not None
cuda_only = pytest.mark.skipif(not CUDA, reason="needs a CUDA device and the tandem_jax_cuda extension")


def _run_on(device, f, *args):
    with jax.default_device(device):
        return np.array(jax.jit(f)(*(jax.device_put(a, device) for a in args)))


@cuda_only
@pytest.mark.parametrize("K", [8, 32])
def test_cuda_kernels_equal_the_xla_path(K):
    # Odd, unaligned and 2^33-bit starts and fills over many thread blocks exercise the geometry
    # the kernels derive on the device, and the fallback keys of rejected draws far into the stream.
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    kd = tj.key_data(tj.key(77))
    typed = lambda kd: jax.random.wrap_key_data(kd, impl=tj.impl_for(K))
    draws = [(lambda d: lambda kd, p, n: tj.stream(typed(kd), p, n, d)[0])(d) for d in ("uint32", "int32", "float32", "uint64", "int64", "float64")]
    normals = [(lambda d: lambda kd, p, n: tj.stream_normal(typed(kd), p, n, d)[0])(d) for d in ("float32", "float64")]
    bounded = [
        (lambda a, b, d, w: lambda kd, p, n: tj.stream_randint(typed(kd), p, n, a, b, d, w)[0])(*c)
        for c in ((0, 1000, "int32", None), (-7, 2**31 + 1, "int64", None), (0, 2**32, "int64", None), (3, 2**40, "int64", None),
                  (0, 2**63 + 1, "uint64", None), (5, 5, "int32", None), (-9, 1000, "int32", 64), (3, 250, "uint8", None))
    ]
    for n in (7, 70001):
        for i, f in enumerate(draws + normals + bounded):
            g = lambda kd, p: f(kd, p, n)
            for p in (0, 1, 12345, 2**33 + 7):
                got, want = _run_on(gpu, g, kd, jnp.uint64(p)), _run_on(cpu, g, kd, jnp.uint64(p))
                if f in normals:
                    close(got, want, want.dtype.type, ulps32=16)
                else:
                    assert np.array_equal(got, want), (i, n, p)


@cuda_only
def test_cuda_kernels_batch_under_vmap():
    keys = jax.random.split(tj.key(5), 3)
    pos = jnp.array([0, 33, 2**33 + 1], jnp.uint64)
    for f in (
        lambda k, p: tj.stream(k, p, 1001, jnp.float32)[0],
        lambda k, p: tj.stream_normal(k, p, 1001, jnp.float64)[0],
        lambda k, p: tj.stream_randint(k, p, 1001, -3, 2**31 + 1, jnp.int64)[0],
    ):
        batched = np.array(jax.jit(jax.vmap(f))(keys, pos))
        for i in range(3):
            assert np.array_equal(batched[i], np.array(f(keys[i], pos[i])))


@cuda_only
def test_cuda_lowers_to_the_kernels():
    k = tj.key(1)
    f = lambda: (tj.uniform(k, 10), tj.normal(k, 10), tj.randint(k, 10, 0, 9), jax.random.bits(k, (10,), jnp.uint32))
    assert jax.jit(f).lower().compile().as_text().count('custom_call_target="tandem_fill"') == 4
    with jax.default_device(jax.devices("cpu")[0]):
        assert ("tandem_fill" in jax.jit(f).lower().compile().as_text()) == CPU


CPU = tj._ffi.tandem_jax_cpu is not None
cpu_only = pytest.mark.skipif(not CPU, reason="needs the tandem_jax_cpu extension")


def _without_cpu_extension(monkeypatch, f):
    """`f()` lowered without the CPU extension, so on the XLA path."""
    monkeypatch.setattr(tj._ffi, "tandem_jax_cpu", None)
    jax.clear_caches()
    try:
        return f()
    finally:
        monkeypatch.undo()
        jax.clear_caches()


@cpu_only
@pytest.mark.parametrize("K", [1, 32])
def test_cpu_fills_equal_the_xla_path(K, monkeypatch):
    # Odd, unaligned and 2^33-bit starts, and fills large enough to split over the thread pool,
    # with rejected bounded draws far into the stream.
    cpu = jax.devices("cpu")[0]
    kd = tj.key_data(tj.key(77))
    typed = lambda kd: jax.random.wrap_key_data(kd, impl=tj.impl_for(K))
    dtypes = ("uint8", "int8", "uint16", "float16", "uint32", "int32", "float32", "uint64", "int64", "float64")
    draws = [(lambda d: lambda kd, p, n: tj.stream(typed(kd), p, n, d)[0])(d) for d in dtypes]
    normals = [(lambda d: lambda kd, p, n: tj.stream_normal(typed(kd), p, n, d)[0])(d) for d in ("float32", "float64")]
    bounded = [
        (lambda a, b, d, w: lambda kd, p, n: tj.stream_randint(typed(kd), p, n, a, b, d, w)[0])(*c)
        for c in ((0, 1000, "int32", None), (-7, 2**31 + 1, "int64", None), (0, 2**32, "int64", None), (3, 2**40, "int64", None),
                  (0, 2**63 + 1, "uint64", None), (5, 5, "int32", None), (-9, 1000, "int32", 64), (3, 250, "uint8", None))
    ]
    cases = [(f, n, p) for f in draws + normals + bounded for n in (7, 300001) for p in (1, 2**33 + 7)]
    run = lambda: [_run_on(cpu, lambda kd, p: f(kd, p, n), kd, jnp.uint64(p)) for f, n, p in cases]
    got, want = run(), _without_cpu_extension(monkeypatch, run)
    for (f, n, p), g, w in zip(cases, got, want):
        if f in normals:
            close(g, w, w.dtype.type)
        else:
            assert np.array_equal(g, w), (cases.index((f, n, p)), n, p)


@cpu_only
def test_cpu_normals_equal_the_c_fixtures_bit_for_bit():
    k = tj.key(42)
    for dtype, name in ((jnp.float64, "pairs64"), (jnp.float32, "pairs32")):
        z, _ = tj.stream_normal(k, 1, len(D[name]), dtype)
        assert np.array_equal(np.array(z), np.array(D[name], dtype)), name


@cpu_only
def test_cpu_fills_batch_under_vmap():
    keys = jax.random.split(tj.key(5), 3)
    pos = jnp.array([0, 33, 2**33 + 1], jnp.uint64)
    for f in (
        lambda k, p: tj.stream(k, p, 300001, jnp.float32)[0],
        lambda k, p: tj.stream(k, p, 1001, jnp.uint8)[0],
        lambda k, p: tj.stream_normal(k, p, 1001, jnp.float64)[0],
        lambda k, p: tj.stream_randint(k, p, 1001, -3, 2**31 + 1, jnp.int64)[0],
    ):
        batched = np.array(jax.jit(jax.vmap(f))(keys, pos))
        for i in range(3):
            assert np.array_equal(batched[i], np.array(f(keys[i], pos[i])))


@cpu_only
def test_cpu_lowers_to_the_fills(monkeypatch):
    k = tj.key(1)
    f = lambda: jax.jit(lambda: (tj.uniform(k, 10), tj.normal(k, 10), tj.randint(k, 10, 0, 9), jax.random.bits(k, (10,), jnp.uint32))).lower().compile().as_text()
    assert f().count('custom_call_target="tandem_fill"') == 4
    assert "tandem_fill" not in _without_cpu_extension(monkeypatch, f)
