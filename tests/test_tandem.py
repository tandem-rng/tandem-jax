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
