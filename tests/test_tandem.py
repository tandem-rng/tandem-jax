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
    forks, pos = tj.fork(KEY, 0, 3)
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
