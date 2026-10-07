"""The spec's conformance files, copies of tandem-spec 2a4bd08 conformance/*.json that CI checks
byte for byte, and every item of its conformance/CHECKLIST.md at 2a4bd08."""

import functools
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tandem_jax as tj  # noqa: E402

HERE = Path(__file__).parent
FILES = {name: json.loads((HERE / "conformance" / f"{name}.json").read_text()) for name in ("below", "fill_below", "normal", "exponential", "choice", "hashes")}
FILLS = [c for name in ("fill_below", "normal", "exponential", "choice") for c in FILES[name]["cases"]]
WIDTH = {"fill_below_u32": 32, "fill_below_u64": 64, "fill_normal_f64": 64, "fill_normal_f32": 32, "fill_exponential_f64": 64, "fill_exponential_f32": 32, "fill_choice": 64}


def named(name, id):
    return next(c for c in FILES[name]["cases"] if c["id"].endswith(" " + id))


def key(c):
    return jax.random.wrap_key_data(jnp.array([int(w, 16) for w in c["key"]], jnp.uint32), impl=tj.impl_for(c["K"]))


def expected(c):
    """The values of a case: integers as uint32 or uint64, floats from their bits."""
    w = WIDTH[c["kind"]] if c["kind"] != "fill_choice" else 32
    raw = np.array([int(v, 16) for v in c["values"]], np.dtype(f"uint{w}"))
    return raw.view(np.dtype(f"float{w}")) if "normal" in c["kind"] or "exponential" in c["kind"] else raw


def end(c):
    """The end position: the case's own, or the spec's rule for the draws a fill consumes."""
    w, n = WIDTH[c["kind"]], c["n"]
    if c["kind"] == "fill_normal_f32":
        n = 2 * ((n + 1) // 2)
    elif n == 0 and c["kind"] not in ("fill_normal_f64", "fill_choice"):
        return c["start"]
    return c.get("end", -(-c["start"] // w) * w + w * n)


@functools.partial(jax.jit, static_argnames=("n", "w"))
def _below(k, p, n, r, w):
    # Eager, each call would compile its retry loop again. The range is traced, so one compile
    # serves every case of a width and length.
    return tj.stream_randint(k, p, n, 0, r, r.dtype, w)


def fill(c):
    """The port's fill for a case, as `f(key, position, n) -> (values, end)`."""
    kind = c["kind"]
    if kind.startswith("fill_below"):
        w = WIDTH[kind]
        r = jnp.asarray(int(c["range"], 16), jnp.dtype(f"uint{w}"))
        return lambda k, p, n: _below(k, jnp.uint64(p), n, r, w)
    if kind == "fill_choice":
        table = tj.choice_table(np.array([int(v, 16) for v in c["weights"]], np.uint64).view(np.float64))
        return lambda k, p, n: tj.stream_choice(k, p, n, table)
    f = tj.stream_normal if "normal" in kind else tj.stream_exponential
    return lambda k, p, n: f(k, p, n, jnp.dtype("float" + kind[-2:]))


def same(got, c):
    got, want = np.asarray(got), expected(c)
    if c["kind"] == "fill_normal_f32" and "tol" in c:
        tol = c["tol"]["ulps"] * 2.0**-23 * np.abs(want) + c["tol"]["abs"]
        return got.dtype == want.dtype and got.shape == want.shape and bool((np.abs(got - want) <= tol).all())
    return got.dtype == want.dtype and np.array_equal(got, want)


@pytest.fixture(params=["native", "xla"])
def path(request, monkeypatch):
    """The native fills where they load, then the XLA path."""
    if request.param == "xla":
        monkeypatch.setattr(tj._ffi, "tandem_jax_cpu", None)
        monkeypatch.setattr(tj._ffi, "tandem_jax_cuda", None)
    jax.clear_caches()
    yield
    monkeypatch.undo()
    jax.clear_caches()


def test_fill_cases_whole_cut_and_one_element_at_a_time(path):
    # Float32 normal fills are cut at pair boundaries only, as the checklist says.
    assert len(FILLS) == 133 and sum(c["n"] == 0 for c in FILLS) == 7
    for c in FILLS:
        k, f, n = key(c), fill(c), c["n"]
        got, pos = f(k, c["start"], n)
        assert same(got, c) and int(pos) == end(c), c["id"]
        cuts = (2, 8, 20, (n - 1) & ~1) if c["kind"] == "fill_normal_f32" else (1, 7, 20, 21, n - 1)
        for cut in sorted({x for x in cuts if 0 < x < n}):
            head, mid = f(k, c["start"], cut)
            tail, pos = f(k, mid, n - cut)
            assert same(np.concatenate([head, tail]), c) and int(pos) == end(c), (c["id"], cut)
        if n and c["kind"] != "fill_normal_f32":
            pos, parts = c["start"], []
            for _ in range(n):
                x, pos = f(k, pos, 1)
                parts.append(np.asarray(x))
            assert same(np.concatenate(parts), c) and int(pos) == end(c), c["id"]


def test_scalar_bounded_draws_retry_on_the_next_draw():
    # The port has no scalar bounded draw. Lemire's loop over the port's plain draws, where a
    # rejection takes the next draw, must give the scalar cases.
    for c in FILES["below"]["cases"]:
        w, r = int(c["kind"][-2:]), int(c["range"], 16)
        draws = [int(x) for x in np.asarray(tj.stream(key(c), c["start"], 4 * c["n"], jnp.dtype(f"uint{w}"))[0])]
        t, out, used = ((1 << w) - r) % r, [], 0
        while len(out) < c["n"]:
            m = draws[used] * r
            used += 1
            if m % (1 << w) >= t:
                out.append(m >> w)
        assert out == [int(v, 16) for v in c["values"]], c["id"]
        assert -(-c["start"] // w) * w + w * used == c["end"], c["id"]


def test_fallback_is_keyed_by_the_global_draw_index():
    for name, a, b in (("fill_below", "CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"), ("fill_below", "CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"),
                       ("normal", "CROSS_NORMAL[1]", "CROSS_NORMAL[0]")):
        assert np.array_equal(expected(named(name, a))[:63], expected(named(name, b))[1:]), a
    assert named("fill_below", "CROSS_BELOW32_AT[4]")["rejected"] > 0


def test_width_follows_the_range():
    c32, c64 = named("fill_below", "CROSS_BELOW32[3]"), named("fill_below", "CROSS_BELOW64[3]")
    got, pos = tj.stream_randint(key(c32), 0, 64, 0, 1000, jnp.uint64)
    assert np.array_equal(np.asarray(got), expected(c32).astype(np.uint64)) and int(pos) == 64 * 32
    assert not np.array_equal(np.asarray(got), expected(c64))
    # Without a width an int64 result takes it from the range, which every case fitting int64 and
    # needing 64 bits, or a 32-bit case, must match.
    for c in FILES["fill_below"]["cases"]:
        r = int(c["range"], 16)
        if (c["kind"] == "fill_below_u32" or r > 2**32) and r < 2**63:
            auto = tj.stream_randint(key(c), c["start"], c["n"], 0, r, jnp.int64)[0]
            assert np.array_equal(np.asarray(auto).astype(np.uint64), expected(c).astype(np.uint64)), c["id"]
    # Range 0: minval, and one draw of the width.
    got, pos = tj.stream_randint(key(c32), 0, 1, 5, 5, jnp.int64)
    assert int(got[0]) == 5 and int(pos) == 32


def test_float32_normals_pair_up():
    pairs, at32 = expected(named("normal", "CROSS_NORMALF")), named("normal", "CROSS_NORMAL32[1]")
    assert np.array_equal(pairs[:33], expected(at32))
    assert np.array_equal(expected(named("normal", "CROSS_NORMAL32[2]"))[:31], expected(named("normal", "CROSS_NORMAL32[0]"))[2:33])
    # A fill of one is the cos half and consumes both draws of its pair.
    z, pos = tj.stream_normal(key(at32), 32, 1, jnp.float32)
    assert same(np.concatenate([z, expected(at32)[1:]]), at32) and int(pos) == 32 + 64


def test_choice_tables():
    for c in FILES["choice"]["cases"]:
        t = tj.choice_table(np.array([int(v, 16) for v in c["weights"]], np.uint64).view(np.float64))
        assert f"{int(t.capacity):016x}" == c["capacity"], c["id"]
        if "cut" in c:
            assert [f"{int(x):016x}" for x in t.cut] == c["cut"] and [f"{int(x):08x}" for x in t.alias] == c["alias"], c["id"]
    assert np.array_equal(expected(named("choice", "CROSS_CHOICE[1]"))[:63], expected(named("choice", "CROSS_CHOICE[0]"))[1:])
    single = named("choice", "CROSS_CHOICE[3]")
    assert len(single["weights"]) == 1 and not expected(single).any()
    for w in ([], [1, -1], [1, np.nan], [1, np.inf], [0.0, -0.0]):
        with pytest.raises(ValueError):
            tj.choice_table(w)


def test_choice_under_jit_and_in_shape():
    t, k = tj.choice_table([3, 0, 1, 7.5, 0.125]), tj.key(42)
    want = np.asarray(tj.stream_choice(k, 33, 1000, t)[0])
    assert np.array_equal(np.asarray(jax.jit(lambda p: tj.stream_choice(k, p, 1000, t)[0])(jnp.uint64(33))), want)
    assert np.array_equal(np.asarray(tj.choice(k, (10, 100), t, 33)), want.reshape(10, 100))
    assert not (want == 1).any()


STREAM_DTYPES = {"UInt32": jnp.uint32, "UInt64": jnp.uint64, "Float64": jnp.float64, "Float32": jnp.float32, "UInt8": jnp.uint8,
                 "Bool": jnp.bool_, "ComplexF64": jnp.complex128, "ComplexF32": jnp.complex64, "Float16": jnp.float16}


def test_stream_hashes():
    # JAX has no UInt128 or Char draws.
    done = 0
    for s in FILES["hashes"]["streams"]:
        assert hashlib.sha256((HERE / "data" / Path(s["file"]).name).read_bytes()).hexdigest() == s["sha256"]
        if s["type"] in STREAM_DTYPES:
            x = np.asarray(tj.stream(key(s), s["start"], s["n"], STREAM_DTYPES[s["type"]])[0])
            assert hashlib.sha256(x.astype(np.uint8) if x.dtype == bool else x).hexdigest() == s["sha256"], s["file"]
            done += 1
    assert done == 10


def test_dump_hashes():
    for d in FILES["hashes"]["dumps"]:
        if "sha256" not in d:
            continue
        h = hashlib.sha256()
        for start in d["starts"]:
            pos = start
            for draw in d["draws"]:
                f = tj.stream_normal if "normal" in draw["kind"] else tj.stream_exponential
                x, pos = f(key(d), pos, draw["n"], jnp.dtype("float" + draw["kind"][-2:]))
                h.update(np.asarray(x).tobytes())
        assert h.hexdigest() == d["sha256"], d["id"]


def test_block_and_2_63_boundaries():
    k = tj.key_data(tj.key(42))
    # The real part ends block 0 and the imaginary part starts block 1.
    z, _ = tj.stream(k, 64, 1, jnp.complex128)
    assert complex(z[0]) == complex(float(tj.stream(k, 64, 1, jnp.float64)[0][0]), float(tj.stream(k, 128, 1, jnp.float64)[0][0]))
    # Reads at any position equal the sequential fill, across blocks, rows and chunks of K = 8.
    k8 = jax.random.wrap_key_data(k, impl=tj.impl_for(8))
    whole = np.asarray(tj.stream(k8, 0, 3000, jnp.uint32)[0])
    for p in (3, 4, 31, 32, 255, 256, 2047, 2048, 2900):
        assert np.array_equal(np.asarray(tj.stream(k8, 32 * p, 100, jnp.uint32)[0]), whole[p : p + 100]), p
    # Positions are arguments of a stateless draw, so any uint64 is a valid successor position.
    _, pos = tj.stream(k, 2**63 - 1, 1, jnp.uint64)
    assert int(pos) == 2**63 + 64
    assert int(tj.stream(k, 2**64 - 128, 1, jnp.uint64)[1]) == 2**64 - 64
    for f in (lambda: tj.stream(k, 2**64 - 64, 1, jnp.uint64), lambda: tj.stream_normal(k, 2**64 - 100, 4, jnp.float32),
              lambda: tj.stream_exponential(k, 2**64 - 64, 1), lambda: tj.stream_randint(k, 2**64 - 64, 2, 0, 9, jnp.int32),
              lambda: tj.stream_choice(k, 2**64 - 64, 1, tj.choice_table([1.0]))):
        with pytest.raises(ValueError, match="2\\^64"):
            f()
