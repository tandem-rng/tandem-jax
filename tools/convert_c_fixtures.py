"""Writes tests/cross_derived.json from the fixture headers of the C reference.

    python tools/convert_c_fixtures.py ../tandem-c/tests > tests/cross_derived.json
"""

import ast
import json
import re
import sys
from pathlib import Path


def balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start : i + 1]


def literal(body):
    """A C initialiser as nested lists, with the integer and float suffixes dropped."""
    body = re.sub(r"(?<=[0-9.])(ull|u|f)\b", "", body)
    return ast.literal_eval(body.replace("{", "[").replace("}", "]"))


def array(text, name):
    m = re.search(rf"\b{name}\[[^\]]*\]\s*=\s*", text)
    return literal(balanced(text, m.end()))


def scalar(text, name):
    return int(re.search(rf"\b{name}\s*=\s*(\d+)", text).group(1))


tests = Path(sys.argv[1])
scalar_below = (tests / "cross_below.h").read_text()
fill, normal, cuda_below, cuda_normal = (
    (tests / f).read_text() for f in ("cross_fill_below.h", "cross_normal.h", "cuda_fill_below.h", "cuda_fill_normal.h")
)
out = {
    # Sequential draws after one Bool, so from bit position 1, rejected draws consume the stream.
    "scalar_below32": [{"range": n, "out": o, "end_pos": e} for n, o, e in array(scalar_below, "CROSS_U32")],
    "scalar_below64": [{"range": n, "out": o, "end_pos": e} for n, o, e in array(scalar_below, "CROSS_U64")],
    # Seed 42, fills from bit positions 0, 1 and 12345, which align up to the draw width.
    "fill_below32": [{"start": s, "range": n, "out": o, "end_pos": e} for s, n, o, e in array(fill, "CROSS_FILL_U32")],
    "fill_below64": [{"start": s, "range": n, "out": o, "end_pos": e} for s, n, o, e in array(fill, "CROSS_FILL_U64")],
    "pairs64": array(normal, "CROSS_NORMAL"),
    "pairs64_end_pos": scalar(normal, "CROSS_NORMAL_END_POS"),
    "pairs32": array(normal, "CROSS_NORMALF"),
    "pairs32_end_pos": scalar(normal, "CROSS_NORMALF_END_POS"),
    # Key of seed 42, K = 32, from the CUDA port. `rejected` counts elements on the fallback.
    "device_below32": [{"range": n, "rejected": r, "out": o} for n, r, o in array(cuda_below, "CROSS_BELOW32")],
    "device_below64": [{"range": n, "rejected": r, "out": o} for n, r, o in array(cuda_below, "CROSS_BELOW64")],
    "device_normal64": [{"pos": p, "n": n, "out": o[:n]} for p, n, o in array(cuda_normal, "CROSS_NORMAL64")],
    "device_normal32": [{"pos": p, "n": n, "out": o[:n]} for p, n, o in array(cuda_normal, "CROSS_NORMAL32")],
}
json.dump(out, sys.stdout, separators=(",", ":"))
print()
