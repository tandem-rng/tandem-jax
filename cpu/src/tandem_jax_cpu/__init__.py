"""The tandem-c fills as one XLA FFI handler. `tandem_jax` registers it on import."""

import ctypes
import pathlib

import jax

_lib = ctypes.cdll.LoadLibrary(str(pathlib.Path(__file__).with_name("libtandem_ffi.so")))


def fill_handler():
    return jax.ffi.pycapsule(_lib.TandemFill)
