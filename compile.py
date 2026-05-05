# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

extra_compile_args = [
    "-O3", "-march=native", "-mtune=native",
    "-mavx2", "-mfma", "-flto",
    "-ffast-math", "-funroll-loops", "-finline-functions",
    "-fomit-frame-pointer", "-fvisibility=hidden",
]

if sys.platform.startswith("linux"):
    extra_compile_args.extend(["-Wno-unused-function", "-Wno-unused-variable"])

setup(
    name="super_fast_sort",
    ext_modules=cythonize(
        Extension(
            "super_fast_sort",
            sources=["super_fast_sort.pyx"],           # ← ONLY pyx file!
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-flto"],
            language="c",
        ),
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
            "initializedcheck": False,
            "language_level": 3,
        },
        annotate=False,
    ),
    zip_safe=False,
)