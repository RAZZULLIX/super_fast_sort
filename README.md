### Files

- `program.c`  
  C implementation of `super_fast_sort_uint32(uint32_t *arr, size_t n)`; in‑place hybrid sort specialized for `uint32_t`.

- `super_fast_sort.pyx`  
  Cython wrapper that exposes `super_fast_sort(arr)` (in‑place sort on a NumPy `uint32` array, returned) and `cleanup()` (frees internal C buffers).

- `compile.py`  
  Build script to compile the Cython extension `super_fast_sort` with aggressive optimization flags.

- `super_fast_sort_vs_numpy_benchmark.py`  
  Benchmark script comparing `super_fast_sort` vs `numpy.sort` on many random distributions and sizes, printing a small text dashboard.

### Commands

From the project directory:

- Build extension:

  ```bash
  python compile.py build_ext --inplace
  ```

- Example benchmark run:

  ```bash
  python super_fast_sort_vs_numpy_benchmark.py --detailed --size-hist --iter 10000 --max-n 100
  ```
Other useful flags: --time N, --distr uniform, --no-fail.

Here are some benchmarks against numpy.sort:

array size 100
<img width="925" height="258" alt="Screenshot from 2026-05-05 14-25-40" src="https://github.com/user-attachments/assets/c07db42b-cac3-474a-aa01-2cae930e1a54" />

array size 10000
<img width="923" height="254" alt="Screenshot from 2026-05-05 14-26-01" src="https://github.com/user-attachments/assets/ebbeea52-513e-4faf-824d-0ae35089f118" />
