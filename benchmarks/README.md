# mkl_umath ASV Benchmarks

Performance benchmarks for [mkl_umath](https://github.com/IntelPython/mkl_umath) using [Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

The `npbench/` suite uses kernels from [npbench](https://github.com/spcl/npbench) to measure end-to-end impact of MKL ufunc acceleration in realistic workloads.

### Coverage

| File | Ufuncs | Dtypes | Sizes/Presets |
|------|--------|--------|---------------|
| `micro/bench_micro.py` | 25 unary (`exp`, `log`, `sin`, `cos`, `sqrt`, `cbrt`, etc.) + `arctan2`, `power` | float32, float64 | 10k, 100k, 1M |
| `npbench/bench_softmax.py` | `exp`, `max`, `sum` | float32 | M (32x8x256x256), L (64x16x448x448) |
| `npbench/bench_arc_distance.py` | `sin`, `cos`, `arctan2`, `sqrt` | float64 | M (1M), L (10M) |
| `npbench/bench_go_fast.py` | `tanh` | float64 | M (6k x 6k), L (20k x 20k) |
| `npbench/bench_mandelbrot.py` | `abs`, `multiply`, `add` | complex128 | M (250/500), L (833/1000) |

## Running Benchmarks

Prerequisites:

```bash
pip install asv psutil
```

Run benchmarks against the current commit:

```bash
asv run --python=same --quick HEAD^!
```

Compare two commits:

```bash
asv continuous --python=same HEAD~1 HEAD
```

View results in a browser:

```bash
asv publish
asv preview
```

## Threading

Set `MKL_NUM_THREADS` to control the thread count used by MKL:

```bash
MKL_NUM_THREADS=8 asv run --python=same --quick HEAD^!
```

If `MKL_NUM_THREADS` is not set, `__init__.py` applies a default: **4** threads when the machine has 4 or more physical cores, or **1** (single-threaded) otherwise. This keeps results comparable across CI machines in the shared pool regardless of their total core count. Physical cores are detected via `psutil.cpu_count(logical=False)` (hyperthreads excluded per MKL recommendation).
