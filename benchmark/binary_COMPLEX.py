import numpy as np
import timeit
import gc
import mkl_umath


functions = ['maximum', 'minimum', 'fmax', 'fmin']
powers = list(range(4, 9))
sizes = [10**p for p in powers]
repeats = 3
dtypes = [np.complex64, np.complex128]

def time_func(func, *args):
    gc.collect()
    timer = timeit.Timer(lambda: func(*args))
    times = timer.repeat(repeat=repeats, number=1)
    return min(times) * 1000  # milliseconds

def benchmark_function(name, np_func, mkl_func):
    for dtype in dtypes:
        print(f"\nFunction: {name} | Dtype: {dtype.__name__}")
        print(f"{'Size':>10} | {'NumPy (ms)':>12} | {'MKL_umath (ms)':>15}")
        print("-" * 45)

        for power, size in zip(powers, sizes):
            try:
                a = np.random.rand(size) + 1j * np.random.rand(size)
                a = a.astype(dtype)
                b = np.random.rand(size) + 1j * np.random.rand(size)
                b = b.astype(dtype)                    
                args = (a, b)

                np_time = time_func(np_func, *args)
                mkl_time = time_func(mkl_func, *args)

                print(f"{f'10**{power}':>10} | {np_time:12.3f} | {mkl_time:15.3f}")
            except MemoryError:
                print(f"{f'10**{power}':>10} | {'OOM':>12} | {'OOM':>15}")
            except Exception as e:
                print(f"{f'10**{power}':>10} | Error: {str(e)}")

# Run benchmarks
for func in functions:
    np_func = getattr(np, func)
    mkl_func = getattr(mkl_umath, func)
    benchmark_function(func, np_func, mkl_func)
