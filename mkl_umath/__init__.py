'''
Implementation of Numpy universal math functions using Intel(R) MKL and Intel(R) C compiler runtime.
'''

from ._version import __version__

from ._ufuncs import *

from ._patch import mkl_umath, use_in_numpy, restore, is_patched
