#!/usr/bin/env python
# Copyright (c) 2019-2023, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib.machinery
import io
import os
import re
from distutils.dep_util import newer
from numpy.distutils.conv_template import process_file as process_c_file
from os import (getcwd, environ, makedirs)
from os import (getcwd, environ, makedirs)
from os.path import join, exists, abspath, dirname
from setuptools import Extension

import skbuild
import skbuild.setuptools_wrap
import skbuild.utils
from skbuild.command.build_py import build_py as _skbuild_build_py
from skbuild.command.install import install as _skbuild_install

# import versioneer

with io.open('mkl_umath/_version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

VERSION = version

CLASSIFIERS = """\
Development Status :: 0 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


def load_module(name, fn):
    """
    Credit: numpy.compat.npy_load_module
    """
    return importlib.machinery.SourceFileLoader(name, fn).load_module()

def separator_join(sep, strs):
    """
    Joins non-empty arguments strings with dot.

    Credit: numpy.distutils.misc_util.dot_join
    """
    assert isinstance(strs, (list, tuple))
    assert isinstance(sep, str)
    return sep.join([si for si in strs if si])

pdir = join(dirname(__file__), 'mkl_umath')
wdir = join(pdir, 'src')

generate_umath_py = join(pdir, 'generate_umath.py')
n = separator_join('_', ('mkl_umath', 'generate_umath'))
generate_umath = load_module(n, generate_umath_py)
del n

def generate_umath_c(build_dir):
    target_dir = join(build_dir, 'src')
    target = join(target_dir, '__umath_generated.c')
    if not exists(target_dir):
        print("Folder {} was expected to exist, but creating".format(target_dir))
        makedirs(target_dir)
    script = generate_umath_py
    if newer(script, target):
        with open(target, 'w') as f:
            f.write(generate_umath.make_code(generate_umath.defdict,
                                             generate_umath.__file__))
    return []

generate_umath_c(pdir)

loops_header_templ = join(wdir, "mkl_umath_loops.h.src")
processed_loops_h_fn = join(wdir, "mkl_umath_loops.h")
loops_header_processed = process_c_file(loops_header_templ)

with open(processed_loops_h_fn, 'w') as fid:
    fid.write(loops_header_processed)

loops_src_templ = join(wdir, "mkl_umath_loops.c.src")
processed_loops_src_fn = join(wdir, "mkl_umath_loops.c")
loops_src_processed = process_c_file(loops_src_templ)

with open(processed_loops_src_fn, 'w') as fid:
    fid.write(loops_src_processed)

skbuild.setup(
    name="mkl_umath",
    version=VERSION,
    ## cmdclass=_get_cmdclass(),
    description = "MKL-based universal functions for NumPy arrays",
    long_description = """Universal functions for real and complex floating point arrays powered by Intel(R) Math Kernel Library Vector (Intel(R) MKL) and Intel(R) Short Vector Math Library (Intel(R) SVML)""",
    long_description_content_type="text/markdown",
    license = 'BSD',
    author="Intel Corporation",
    url="http://github.com/IntelPython/mkl_umath",
    packages=[
        "mkl_umath",
    ],
    package_data={"mkl_umath": ["tests/*.*", "tests/helper/*.py"]},
    include_package_data=True,
    zip_safe=False,
    setup_requires=["Cython"],
    install_requires=[
        "numpy",
    ],
    extras_require={
        "docs": [
            "Cython",
            "sphinx",
            "sphinx_rtd_theme",
            "pydot",
            "graphviz",
            "sphinxcontrib-programoutput",
        ],
        "coverage": ["Cython", "pytest", "pytest-cov", "coverage", "tomli"],
    },
    keywords="mkl_umath",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    platforms=["Linux", "Windows"]
)
