# Copyright (c) 2019-2025, Intel Corporation
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
import sys
from setuptools.modified import newer
from os import makedirs
from os.path import join, exists, dirname

import skbuild

sys.path.insert(0, dirname(__file__))  # Ensures local imports work
from _vendored.conv_template import process_file as process_c_file


# TODO: rewrite generation in CMake, see NumPy meson implementation
# https://github.com/numpy/numpy/blob/c6fb3357541fd8cf6e4faeaeda3b1a9065da0520/numpy/_core/meson.build#L623
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
    packages=[
        "mkl_umath",
    ],
    package_data={"mkl_umath": ["tests/*.*"]},
    include_package_data=True,
)
