#!/usr/bin/env python
# Copyright (c) 2017-2020, Intel Corporation
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

import sys
from os import (getcwd, environ, makedirs)
from os.path import join, exists, abspath, dirname
import importlib.machinery # requires Python >= 3.4
from distutils.dep_util import newer

from numpy.distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
import platform
from numpy import get_include as get_numpy_include
from distutils.sysconfig import get_python_inc as get_python_include

def ensure_Intel_compiler():
    ccompiler = new_compiler()
    customize_compiler(ccompiler)
    if hasattr(ccompiler, 'compiler'):
        compiler_name = ccompiler.compiler[0]
    else:
        compiler_name = ccompiler.__class__.__name__

    assert ('icl' in compiler_name or 'icc' in compiler_name), \
        "Intel(R) C Compiler is required to build mkl_umath, found {}".format(compiler_name)
    

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


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('mkl_umath', parent_package, top_path)

    mkl_root = environ.get('MKLROOT', None)
    if mkl_root:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }
    else:
        mkl_info = get_info('mkl')

    print(mkl_info)
    mkl_include_dirs = mkl_info.get('include_dirs', [])
    mkl_library_dirs = mkl_info.get('library_dirs', [])
    mkl_libraries = mkl_info.get('libraries', ['mkl_rt'])

    pdir = dirname(__file__)
    wdir = join(pdir, 'src')
    mkl_info = get_info('mkl')

    generate_umath_py = join(pdir, 'generate_umath.py')
    n = separator_join('_', (config.name, 'generate_umath'))
    generate_umath = load_module(n, generate_umath_py)
    del n

    def generate_umath_c(ext, build_dir):
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
        config.add_include_dirs(target_dir)
        return []

    sources = [generate_umath_c]

    # ensure_Intel_compiler()

    if platform.system() == "Windows":
        eca = ['/fp:fast=2', '/Qimf-precision=high', '/Qprec-sqrt', '/Qstd=c99', '/Qprotect-parens']
    else:
        eca = ['-fp-model', 'fast=2', '-fimf-precision=high', '-prec-sqrt', '-fprotect-parens']

    numpy_include_dir = get_numpy_include()
    python_include_dir = get_python_include()
    config.add_library(
        'loops_intel',
        sources = [
            join(wdir, 'loops_intel.h.src'),
            join(wdir, 'loops_intel.c.src'),
        ],
        include_dirs = [wdir] + mkl_include_dirs + [numpy_include_dir, python_include_dir],
        depends = [
            join(wdir, 'blocking_utils.h'),
            join(wdir, 'fast_loop_macros.h'),
            join(numpy_include_dir, 'numpy', '*object.h'),
            join(python_include_dir, "Python.h")
        ],
        libraries=mkl_libraries,
        extra_compiler_args=eca,
        macros=getattr(config, 'define_macros', getattr(config.get_distribution(), 'define_macros', []))
    )

    config.add_extension(
        name = '_ufuncs',
        sources = [
            join(wdir, 'ufuncsmodule.c'),
        ] + sources,
        depends = [
            join(wdir, 'loops_intel.c.src'),
            join(wdir, 'loops_intel.h.src'),
        ],
        include_dirs = [wdir] + mkl_include_dirs,
        libraries = mkl_libraries + ['loops_intel'],
        library_dirs = mkl_library_dirs,
        extra_compile_args = [
            '-DNDEBUG',
            # '-ggdb', '-O0', '-Wall', '-Wextra', '-DDEBUG',
        ]
    )

    from Cython.Build import cythonize
    from setuptools import Extension
    cythonize(Extension('_patch', sources=[join(wdir, 'patch.pyx'),]))

    config.add_extension(
        name = '_patch',
        sources = [
            join(wdir, 'patch.c'),
        ],
        libraries = mkl_libraries + ['loops_intel'],
        library_dirs = mkl_library_dirs,
        extra_compile_args = [
            '-DNDEBUG',
            #'-ggdb', '-O0', '-Wall', '-Wextra', '-DDEBUG',
        ]
    )

    config.add_data_dir('tests')

#    if have_cython:
#        config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
