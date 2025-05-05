# Copyright (c) 2025, Intel Corporation
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

# Adapted from 
# https://github.com/numpy/numpy/blob/maintenance/2.2.x/numpy/_core/code_generators/generate_umath_doc.py

import sys
import os
import textwrap
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    import ufunc_docstrings_numpy1 as docstrings
else:
    import ufunc_docstrings_numpy2 as docstrings
sys.path.pop(0)

def normalize_doc(docstring):
    docstring = textwrap.dedent(docstring).strip()
    docstring = docstring.encode('unicode-escape').decode('ascii')
    docstring = docstring.replace(r'"', r'\"')
    docstring = docstring.replace(r"'", r"\'")
    # Split the docstring because some compilers (like MS) do not like big
    # string literal in C code. We split at endlines because textwrap.wrap
    # do not play well with \n
    docstring = '\\n\"\"'.join(docstring.split(r"\n"))
    return docstring

def write_code(target):
    with open(target, 'w') as fid:
        fid.write(
            "#ifndef NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
            "#define NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
        )
        for place, string in docstrings.docdict.items():
            cdef_name = f"DOC_{place.upper().replace('.', '_')}"
            cdef_str = normalize_doc(string)
            fid.write(f"#define {cdef_name} \"{cdef_str}\"\n")
        fid.write("#endif //NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to the output directory"
    )
    args = parser.parse_args()

    outfile = os.path.join(os.getcwd(), args.outfile)
    write_code(outfile)


if __name__ == '__main__':
    main()
