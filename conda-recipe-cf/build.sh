#!/bin/bash
set -e

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icx_for_conda.cfg
ICXCFG="$(pwd)/icx_for_conda.cfg"
export ICXCFG

read -r GLIBC_MAJOR GLIBC_MINOR <<< "$(conda list '^sysroot_linux-64$' \
    | tail -n 1 | awk '{print $2}' | grep -oP '\d+' | head -n 2 | tr '\n' ' ')"

if [-d "build"]; then
    rm -rf build
fi

export CC=icx
export CXX=icpx

${PYTHON} -m build -w -n -x

${PYTHON} -m wheel tags --remove \
    --platform "manylinux_${GLIBC_MAJOR}_${GLIBC_MINOR}_x86_64" \
    dist/mkl_umath*.whl

${PYTHON} -m pip install dist/mkl_umath*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix "${PREFIX}"
    -vv

if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    mkdir -p "${WHEELS_OUTPUT_FOLDER}"
    cp dist/mkl_umath*.whl "${WHEELS_OUTPUT_FOLDER}"
fi
