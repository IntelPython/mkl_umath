# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icx_for_conda.cfg
export ICXCFG="$(pwd)/icx_for_conda.cfg"

export CMAKE_GENERATOR="Ninja"
SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    # Install packages and assemble wheel package from built bits
    if [ "$CONDA_PY" == "36" ]; then
        WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
    else
        WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
    fi
    ${PYTHON} setup.py install bdist_wheel ${WHEELS_BUILD_ARGS} ${SKBUILD_ARGS}
    cp dist/mkl_umath*.whl ${WHEELS_OUTPUT_FOLDER}
else
    # Perform regular install
    ${PYTHON} setup.py install ${SKBUILD_ARGS}
fi
