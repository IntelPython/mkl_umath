# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export BUILD_PREFIX=$CONDA_PREFIX
export HOST=x86_64-conda-linux-gnu
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

# if [ -e "_skbuild" ]; then
#   python setup.py clean --all
# fi

export CMAKE_GENERATOR="Ninja"
SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
echo "python setup.py install ${SKBUILD_ARGS}"
python setup.py install ${SKBUILD_ARGS}
