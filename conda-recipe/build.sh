#!/bin/bash -ex

# -Ccompile-args=-v makes ninja print full compiler commands (verbose build)
CC=icx CXX=icpx $PYTHON -m pip install --no-build-isolation --no-deps \
    -Csetup-args="-Dmkl_threading=gnu_thread" \
    -Ccompile-args=-v \
    . \
    -vv
