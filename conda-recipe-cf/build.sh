#!/bin/bash -x

CC=icx CXX=icpx $PYTHON -m pip install --no-build-isolation --no-deps -Csetup-args="-Dmkl_threading=gnu_thread" .
