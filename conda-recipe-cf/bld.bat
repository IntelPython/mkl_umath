REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

set "CMAKE_GENERATOR=Ninja"
set "CMAKE_ARGS=-DCMAKE_C_COMPILER:PATH=icx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

%PYTHON% -m pip install --no-build-isolation --no-deps . --verbose
if errorlevel 1 exit 1
