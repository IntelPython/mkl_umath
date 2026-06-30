REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

set "CC=icx"
set "CXX=icx"

%PYTHON% -m pip install --no-build-isolation --no-deps .
if errorlevel 1 exit 1
