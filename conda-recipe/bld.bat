REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

set "CC=icx"
set "CXX=icx"

%PYTHON% -m build -w -n -x
if %ERRORLEVEL% neq 0 exit 1

for /f %%f in ('dir /b /S .\dist') do (
    %PYTHON% -m pip install %%f ^
        --no-build-isolation ^
        --no-deps ^
        --only-binary :all: ^
        --no-index ^
        --prefix %PREFIX% ^
        -vv
    if %ERRORLEVEL% neq 0 exit 1
)

if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    copy dist\mkl_umath*.whl %WHEELS_OUTPUT_FOLDER%
    if %ERRORLEVEL% neq 0 exit 1
)
