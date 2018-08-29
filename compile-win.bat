::::::::::::::::::::::::::::::::::::::::::::::: ENVIRONMENT VARIABLE DEFINITIONS

set EIGEN_INCLUDE_PATH=C:\Programs\eigen-eigen-b3f3d4950030
set DZNL_INCLUDE_PATH=C:\Users\Zhang\Documents\GitHub\dznl

set WIN_MPFR_INCLUDE_PATH=C:\Programs\WinMPFR\include

set WIN_MPFR_LIBRARIES=^
C:\Programs\WinMPFR\lib\libmpfr.a ^
C:\Programs\WinMPFR\lib\libgmp.a ^
C:\Programs\msys64\mingw64\lib\gcc\x86_64-w64-mingw32\8.2.0\libgcc.a ^
C:\Programs\msys64\mingw64\lib\gcc\x86_64-w64-mingw32\8.2.0\libgcc_s.a

set MSVC_ENV_SCRIPT_PATH=C:\Program Files (x86)\Microsoft Visual Studio\^
2017\Community\VC\Auxiliary\Build\vcvars64.bat

set ICC_ENV_SCRIPT_PATH=C:\Program Files (x86)\IntelSWTools\^
compilers_and_libraries_2018.3.210\windows\bin\ipsxe-comp-vars.bat

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: set MINGW_LINK_FLAGS=-lmpfr -lgmp
set WIN_LINK_FLAGS=/MT
:: /link %WIN_MPFR_LIBRARIES% /ignore:4049,4217

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if not exist "bin" mkdir "bin"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: GCC

g++ -std=c++17 -Wall -Wextra -pedantic ^
-O3 -march=native -fwhole-program ^
-isystem"%EIGEN_INCLUDE_PATH%" -I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-gcc.exe"
:: %MINGW_LINK_FLAGS%

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: CLANG

clang++ -std=c++17 -Wall -Wextra -pedantic ^
-O3 -march=native ^
-isystem"%EIGEN_INCLUDE_PATH%" -I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-clang.exe"
:: %MINGW_LINK_FLAGS%

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: MSVC

call "%MSVC_ENV_SCRIPT_PATH%"

cl /std:c++17 /EHsc /O2 /favor:blend ^
/I"%WIN_MPFR_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-msvc.exe" ^
%WIN_LINK_FLAGS%

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ICC

if exist "%ICC_ENV_SCRIPT_PATH%" (

call "%ICC_ENV_SCRIPT_PATH%" intel64 vs2017

icl /Qstd=c++17 /O3 /QxHost ^
/I"%WIN_MPFR_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-icc.exe" ^
%WIN_LINK_FLAGS%

)
