::::::::::::::::::::::::::::::::::::::::::::::: ENVIRONMENT VARIABLE DEFINITIONS

set BOOST_INCLUDE_PATH=C:\Programs\boost_1_69_0
set EIGEN_INCLUDE_PATH=C:\Programs\eigen-eigen-323c052e1731
set DZNL_INCLUDE_PATH=C:\Users\Zhang\Documents\GitHub\dznl

set MSVC_ENV_SCRIPT_PATH=C:\Program Files (x86)\Microsoft Visual Studio\^
2017\Community\VC\Auxiliary\Build\vcvars64.bat

set ICC_ENV_SCRIPT_PATH=C:\Program Files (x86)\IntelSWTools\^
compilers_and_libraries_2018.3.210\windows\bin\ipsxe-comp-vars.bat

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if not exist "bin" mkdir "bin"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: GCC

g++ -std=c++17 -Wall -Wextra -pedantic ^
-O3 -march=native -fwhole-program ^
-isystem"%BOOST_INCLUDE_PATH%" -isystem"%EIGEN_INCLUDE_PATH%" ^
-I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-gcc.exe"

g++ -std=c++17 -Wall -Wextra -pedantic ^
-Ofast -march=native -fwhole-program ^
-isystem"%BOOST_INCLUDE_PATH%" -isystem"%EIGEN_INCLUDE_PATH%" ^
-I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-gcc-fast.exe"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: CLANG

clang++ -std=c++17 -Wall -Wextra -pedantic ^
-O3 -march=native ^
-isystem"%BOOST_INCLUDE_PATH%" -isystem"%EIGEN_INCLUDE_PATH%" ^
-I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-clang.exe"

clang++ -std=c++17 -Wall -Wextra -pedantic ^
-Ofast -march=native ^
-isystem"%BOOST_INCLUDE_PATH%" -isystem"%EIGEN_INCLUDE_PATH%" ^
-I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp -o"bin/rksearch-clang-fast.exe"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: MSVC

REM call "%MSVC_ENV_SCRIPT_PATH%"

REM del *.obj

REM cl /std:c++17 /EHsc /O2 /GL /favor:blend ^
REM /I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
REM rksearch-main.cpp /Fe"bin/rksearch-msvc.exe" /MT

REM del *.obj

REM cl /std:c++17 /EHsc /O2 /GL /favor:blend /fp:fast ^
REM /I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
REM rksearch-main.cpp /Fe"bin/rksearch-msvc-fast.exe" /MT

REM del *.obj

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ICC

REM if exist "%ICC_ENV_SCRIPT_PATH%" (

REM call "%ICC_ENV_SCRIPT_PATH%" intel64 vs2017

REM del *.obj

REM icl /Qstd=c++17 /O3 /Qipo /QxHost ^
REM /I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
REM rksearch-main.cpp /Fe"bin/rksearch-icc.exe" /MT /EHsc

REM del *.obj

REM icl /Qstd=c++17 /fast ^
REM /I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
REM rksearch-main.cpp /Fe"bin/rksearch-icc-fast.exe" /MT /EHsc

REM del *.obj

REM )
