::::::::::::::::::::::::::::::::::::::::::::::: ENVIRONMENT VARIABLE DEFINITIONS

set BOOST_INCLUDE_PATH=C:\Programs\boost_1_68_0
set EIGEN_INCLUDE_PATH=C:\Programs\eigen-eigen-b3f3d4950030
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

call "%MSVC_ENV_SCRIPT_PATH%"

del *.obj

cl /std:c++17 /EHsc /O2 /GL /favor:blend ^
/I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-msvc.exe" /MT

del *.obj

cl /std:c++17 /EHsc /O2 /GL /favor:blend /fp:fast ^
/I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-msvc-fast.exe" /MT

del *.obj

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ICC

if exist "%ICC_ENV_SCRIPT_PATH%" (

call "%ICC_ENV_SCRIPT_PATH%" intel64 vs2017

icl /Qstd=c++17 /O3 /Qipo /QxHost ^
/I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-icc.exe" /MT

icl /Qstd=c++17 /fast ^
/I"%BOOST_INCLUDE_PATH%" /I"%EIGEN_INCLUDE_PATH%" /I"%DZNL_INCLUDE_PATH%" ^
rksearch-main.cpp /Fe"bin/rksearch-icc-fast.exe" /MT

)
