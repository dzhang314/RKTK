# suppressed remarks:
# 981:  evaluation of operands in unspecified order
# 1572: unreliability of floating-point equality
# 2547: dual specification as both system and non-system include directory

icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -Wall -Werror -w3 -wd981,1572,2547 -c src/linalg_subroutines.cpp -o obj/linalg_subroutines.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -Wall -Werror -w3 -wd981,1572,2547 -c src/objective_function.cpp -o obj/objective_function.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -Wall -Werror -w3 -wd981,1572,2547 -c src/bfgs_subroutines.cpp -o obj/bfgs_subroutines.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -Wall -Werror -w3 -wd981,1572,2547 src/rksearch_main.cpp obj/*.o -o bin/rksearch-intel -lmpfr -lgmp
rm obj/*.o

icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -march=skylake-avx512 -mtune=skylake-avx512 -Wall -Werror -w3 -wd981,1572,2547 -c src/linalg_subroutines.cpp -o obj/linalg_subroutines.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -march=skylake-avx512 -mtune=skylake-avx512 -Wall -Werror -w3 -wd981,1572,2547 -c src/objective_function.cpp -o obj/objective_function.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -march=skylake-avx512 -mtune=skylake-avx512 -Wall -Werror -w3 -wd981,1572,2547 -c src/bfgs_subroutines.cpp -o obj/bfgs_subroutines.o
icpc -std=c++11 -fast -use-intel-optimized-headers -no-inline-max-size -no-inline-max-total-size -no-inline-min-size -march=skylake-avx512 -mtune=skylake-avx512 -Wall -Werror -w3 -wd981,1572,2547 src/rksearch_main.cpp obj/*.o -o bin/rksearch-intel-skylake-avx512 -lmpfr -lgmp
rm obj/*.o
