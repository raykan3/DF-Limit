#!/bin/sh
# builds the C engine and its tests (unix); CFLAGS as in .github/workflows/c.yml
set -e
cd "$(dirname "$0")"
CFLAGS="-std=c11 -O2 -Wall -Wno-unused-function -Wno-logical-op-parentheses ${EXTRA_CFLAGS:-}"
for f in dfl_special dfl_mgf dfl_single dfl_moments dfl_double dfl_api dfl_threads tables; do cc $CFLAGS -c $f.c -o $f.o; done
cc $CFLAGS -O3 -c dfl_gen.c -o dfl_gen.o            # -O3: the l-rule of the noncentral kernel is written to be vectorized
c++ -std=c++11 -O2 -c faddeeva/Faddeeva.cc -o Faddeeva.o
c++ -std=c++11 -O2 -c faddeeva/faddeeva_shim.cpp -o faddeeva_shim.o
OBJ="dfl_special.o dfl_mgf.o dfl_single.o dfl_gen.o dfl_moments.o dfl_double.o dfl_api.o dfl_threads.o tables.o Faddeeva.o faddeeva_shim.o"
for t in ${TESTS:-prim single gen fixed vmath mom dbl}; do cc $CFLAGS -c dfl_test_$t.c -o dfl_test_$t.o; c++ -o dfl_test_$t dfl_test_$t.o $OBJ -lm -lpthread; done
