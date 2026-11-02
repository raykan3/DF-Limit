# The C engine in MATLAB

`ltratio_mex` is a MEX file built from the C engine in `c/` (a port of `julia/DFExact.jl`, checked against its values
on every push).  It needs nothing else: no runtime, no library, 2.4 MB.  Two ways to use it:

- **Self-contained**: `ltratio.m` offers the functions of the MATLAB programs in `public/` with the same arguments.
- **Under the MATLAB programs**: `dfx_core.m`, `tr_limdist.m` and `tr_limmom.m` have the names and signatures of the
  engines in `public/`; with this folder ahead of `public/` on the path, `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`,
  `ltratio_mom`, `ltratio_paper` and `ltratio_test` run unchanged on the C engine (`ltratio_test` passes all 81
  checks in 10 s instead of about ten minutes).

```matlab
addpath ltratio-macos                                 % the unpacked zip (see below)
ltratio('icdf', [0.01 0.05 0.10], 'DU')               % asymptotic critical values of tau_mu
ltratio('cdf', -2.86157, 'DU', -5)                    % limiting power of the 5% test at c = -5
ltratio('cdf', -2, 'DU', -5, struct('g0', 1.5))       % fixed start
ltratio('pdf', [-1 0 1], 'DK', 2, struct('c0', 2))    % t-ratio centered at the true value
ltratio('mom', 4, 'SU', -5)                           % moments
ltratio('threads')                                    % the number of threads (all processors; set with a second argument)
```

`opts.method` defaults to `'auto'`: the single integral at c0 = 0 and with a fixed start g0 (the noncentral kernel)
and the double integral when c0 is not 0 with a zero start, the faster representation in each case.  The MEX file itself takes numeric arguments only (see the header of
`ltratio_mex.c`): `'single'`, `'double'`, `'dist'`, `'icdf'`, `'mom'`, `'tstar'`, `'threads'`.

Seconds per value of the distribution function, MATLAB R2026b on an M4 laptop (ten cores), `timeit`:

| case c c0 x | MATLAB programs (1 thread) single / double | C MEX 1 thread single / double | C MEX 10 threads single / double |
|---|---|---|---|
| DK 0 0 -1.94 | 0.041 / 0.058 | 0.009 / 0.0017 | 0.0020 / 0.0009 |
| DU 0 0 -2.86 | 0.107 / 0.137 | 0.019 / 0.020 | 0.0029 / 0.0095 |
| DT 0 0 -3.41 | 0.064 / 0.289 | 0.011 / 0.024 | 0.0026 / 0.011 |
| DU -5 0 -2.86 | 0.069 / 0.075 | 0.013 / 0.014 | 0.0028 / 0.0087 |
| DU 10 0 20 | 0.165 / 0.293 | 0.028 / 0.096 | 0.0061 / 0.023 |
| DU 0 1 -1 | 0.312 / 0.087 | 0.040 / 0.021 | 0.0086 / 0.0093 |
| DT -2.5 -5 -1.1 | 0.345 / 0.214 | 0.039 / 0.017 | 0.0084 / 0.0077 |
| DK 2 2 0 | 0.018 / 0.214 | 0.0005 / 0.016 | 0.0002 / 0.0036 |
| DU 5 -5 126.85 | 1.394 / 0.364 | 0.395 / 0.194 | 0.077 / 0.043 |
| SU -5 0 -2 | 0.088 / 0.546 | 0.016 / 0.039 | 0.0036 / 0.016 |
| ST -10 0 -3 | 0.072 / 0.678 | 0.013 / 0.041 | 0.0029 / 0.017 |

## Downloading a built MEX file

The workflow "MATLAB MEX" (`.github/workflows/mex.yml`) builds, on every push to `main` that touches `c/`, one zip
per platform (`ltratio-linux-x86_64`, `ltratio-macos` with the Apple silicon and the Intel MEX files,
`ltratio-windows-x86_64`) and attaches them as artifacts to the run (Actions tab; kept for 30 days); a tag `v*`
attaches them to a GitHub release.  Each zip unpacks to a folder with the MEX file and the m-files above.  Add the
folder to the MATLAB path.  No MATLAB is installed on the runners: the MEX file is compiled against the stub
`mexstub.h`, which declares the few functions of the MEX API it uses by their versioned symbol names (the "800"
API of R2018a and later), and on Windows is linked against import libraries generated from `win/libmx.def` and
`win/libmex.def`.

## Building locally

From the repository root, on macOS (Linux: `c++ -shared`, extension `mexa64`; add `-fPIC`):

```sh
mkdir -p build
for f in dfl_special dfl_mgf dfl_single dfl_moments dfl_double dfl_api dfl_threads tables; do cc -std=c11 -O2 -c c/$f.c -o build/$f.o; done
cc -std=c11 -O3 -c c/dfl_gen.c -o build/dfl_gen.o    # -O3: the l-rule of the fixed start is written to be vectorized
c++ -std=c++11 -O2 -c c/faddeeva/Faddeeva.cc -o build/Faddeeva.o
c++ -std=c++11 -O2 -c c/faddeeva/faddeeva_shim.cpp -o build/faddeeva_shim.o
cc -std=c11 -O2 -DDFLIMIT_STUB -c c/mex/ltratio_mex.c -o build/ltratio_mex.o
c++ -bundle -undefined dynamic_lookup -o ltratio_mex.mexmaca64 build/*.o
```

On Windows, with the Visual Studio tools: `lib /def:c\mex\win\libmx.def /machine:x64 /out:libmx.lib` (and
`libmex.def`), then `cl /LD /O2 /std:c11 /DDFLIMIT_STUB /Fe:ltratio_mex.mexw64 c\mex\ltratio_mex.c c\*.c
c\faddeeva\Faddeeva.cc c\faddeeva\faddeeva_shim.cpp libmx.lib libmex.lib` (leave out the `dfl_test_*.c` files).
With a MATLAB installation, compiling against its headers instead (drop `-DDFLIMIT_STUB`, add
`-I$MATLAB/extern/include`) checks that the stub still matches.  The engine's own tests (`c/dfl_test_*.c` against
the Julia reference values in `c/test/`, written by `c/test/*_ref.jl`) are run by the workflow "C engine"
(`c.yml`); `-DDFL_STRICT_COMPLEX` compiles the engine with complex numbers as a struct, which catches any use of
the complex arithmetic operators that MSVC lacks.  `c/tables.c` (the K_0 cells and the series) is generated by
`c/gen_tables.jl` from the Julia engine.
