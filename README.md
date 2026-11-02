# DF Limit

Bao, Broda and Kan, "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio".

The limiting distribution function, density, quantiles and moments of the Dickey-Fuller t-ratios tau, tau_mu and tau_tau under local alternatives phi = 1 + c/n, for the statistic centered at the unit root or at any local null value phi_0 = 1 + c_0/n, with a zero, a fixed or a stationary initial observation of the order of sqrt(n).

- `LaTeX/` — the paper: `LTR_BBK.tex`, `LTR_BBK.bib`, and the compiled `LTR_BBK.pdf`.
- `public/` — the MATLAB programs (start with `Contents.m` and the manual `doc/ltratio_manual.pdf`): the user-level routines `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, the two engines (the single integral `dfx_*`, the double integral `tr_*`), the validation suite `ltratio_test` and `ltratio_paper`, which reproduces the tables; `public/validation/` holds the script behind the accuracy table and the logs.
- `julia/` — the Julia engine `DFExact.jl`, both representations with the arguments of the MATLAB programs (see `julia/README.md`).
- `c/` — a C port of the engine, checked against the Julia values on every push, and in `c/mex/` the MATLAB MEX file built from it with drop-in replacements of the MATLAB engines, 20 to 50 times faster than the MATLAB programs; the workflow "MATLAB MEX" builds it for Linux, macOS and Windows (see `c/mex/README.md`).
- `demo/` — an interactive web page for the limits (all six cases, the local null value and the fixed start): open `demo/ltr_demo.html` in a browser. It is built from `ltr_demo_template.html`, `ltrlim.js` (the JavaScript port of the engines) and `ntcoef.js` by `python make_demo.py`.

```matlab
ltratio_icdf([0.01 0.05 0.10], 'DU')        % asymptotic critical values of tau_mu
ltratio_cdf(-2.86157, 'DU', -5)              % limiting power of the 5% test at c = -5
ltratio_cdf(-2.86157, 'SU', -5)              % the same with a stationary start
ltratio_cdf(0, 'DK', 2, struct('c0', 2))     % t-ratio centered at the true value
ltratio_mom(4, 'DT')                         % first four moments of tau_tau
```

## The MEX file: the C engine in MATLAB

The MATLAB programs in `public/` take 0.05 to 0.5 seconds per value. The MEX file `ltratio_mex`, built from the C engine, gives the same values 20 to 50 times faster and needs nothing else: no compiler, no runtime, one file of 2.5 MB.

1. Download the zip for your platform: from the [latest release](https://github.com/raykan3/DF-Limit/releases) (`ltratio-windows-x86_64.zip`, `ltratio-macos.zip` with the Apple silicon and the Intel MEX files, `ltratio-linux-x86_64.zip`), or the artifact of the latest run of the workflow ["MATLAB MEX"](https://github.com/raykan3/DF-Limit/actions/workflows/mex.yml) (Actions tab; every push to `main` that touches `c/` rebuilds them).
2. Unzip, and in MATLAB add the folder to the path: `addpath ltratio-windows-x86_64`.
3. Call it, with the arguments of `ltratio_cdf` and the others:

```matlab
ltratio('cdf', -2.86157, 'DU', -5)                  % P[tau_mu <= x] at c = -5
ltratio('pdf', [-3 -2 -1], 'DT', 0)                  % density
ltratio('icdf', [0.01 0.05 0.10], 'DU')              % quantiles
ltratio('mom', 4, 'SU', -5)                          % moments 1..4
ltratio('cdf', 0, 'DK', 2, struct('c0', 2))          % local null value
ltratio('cdf', -2, 'DU', -5, struct('g0', 1.5))      % fixed start
ltratio('threads')                                   % threads in use (all cores; ltratio('threads', 1) for one)
```

The folder also contains `dfx_core.m`, `tr_limdist.m` and `tr_limmom.m`, replacements of the engines in `public/`: with the folder ahead of `public/` on the path (`addpath` puts it first), `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, `ltratio_paper` and `ltratio_test` run unchanged on the C engine; `ltratio_test` then takes 10 seconds. Details, timings and how to build the MEX file yourself: `c/mex/README.md`.

To compile the paper: `pdflatex LTR_BBK; bibtex LTR_BBK; pdflatex LTR_BBK; pdflatex LTR_BBK` in `LaTeX/`.
To check the programs: `ltratio_test` in MATLAB from `public/` (about two minutes).

The companion paper and programs for the exact finite-sample distribution are in the repository DF-Test.
