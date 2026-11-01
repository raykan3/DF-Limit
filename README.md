# DF Limit

Bao, Broda and Kan, "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio".

The limiting distribution function, density, quantiles and moments of the Dickey-Fuller t-ratios tau, tau_mu and tau_tau under local alternatives phi = 1 + c/n, for the statistic centered at the unit root or at any local null value phi_0 = 1 + c_0/n, with a zero, a fixed or a stationary initial observation of the order of sqrt(n).

- `LaTeX/` — the paper: `LTR_BBK.tex`, `LTR_BBK.bib`, and the compiled `LTR_BBK.pdf`.
- `public/` — the MATLAB programs (start with `Contents.m` and the manual `doc/ltratio_manual.pdf`): the user-level routines `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, the two engines (the single integral `dfx_*`, the double integral `tr_*`), the validation suite `ltratio_test` and `ltratio_paper`, which reproduces the tables; `public/validation/` holds the script behind the accuracy table and the logs.
- `demo/` — an interactive web page for the limits (all six cases, the local null value and the fixed start): open `demo/tr_demo.html` in a browser. It is built from `tr_demo_template.html`, the JavaScript port of the engines and `ntcoef.js` by `python make_demo.py`.

```matlab
ltratio_icdf([0.01 0.05 0.10], 'DU')        % asymptotic critical values of tau_mu
ltratio_cdf(-2.86157, 'DU', -5)              % limiting power of the 5% test at c = -5
ltratio_cdf(-2.86157, 'SU', -5)              % the same with a stationary start
ltratio_cdf(0, 'DK', 2, struct('c0', 2))     % t-ratio centered at the true value
ltratio_mom(4, 'DT')                         % first four moments of tau_tau
```

To compile the paper: `pdflatex LTR_BBK; bibtex LTR_BBK; pdflatex LTR_BBK; pdflatex LTR_BBK` in `LaTeX/`.
To check the programs: `ltratio_test` in MATLAB from `public/` (about two minutes).

The companion paper and programs for the exact finite-sample distribution are in the repository DF-Test.
