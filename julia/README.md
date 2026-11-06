# The Julia engine

`DFExact.jl` computes the limiting distribution function, density, quantiles and moments of the three Dickey-Fuller
t-ratios under local alternatives, for the statistic centered at the unit root or at a local null value c0, with a
zero, a fixed or a stationary start, by both representations of the paper (the line integral of Theorem 2, the
conditional integral of Theorem 3).  The user-level functions take the arguments of the MATLAB programs in `public/`:

```julia
include("julia/DFExact.jl"); using .DFExact        # from the repository root; start Julia with -t auto
ltratio_icdf([0.01, 0.05, 0.10], "DU")             # asymptotic critical values of tau_mu
ltratio_cdf(-2.86157, "DU", -5)                    # limiting power of the 5% test at c = -5
ltratio_cdf(-2.86157, "SU", -5)                    # the same with a stationary start
ltratio_cdf(0, "DK", 2; c0 = 2)                    # t-ratio centered at the true value
ltratio_cdf(-2, "DU", -5; g0 = 1.5)                # fixed start
ltratio_pdf(-2.0, "DT", -5)                        # density
ltratio_mom(4, "DT")                               # first four moments of tau_tau
```

The case `"DG"` is the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996) at cbar = -13.5 (the DF-GLS
tests, Section 9 of the paper; zero start, line integral only, Julia only).  `x` may be an array; `method = :auto` (the default) takes the line integral at c0 = 0 and with a fixed start (the
noncentral kernel), and when c0 != 0 with a zero start the conditional integral, except for c0 > 0 at x >= 0, where the
line integral has no breakpoint and is the faster one; it recomputes the values of the line
integral below 1e-3 by the conditional integral (the line integral's error is absolute, the conditional integral's
relative), returns the line integral's value at a point where the conditional integral reports that its error estimate
is not met, and returns a computed density below zero as zero; `:single` and `:double` force one representation.  The engine functions take the model number 1, 2, 3 for K, U, T:
`DFExact.cdf(q, model; c, c0, start = :zero | :stationary)`, `DFExact.density`, `DFExact.quantile`,
`DFExact.moments(model; c, c0, g0, start)`, `DFExact.tstar(model, c)`, `DFExact.tr_limdist(x, model, c; ...)`.
Install the dependencies once with `julia --project=julia -e 'using Pkg; Pkg.instantiate()'`.

Checks: `julia -t auto --project=julia julia/quickcheck.jl` (10 seconds; run by CI on three platforms),
`julia -t auto --project=julia julia/test_dfexact.jl` (stored reference values),
`julia -t auto --project=julia julia/certify_dfexact.jl` (node doubling, contour independence and the uniform rule
over c from -20 to 10; 15 minutes).  `julia/abadir/` compares case DK at c = 0 with the formulas of Abadir (1995):
`python3 abadir_series.py` (mpmath) then `julia -t auto --project=.. abadir_check.jl`.  The C engine in `c/` is a
port of this code and is checked against its values (`c/test/`).
