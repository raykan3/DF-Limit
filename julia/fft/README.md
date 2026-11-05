# The conditional integral by the FFT (experiment, branch fft, 5 November 2026)

Question (Simon): how much would the exact limiting distribution speed up if one or more of its inversions were done
by the FFT, Carr-Madan style?

## Where an FFT can act

The line integral (Theorem 2) does the s-inversion in closed form and integrates once over t on a line that is the
same for every x; there is nothing left to transform.  The s-inversion itself is unsuitable for an FFT in any case:
|M(s, t)| decays only like |Im s|^{-1/2} (DK) or |Im s|^{-1} (DU, DT), the density of U having an inverse-square-root
or logarithmic singularity, so a truncated Fourier sum in s converges algebraically; the closed form is exact and free.

The conditional integral (Theorem 3) is the representation with a per-v inner inversion in t, a new saddle path for
every v.  With the line kept fixed, H(v; x) = (2 pi i)^{-1} int P(t) Psi_t(a(v)) e^{-tv} dt is, for a fixed value of
a, a Laplace inversion in v, which one FFT evaluates on a uniform v-grid for all v at once.  The dependence on x and
on c0 is entirely through the curve a(v) = 1/2 + x sqrt(v) + c0 v, so a table of H~(v, a) on a grid in (v, a) serves
every x and every c0 at the given (case, c).

## What was done (case DK, zero start)

`fft_cond.jl`: Psi_t(a) = erf(sqrt(s1 a))/sqrt(s1) = sqrt(a) E_t(a) with E_t entire in a; for each of na + 1
Chebyshev-Lobatto nodes a_j in [0, amax] one FFT of length N of g_j(t) = P(t) E_t(a_j) on the line Re t = sigma
(sigma = 0 serves; the integrand is analytic for Re t < (c^2 + pi^2)/2, so the trapezoidal rule in omega converges
geometrically) gives E(v_k, a_j) on the uniform v-grid with dv = 2 pi/(N dw); then, per x, F(x) = int sqrt(a(v))
E(v, a(v)) dv by composite Gauss-Legendre in a variable that removes the square-root onset at the support edge
(a = p^2 for x < 0, w = sqrt(v) for x >= 0), with E interpolated by a 6-point Lagrange rule in v and the barycentric
Chebyshev formula in a.  Check of the table: int H~(v, a) dv = P(R <= a) = P(0) erf(sqrt(s1(0) a))/sqrt(s1(0)) to
1e-15.  P(t) and s1(t) along the line come from the engine (`DFExact.tparts`).

`run_dk.jl c [N na dw sigma]` compares with `DFExact.cdf` on x in [-4, 4]; `sweep_dk.jl`, `sweep2_dk.jl`,
`sweep3_dk.jl` vary the parameters; `time_dk.jl` times 1001 points against the engine; `engine_timing.jl` times the
engine alone.  Run with `julia -t 10 --project=julia/fft julia/fft/run_dk.jl 0.0` from the repository root (the
environment here adds FFTW to the engine's).

## Results (M4 laptop, 10 threads, Julia; engine = the line integral in DFExact.jl, warm)

| setting (N = 2^16, dw = 0.25, v up to 25, na = 96) | max abs error on x in [-4, 4] |
|---|---|
| c = 0 | 3e-12 (1e-12 with na = 160) |
| c = 1, c = 2 | 2e-12, 1e-12 |
| c = -5 | 1e-11 |
| c = -20, amax = 3 | 2e-9; 5e-11 with N = 2^17 (the decay in omega is slower at large |c|) |
| c = 0, dw = 0.5 / 1.0 | 2e-8 / 9e-5 (the integrand grows in the strip with a) |
| c = 0, na = 48 / 64 | 1e-6 / 2e-8 |
| c = 0, N = 2^15 (v up to 12.6) | 7e-11 (aliasing of the tail of V) |

Costs: the table (97 FFTs of length 65536 and as many evaluations of the complex error function at the nodes) takes
0.08 s on ten threads once FFTW's plan exists (the first plan in a process, measured, costs about 0.5 s); one value
of F then costs 0.19 ms (10 panels x 64 nodes), 0.09 ms (6 x 48, 2e-11) or 0.06 ms (na = 64, 2e-8).  The engine's
line integral takes 0.47 ms per value on a 1001-point grid (0.6 ms per fresh value; the layouts are cached per x),
the double integral 1.5 ms.  So, per (case, c): one value, the FFT route is slower by the table; a grid of about
200 values breaks even; 1001 values take 0.27 s against 0.47-0.64 s, a factor of 2-3.  Against the C engine's 2-9
ms per value (no batched layouts) the factor on a dense grid would be 10-40.

## What is not done, and why

Cases DU and DT: the inner kernel is Theorem 3(i)'s P(kappa_1, kappa_2; a), itself an l-integral (about 40 nodes
with a complex erf each, so a table costs 2-3 s on ten threads), or part (ii)'s K_0 density with G and the 1/t pole;
and R takes both signs, so the a-grid must cross a = 0, where H~(v, a) behaves like a log|a| (the logarithmic
singularity of the density of R): a Chebyshev interpolant across it converges algebraically, so the singular part
would have to be removed analytically or the grid graded on both sides.  A fixed start replaces Psi by the
noncentral probability (Pnc) of the paper, a table per (case, c, g0).  None of this changes the conclusion: the
FFT buys a whole curve in x (and all c0) for the price of a few hundred engine values, at 1e-11 rather than 1e-14.
