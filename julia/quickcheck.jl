# Quick check of DFExact.jl (about a minute after compilation): the stored reference values, node doubling and
# contour independence at a few points of every model for c = 0, -5 and 3, the K_0 evaluation against AMOS,
# and the normalization of the density.  Exits with an error if anything is off; the full certification is
# certify_dfexact.jl.  Run with julia -t auto --project=julia julia/quickcheck.jl
using Printf, SpecialFunctions, Random
include(joinpath(@__DIR__, "DFExact.jl")); using .DFExact; const D = DFExact
@printf("Julia %s, %s, %d threads, %d workers, vector math: %s\n", VERSION, Sys.MACHINE, Threads.nthreads(), D.NWORKERS[], D.vecmath())
fails = String[]
check(ok, msg) = (ok || push!(fails, msg); @printf("  %s %s\n", ok ? "ok  " : "FAIL", msg))

# 1. K_0 against AMOS
Random.seed!(1); worst = 0.0
for _ in 1:20000
    z = complex(rand() * 40, (rand() - 0.5) * 80)
    ref = besselkx(0, z); global worst = max(worst, abs(D.k0x(z) - ref) / abs(ref))
end
check(worst < 5e-14, @sprintf("K_0 vs AMOS: worst relative error %.1e", worst))

# 2. stored reference values (the twelve points of test_dfexact.jl)
import JSON
ref = JSON.parsefile(joinpath(@__DIR__, "python_exact_reference.json"))
wF = wf = 0.0
for r in ref
    F = D.cdf(Float64(r["q"]), r["model"]; c = Float64(r["c"])); f = D.density(Float64(r["q"]), r["model"]; c = Float64(r["c"]))
    global wF = max(wF, abs(F - r["cdf"])); global wf = max(wf, abs(f - r["density"]) / abs(r["density"]))
end
check(wF < 1e-12 && wf < 1e-10, @sprintf("reference values: |dF| %.1e, |df/f| %.1e", wF, wf))

# 3. node doubling and contour independence at a few points
wd = 0.0; wt = 0.0
for model in 1:3, (c, q) in ((0.0, -2.0), (0.0, 0.5), (-5.0, -3.0), (3.0, 2.0), (3.0, 8.0))
    F = D.cdf(q, model; c = c); f = D.density(q, model; c = c)
    D.NODEFAC[] = 2; F2 = D.cdf(q, model; c = c); f2 = D.density(q, model; c = c); D.NODEFAC[] = 1
    global wd = max(wd, abs(F - F2), abs(f - f2) / abs(f))
    if c > 0
        F3 = D.cdf(q, model; c = c, tau = D.tstar(model, c) / 4); global wt = max(wt, abs(F - F3))
    end
end
check(wd < 1e-10, @sprintf("node doubling: worst change %.1e", wd))
check(wt < 1e-10, @sprintf("contour independence (c = 3): worst change %.1e", wt))

# 4. normalization of the density (M2, c = 0) and the 5% quantile of M2
qs = range(-8, 6, length = 561); h = step(qs)
mass = sum(D.density(q, 2) for q in qs) * h
check(abs(mass - 1) < 1e-9, @sprintf("density of M2 integrates to %.12f", mass))
q05 = D.quantile(0.05, 2)
check(abs(q05 + 2.861568864) < 1e-8, @sprintf("M2 5%% quantile %.9f", q05))

# 5. the stationary start: the limiting power of the 5% test of M2 at c = -5 (0.125 in Bao, Broda and Kan's table)
Fs = D.cdf(-2.86157, 2; c = -5.0, start = :stationary)
D.NODEFAC[] = 2; Fs2 = D.cdf(-2.86157, 2; c = -5.0, start = :stationary); D.NODEFAC[] = 1
check(abs(Fs - 0.125) < 5e-4 && abs(Fs - Fs2) < 1e-10, @sprintf("stationary start, M2 c=-5 power %.6f (doubling %.1e)", Fs, abs(Fs - Fs2)))

# 6. small |q|: the fixed layouts hand over to the adaptive code below |q| = 0.034; both paths and the uniform rule agree
wq = 0.0
for (m, q) in ((1, -0.0061), (2, -0.0061), (1, -0.02), (3, -0.02))
    fb = D.density(q, m); D.BATCHED[] = false; fs = D.density(q, m); D.BATCHED[] = true
    fu = D.density_uniform(q, m; hw = 0.1)
    global wq = max(wq, abs(fb - fs) / fs, abs(fb - fu) / fs)
end
check(wq < 1e-10, @sprintf("small |q| (0.006, 0.02): batched vs adaptive vs uniform %.1e", wq))

# 7. large c: the roots and P(t) from the Girsanov forms (M2, c = 10, x = 20; the determinant itself loses 8 digits there)
F10 = D.cdf(20.0, 2; c = 10.0); D.NODEFAC[] = 2; F10b = D.cdf(20.0, 2; c = 10.0); D.NODEFAC[] = 1
check(abs(F10 - 0.001645968250) < 1e-11 && abs(F10 - F10b) < 1e-11, @sprintf("M2 c=10: F(20) = %.12f (doubling %.1e)", F10, abs(F10 - F10b)))
# the Girsanov determinant against D(0, t) where both are exact: e^c cosh(th) det(I - Q Sig) = D(0, t)
wg = 0.0
for m in 2:3, c in (2.5, -1.0), t in (complex(0.05), complex(0.3, 0.7), complex(2.0, -4.0), complex(c^2 / 2 + 0.2, 0.3))
    G = D.girsanov(m, c, t)
    global wg = max(wg, abs(exp(c) * cosh(G.th) * G.D / D.Dfun(m, c, complex(0.0), t) - 1))
end
check(wg < 1e-12, @sprintf("Girsanov determinant vs D(0, t): worst relative difference %.1e", wg))
# the domain end t_* of case DK at c < 0: the first zero of D(0, t) lies just below the pole of eta at (c^2 + pi^2)/2, which
# a bracketing on the sign of eta alone could skip (61.6 for 49.2 at c = -9.5, 313 for 204.5 at c = -20)
wts = maximum(abs(D.tstar(1, c) - D.dksmax(c)) / D.dksmax(c) for c in (-9.5, -20.0, -30.0))
check(wts < 1e-12 && abs(D.tstar(1, -9.5) - 49.18660310687472) < 1e-9, @sprintf("t_* of case DK at c = -9.5, -20, -30 vs dksmax: %.1e", wts))
# the GLS-detrended trend case (model 4, case DG): at c = 0, P(tau <= 0) = P{V(1)^2 <= 1} with Var V(1) = 0.2 (1 - lam)^2 in
# closed form; node doubling and the contour at the 5% point
lamg = D.lam_gls(D.CBAR_GLS); F0g = D.ltratio_cdf(0.0, "DG", 0.0); F0c = erf(1 / sqrt(0.4 * (1 - lamg)^2))
Fg = D.cdf(-2.8456, 4; c = -5.0); D.NODEFAC[] = 2; Fg2 = D.cdf(-2.8456, 4; c = -5.0); D.NODEFAC[] = 1; Fg3 = D.cdf(-2.8456, 4; c = -5.0, tau = 0.5)
check(abs(F0g - F0c) < 1e-12 && abs(Fg - Fg2) < 1e-11 && abs(Fg - Fg3) < 1e-11, @sprintf("case DG (GLS, trend): P(tau <= 0) at c = 0 vs the closed form %.1e; doubling %.1e, contour %.1e", abs(F0g - F0c), abs(Fg - Fg2), abs(Fg - Fg3)))

# 8. the moments: E tau_mu^2 at the unit root is 3.05599535 (Nabeya, 1999); the first implementation agrees
m2 = D.moments(2; pmax = 2)[2]
check(abs(m2 - 3.05599535) < 1e-8 && abs(m2 - D.moments_dfx(2; pmax = 2)[2]) < 1e-12, @sprintf("E tau_mu^2 at c = 0: %.8f", m2))

# 9. the local null value c0: the t-ratio centered at the true value (DK, c = c0 = 2, x = 0: 0.650436 in Larsson's table),
#    P(U - c0 V <= 0) at c = 10, c0 = 5 (DU: 0.000302655553129 by a one-dimensional inversion), node doubling with two
#    breakpoints (DT, c = -2.5, c0 = -5), and the general kernel at c0 = 0 against the specialized code
F22 = D.cdf(0.0, 1; c = 2.0, c0 = 2.0)
check(abs(F22 - 0.650436) < 1e-6, @sprintf("DK c = c0 = 2: F(0) = %.6f (0.650436)", F22))
F105 = D.cdf(0.0, 2; c = 10.0, c0 = 5.0)
check(abs(F105 - 0.000302655553129) < 1e-11, @sprintf("DU c = 10, c0 = 5: F(0) = %.15f", F105))
Fg = D.cdf(-1.1, 3; c = -2.5, c0 = -5.0); fg = D.density(-1.1, 3; c = -2.5, c0 = -5.0)
D.NODEFAC[] = 2; Fg2 = D.cdf(-1.1, 3; c = -2.5, c0 = -5.0); fg2 = D.density(-1.1, 3; c = -2.5, c0 = -5.0); D.NODEFAC[] = 1
check(abs(Fg - Fg2) < 1e-10 && abs(fg - fg2) / fg < 1e-10, @sprintf("DT c = -2.5, c0 = -5: F(-1.1) = %.12f, doubling %.1e / %.1e", Fg, abs(Fg - Fg2), abs(fg - fg2) / fg))
Fs = D.cdf(-2.86, 2); Fsg = D.cdf(-2.86, 2; gen = true); fs = D.density(-2.86, 2); fsg = D.density(-2.86, 2; gen = true)
check(abs(Fs - Fsg) < 1e-12 && abs(fs - fsg) / fs < 1e-11, @sprintf("general kernel at c0 = 0 vs the specialized code: %.1e / %.1e", abs(Fs - Fsg), abs(fs - fsg) / fs))
check(isinf(D.density(-2.0, 1; c0 = 2.0)), "DK c0 = 2: f(-sqrt(2 c0)) = Inf")

# 10. moments with a local null value and a start: the mean of DU at c = -5 against the Gauss-Legendre integral of the density
#     (192 nodes on [-12, 6]), and the stationary start as the Gauss-Hermite average over the start of the fixed-start moments
using LinearAlgebra
let n = 192, a = -12.0, b = 6.0
    E = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1]))
    xg = a .+ (b - a) .* (E.values .+ 1) ./ 2; wg = (b - a) .* E.vectors[1, :] .^ 2
    f = [D.density(x, 2; c = -5.0) for x in xg]
    m1 = D.moments(2; c = -5.0, pmax = 1)[1]
    check(abs(sum(wg .* xg .* f) - m1) < 1e-9 && abs(sum(wg .* f) - 1) < 1e-9, @sprintf("DU c=-5: mean %.10f vs the density integral (%.1e), mass 1 %+.1e", m1, abs(sum(wg .* xg .* f) - m1), sum(wg .* f) - 1))
    E = eigen(SymTridiagonal(zeros(16), sqrt.(1:15))); gz = E.values; gw = E.vectors[1, :] .^ 2
    mst = D.moments(2; c = -5.0, start = :stationary, pmax = 2)
    mav = sum(gw[k] .* D.moments(2; c = -5.0, g0 = sqrt(0.1) * gz[k], pmax = 2) for k in 1:16)
    check(maximum(abs.(mst .- mav)) < 1e-7, @sprintf("SU c=-5: moments %s vs the Gauss-Hermite average of the fixed start %.1e", mst, maximum(abs.(mst .- mav))))
end

# 11. the double integral (Theorem 3 of Raymond's draft; the only route for a fixed start in cases DU, DT): against the
#     single integral with a zero start (DU, c = 2, c0 = 1) and a stationary start (SU, c = -5), the closed form of
#     P[J(1)^2 <= 1 + g0^2] in case DK with a fixed start, and the MATLAB programs' value for DU, c = -5, g0 = 1.5, x = -2
Fd = ltratio_cdf([-0.5, 1.3], "DU", 2; c0 = 1, method = :double); Fs = ltratio_cdf([-0.5, 1.3], "DU", 2; c0 = 1)
check(maximum(abs.(Fd .- Fs)) < 1e-11, @sprintf("double vs single integral, DU c = 2 c0 = 1: %.1e", maximum(abs.(Fd .- Fs))))
Fd = ltratio_cdf(-2.0, "SU", -5; method = :double); Fs = ltratio_cdf(-2.0, "SU", -5)
check(abs(Fd - Fs) < 1e-11, @sprintf("double vs single integral, SU c = -5: %.1e", abs(Fd - Fs)))
let c = -5.0, g0 = 1.5, v = (exp(-10) - 1) / (-10), m = 1.5 * exp(-5), r = sqrt(1 + 1.5^2)
    Fex = (erf((r - m) / sqrt(2v)) + erf((r + m) / sqrt(2v))) / 2
    Fg = ltratio_cdf(0.0, "DK", c; g0)
    check(abs(Fg - Fex) < 1e-11, @sprintf("DK c = -5 g0 = 1.5: F(0) = %.15f vs the closed form (%.1e)", Fg, abs(Fg - Fex)))
end
Fg = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5)
check(abs(Fg - 0.864977240077167) < 1e-11, @sprintf("DU c = -5 g0 = 1.5: F(-2) = %.15f vs the MATLAB programs (%.1e)", Fg, abs(Fg - 0.864977240077167)))

# 12. the user-level interface: the 5% critical value of tau_mu, non-finite x, invalid arguments
q05 = ltratio_icdf(0.05, "DU")
check(abs(q05 + 2.86156886) < 1e-7, @sprintf("ltratio_icdf(0.05, \"DU\") = %.8f", q05))
Fx = ltratio_cdf([-Inf, 0.0, Inf, NaN], "DK", -5)
check(Fx[1] == 0 && Fx[3] == 1 && isnan(Fx[4]) && abs(Fx[2] - erf(1 / sqrt(2 * (exp(-10) - 1) / (-10)))) < 1e-12, "ltratio_cdf: non-finite x, and F(0) of DK at c = -5 in closed form")
check(try; ltratio_cdf(-2.0, "SU", 1); false; catch e; e isa ArgumentError; end, "ltratio_cdf rejects a stationary start with c > 0")

# 13. the tails with c0 != 0 (Raymond Kan, November 2026): the far roots of X(v) are kept, so the single integral does not
# fail there, and with method = :auto the values below 1e-3 come from the double integral (the single integral's error is
# absolute); a negative computed density is returned as zero
fe = try; ltratio_pdf([9.0, 3.0], "SU", -1.2; c0 = -1.2, method = :single); ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2, method = :single); ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :single); true; catch; false; end
check(fe, "no error at the far roots of X(v): SU c = -1.2, c0 = -1.2, 1.2, -0.05")
fs = ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :single); fd = ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :double)
check(abs(fs / fd - 1) < 1e-10, @sprintf("SU c = -1.2 c0 = -0.05, x = 3: single (far root kept) vs double, f = %.6e (%.1e relative)", fd, abs(fs / fd - 1)))
fa = ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2); fd = ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2, method = :double)
check(fa == fd && fa > 0, @sprintf("SU c = -1.2 c0 = 1.2, x = -9: the automatic method takes the double integral in the tail, f = %.3e", fa))
check(all(ltratio_pdf([8.5, 9.0, 12.0], "SU", -1.2; c0 = -1.2, method = :single) .>= 0), "a negative computed density is returned as zero")

isempty(fails) || error("quick check failed: " * join(fails, "; "))
println("quick check passed")
