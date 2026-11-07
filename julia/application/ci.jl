# Confidence sets for the persistence parameter c by inverting the Wald pivots: the t-ratio centered at c0, with the limit
# (U - c0 V)/sqrt(V) under c = c0, and the coefficient statistic n(phi^ - 1 - c0/n), with the limit (U - c0 V)/V.  The
# 95% belts (the 2.5%, 50% and 97.5% quantiles of the pivots under c = c0) and the median curve, which gives the
# median-unbiased estimator, for case U with a zero and a stationary start and for the GLS-detrended trend case DG; the
# probability that a false value c0 is excluded from the 95% set when the true value is c; and one worked example.
# Writes LaTeX/figures/ci_belts_U.dat, ci_belts_T.dat, ci_belts_G.dat, ci_excl_c5.dat, ci_excl_c10.dat (the exclusion
# probabilities for case U with a zero and a stationary start, case T and the GLS-detrended trend case).  Run from the repository root:
# julia --project=julia -t auto julia/application/ci.jl
include(joinpath(@__DIR__, "..", "DFExact.jl")); using .DFExact
using Printf

const FIG = joinpath(@__DIR__, "..", "..", "LaTeX", "figures")
function bisect(f, lo, hi; tol = 1e-8)
    flo = f(lo); fhi = f(hi); flo * fhi < 0 || error("no bracket")
    for _ in 1:200
        m = (lo + hi) / 2; fm = f(m)
        (fm * flo < 0) ? (hi = m; fhi = fm) : (lo = m; flo = fm)
        hi - lo < tol && break
    end
    (lo + hi) / 2
end
# the case name of a start at c0: the stationary start exists for c0 < 0 only; at c0 >= 0 the zero start
casof(cas, stat, c0) = (stat && c0 < 0) ? "S" * cas[2:2] : cas
# the distribution function under c of the pivot centered at c0: t-ratio P(tau_{c0} <= x); coefficient P((U - c0 V)/V <= y)
Ft(x, cas, c, c0) = DFExact.ltratio_cdf(x, cas, c; c0 = c0)
Fr(y, cas, c, c0) = DFExact.ltratio_cdf(0.0, cas, c; c0 = c0 + y)
# the quantiles of the pivots under c = c0
qt(p, cas, c0) = DFExact.ltratio_icdf(p, cas, c0; c0 = c0)
qr(p, cas, c0) = bisect(y -> Fr(y, cas, c0, c0) - p, -80.0, 40.0)

c0s = collect(4.0:-0.5:-30.0)
ps = (0.025, 0.5, 0.975)
# ---- belts, case U: zero start and stationary start; and case DG (zero start)
belts = Dict{Tuple{String,Bool},Matrix{Float64}}()
for (cas, stat) in (("DU", false), ("DU", true), ("DT", false), ("DG", false))
    B = zeros(length(c0s), 6)
    for (i, c0) in enumerate(c0s)
        cs = casof(cas, stat, c0)
        for (j, p) in enumerate(ps)
            B[i, j] = c0 + qr(p, cs, c0)                              # the belt for rho^ = n(phi^ - 1)
            B[i, j+3] = qt(p, cs, c0)                                 # the belt for the t-ratio centered at c0
        end
    end
    belts[(cas, stat)] = B
    @printf("belts %s stat=%s done\n", cas, stat)
end
open(joinpath(FIG, "ci_belts_U.dat"), "w") do io
    println(io, "c0 rlo0 rmed0 rhi0 tlo0 tmed0 thi0 rloS rmedS rhiS tloS tmedS thiS")
    Z = belts[("DU", false)]; S = belts[("DU", true)]
    for i in eachindex(c0s); @printf(io, "%.1f %s %s\n", c0s[i], join([@sprintf("%.6f", v) for v in Z[i, :]], " "), join([@sprintf("%.6f", v) for v in S[i, :]], " ")); end
end
for (cas, nm) in (("DT", "T"), ("DG", "G"))
    open(joinpath(FIG, "ci_belts_$nm.dat"), "w") do io
        println(io, "c0 rlo0 rmed0 rhi0 tlo0 tmed0 thi0")
        Z = belts[(cas, false)]
        for i in eachindex(c0s); @printf(io, "%.1f %s\n", c0s[i], join([@sprintf("%.6f", v) for v in Z[i, :]], " ")); end
    end
end
# ---- the probability that c0 is excluded from the 95% set when the true value is c: the power of the two-sided 5% test
for c in (-5.0, -10.0)
    open(joinpath(FIG, @sprintf("ci_excl_c%d.dat", round(Int, -c))), "w") do io
        println(io, "c0 r0 t0 rS tS rT tT rG tG")
        for (i, c0) in enumerate(c0s)
            vals = Float64[]
            for (cas, stat) in (("DU", false), ("DU", true), ("DT", false), ("DG", false))
                B = belts[(cas, stat)]; ct = stat ? "SU" : cas            # the true distribution under c (c < 0: the stationary start exists)
                rlo, rhi = B[i, 1] - c0, B[i, 3] - c0                      # quantiles of the coefficient pivot (U - c0 V)/V
                er = Fr(rlo, ct, c, c0) + 1 - Fr(rhi, ct, c, c0)
                et = Ft(B[i, 4], ct, c, c0) + 1 - Ft(B[i, 6], ct, c, c0)
                push!(vals, er, et)
            end
            @printf(io, "%.1f %s\n", c0, join([@sprintf("%.6f", v) for v in vals], " "))
        end
    end
    println("exclusion at c = $c done")
end
# ---- the worked example: case U, observed rho^ = -8 and tau^ = -2 (n se = 4)
rho, tau = -8.0, -2.0
function sets(stat)
    cs(c0) = casof("DU", stat, c0)
    # coefficient form: {c0 : c0 + q_.025(c0) <= rho <= c0 + q_.975(c0)}; t-ratio form: {c0 : q_.025(c0) <= tau (1 - c0/rho) <= q_.975(c0)}
    glo = c0 -> c0 + qr(0.025, cs(c0), c0) - rho; ghi = c0 -> c0 + qr(0.975, cs(c0), c0) - rho; gmd = c0 -> c0 + qr(0.5, cs(c0), c0) - rho
    hlo = c0 -> qt(0.025, cs(c0), c0) - tau * (1 - c0 / rho); hhi = c0 -> qt(0.975, cs(c0), c0) - tau * (1 - c0 / rho); hmd = c0 -> qt(0.5, cs(c0), c0) - tau * (1 - c0 / rho)
    # the belts increase in c0, so the upper end of the set is where the lower bound meets the observation, and vice versa
    r = (bisect(ghi, -40.0, 10.0; tol = 1e-4), bisect(gmd, -40.0, 10.0; tol = 1e-4), bisect(glo, -40.0, 10.0; tol = 1e-4))
    t = (bisect(hhi, -40.0, 10.0; tol = 1e-4), bisect(hmd, -40.0, 10.0; tol = 1e-4), bisect(hlo, -40.0, 10.0; tol = 1e-4))
    r, t
end
for stat in (false, true)
    r, t = sets(stat)
    @printf("example rho=%g tau=%g, %s start: coefficient set [%.2f, %.2f] median-unbiased %.2f; t-ratio set [%.2f, %.2f] median-unbiased %.2f\n",
        rho, tau, stat ? "stationary" : "zero", r[1], r[3], r[2], t[1], t[3], t[2])
end
