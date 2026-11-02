# Certification of DFExact.jl by node doubling (every adaptive node count and every nodes-per-panel count of the fixed layouts), contour independence and the uniform rule, for c > 0 up to the
# far upper tail and for c <= 0; run with julia -t auto --project=julia julia/certify_dfexact.jl (about 15 minutes).
using Printf
include(joinpath(@__DIR__, "DFExact.jl")); using .DFExact; const D = DFExact
for model in 1:3, c in (-3.0, 0.0, 2.0)
    D.cdf(-1.0, model; c = c); D.density(-1.0, model; c = c); D.cdf(0.5, model; c = c); D.density(0.5, model; c = c)
end
println("c > 0, default tau; q to the far upper tail")
for model in 1:3, c in (1.0, 2.0, 3.0, 5.0, 10.0)
    tau = D.default_tau(model, c); ts = D.tstar(model, c)
    qs = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0]
    wF = wf = wFt = wFu = wfu = 0.0; whereF = wheref = ""; tF = tf = 0.0
    Fs = Float64[]; mono = true; Fprev = -Inf
    for q in qs
        tF += @elapsed F = D.cdf(q, model; c = c); tf += @elapsed f = D.density(q, model; c = c)
        push!(Fs, F); mono &= F >= Fprev - 1e-12; Fprev = F
        D.NODEFAC[] = 2; D.NPAN[] *= 2; D.NPANK[] *= 2; D.NPANR[] *= 2; F2 = D.cdf(q, model; c = c); f2 = D.density(q, model; c = c); D.NODEFAC[] = 1; D.NPAN[] ÷= 2; D.NPANK[] ÷= 2; D.NPANR[] ÷= 2
        F3 = D.cdf(q, model; c = c, tau = ts / 4)
        if abs(F - F2) > wF; wF = abs(F - F2); whereF = "q=$q"; end
        if abs(f - f2) / abs(f) > wf && f > 1e-12; wf = abs(f - f2) / abs(f); wheref = "q=$q"; end
        wFt = max(wFt, abs(F - F3))
        if q in (0.0, 3.0, 20.0) && tau >= 1e-3                      # (the uniform rule needs the step tau/4: hopeless for c = 10)
            hw = min(0.15, tau / 4)
            Fu = D.cdf_uniform(q, model; c = c, hw = hw); fu = D.density_uniform(q, model; c = c, hw = hw)
            wFu = max(wFu, abs(F - Fu)); wfu = max(wfu, abs(f - fu) / abs(f))
        end
    end
    @printf("M%d c=%g tau=%.4f: F(8)=%.6f F(20)=%.6f F(80)=%.9f monotone=%s; doubling |dF| %.1e (%s), |df/f| %.1e (%s); vs tau=t*/4 %.1e; vs uniform |dF| %.1e, |df/f| %.1e; mean F %.1f ms, f %.1f ms\n",
            model, c, tau, Fs[10], Fs[12], Fs[14], mono, wF, whereF, wf, wheref, wFt, wFu, wfu, 1e3tF / length(qs), 1e3tf / length(qs))
    flush(stdout)
end
println()
println("normalization and mean, c = 3 and c = -10 (trapezoid on a fine grid of the density):")
for model in 1:3, c in (-10.0, 3.0)
    lo = c < 0 ? -sqrt(-c / 2) - 12 : -8.0; hi = c < 0 ? -sqrt(-c / 2) + 12 : 120.0
    qs = range(lo, hi, length = c > 0 ? 6401 : 2401); h = step(qs)
    fs = [D.density(q, model; c = c) for q in qs]
    @printf("  M%d c=%g: int f = %.12f, mean = %.6f, F(hi) = %.12f, min f = %.1e\n", model, c, sum(fs) * h, sum(fs .* qs) * h, D.cdf(hi, model; c = c), minimum(fs))
end
println()
println("c <= 0 regression (certified paths), q in [-4, 3] shifted:")
for model in 1:3, c in (0.0, -5.0, -20.0)
    ctr = c < 0 ? -sqrt(-c / 2) : 0.0
    qs = ctr .+ [-4.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
    wF = wf = wFu = wfu = 0.0; tF = tf = 0.0
    for q in qs
        tF += @elapsed F = D.cdf(q, model; c = c); tf += @elapsed f = D.density(q, model; c = c)
        D.NODEFAC[] = 2; D.NPAN[] *= 2; D.NPANK[] *= 2; D.NPANR[] *= 2; F2 = D.cdf(q, model; c = c); f2 = D.density(q, model; c = c); D.NODEFAC[] = 1; D.NPAN[] ÷= 2; D.NPANK[] ÷= 2; D.NPANR[] ÷= 2
        wF = max(wF, abs(F - F2)); wf = max(wf, abs(f - f2) / abs(f))
        if q in qs[[2, 6, 9]]
            Fu = D.cdf_uniform(q, model; c = c, hw = 0.15); fu = D.density_uniform(q, model; c = c, hw = 0.15)
            wFu = max(wFu, abs(F - Fu)); wfu = max(wfu, abs(f - fu) / abs(f))
        end
    end
    @printf("M%d c=%5.1f: doubling |dF| %.1e, |df/f| %.1e; vs uniform |dF| %.1e, |df/f| %.1e; mean F %.1f ms, f %.1f ms\n", model, c, wF, wf, wFu, wfu, 1e3tF / length(qs), 1e3tf / length(qs))
    flush(stdout)
end
