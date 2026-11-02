# Abadir (1995) against DFExact.jl in case DK at c = 0: for z > 0 his (2.17)-(2.18), single integrals of parabolic
# cylinder functions (here in their Bessel form, D_{-1/2}(w) = sqrt(w/2pi) K_{1/4}(w^2/4), D_{1/2}(w) = (w/2) sqrt(w/2pi)
# {K_{1/4} + K_{3/4}}(w^2/4), with the exponentials scaled); for z < 0 his (2.11), the triple series evaluated by
# abadir_series.py (mpmath, 40 digits) into abadir_series.json.  Run from this directory:
#   julia -t auto --project=.. abadir_check.jl
using Printf, QuadGK, SpecialFunctions, JSON
include(joinpath(@__DIR__, "..", "DFExact.jl")); using .DFExact; const D = DFExact
# exp(s^2/2 - w^2/4) D_{-1/2}(w) and exp(s^2/2 - w^2/4) D_{1/2}(w) without overflow
function scaled(s, z)
    co = coth(2z * s); w = s + z * co
    e = exp((s * s - w * w) / 2)                                   # = exp(s^2/2 - w^2/2) <= 1
    k1 = besselkx(0.25, w * w / 4); k3 = besselkx(0.75, w * w / 4)
    r = sqrt(w / (2pi)) * e
    r * k1, (w / 2) * r * (k1 + k3), co
end
function F217(z)
    g = s -> begin
        d, _, co = scaled(s, z)
        sqrt(2z / sinh(2z * s)) * d / s
    end
    1 - quadgk(g, 0.0, Inf; rtol = 1e-14)[1] / pi
end
function f218(z)
    g = s -> begin
        d, d1, co = scaled(s, z)
        sqrt(2 / (z * sinh(2z * s))) * ((s * s - 0.5) * d + (z * co - s) * d1) / s
    end
    quadgk(g, 0.0, Inf; rtol = 1e-14)[1] / pi
end
ser = JSON.parsefile(joinpath(@__DIR__, "abadir_series.json"))
zs = [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
println("case DK, c = 0: DFExact.jl against Abadir (1995), (2.11) for z < 0 (density by central differences), (2.17)-(2.18) for z > 0")
wF = wf = 0.0
for z in zs
    F = D.cdf(z, 1); f = D.density(z, 1)
    if z < 0
        key = z == round(z) ? string(Int(z)) : string(z)
        Fa = parse(Float64, ser[key]["F"]); fa = parse(Float64, ser[key]["f"])
    else
        Fa = F217(z); fa = f218(z)
    end
    global wF = max(wF, abs(F - Fa)); global wf = max(wf, abs(f - fa))
    @printf("z = %5.2f: F = %.15f (Abadir %.15f, diff %.1e)  f = %.15f (Abadir %.15f, diff %.1e)\n", z, F, Fa, F - Fa, f, fa, f - fa)
end
@printf("worst |dF| %.1e, worst |df| %.1e\n", wF, wf)
f0 = D.density(0.0, 1)
@printf("density at zero: %.9f (Abadir's (3.6): (pi e)^{-1/2} = %.7f); mode at %.5f\n", f0, (pi * exp(1))^(-0.5), -0.7235)
# the mode
m = -0.72; for it in 1:30; h = 1e-4; d1 = (D.density(m + h, 1) - D.density(m - h, 1)) / 2h; d2 = (D.density(m + h, 1) - 2D.density(m, 1) + D.density(m - h, 1)) / h^2; global m -= d1 / d2; end
@printf("mode of the density of tau at c = 0: %.6f (Abadir: -0.724)\n", m)
