# The local power of the 5% Dickey-Fuller tests, coefficient form and t-ratio, in the three regressions, with a zero and a
# stationary start, and the Elliott-Rothenberg-Stock power envelope (cases K and U, where it is the same curve).
# Writes LaTeX/figures/power_{K,U,T}.dat (columns c, tau0, rho0, tauS, rhoS, env) and prints the critical values, the
# points of 50% power and the checks.  Run from the repository root: julia --project=julia -t auto julia/application/power.jl
include(joinpath(@__DIR__, "..", "DFExact.jl")); using .DFExact
using Printf

# ---- the coefficient statistic: P[n(phi^ - 1) <= x] -> P[U - xV <= 0] = F_tau(0; c0 = x)
# (the line integral throughout; the conditional integral agrees with it to 2e-12 at every point of the curves since
# the domain-end fix of tstar, and method = :auto would give the same curves)
rho_cdf(x, cas, c) = DFExact.ltratio_cdf(0.0, cas, Float64(c); c0 = Float64(x), method = :single)
function bisect(f, lo, hi; tol = 1e-9)
    flo = f(lo); fhi = f(hi); @assert flo * fhi < 0
    for _ in 1:200
        m = (lo + hi) / 2; fm = f(m)
        (fm * flo < 0) ? (hi = m; fhi = fm) : (lo = m; flo = fm)
        hi - lo < tol && break
    end
    (lo + hi) / 2
end

# ---- the power envelope in case K: the likelihood ratio of the zero-start process, S = cb U - cb^2 V/2, rejects for
# large S; with a = cb/2 and cb < 0, S > k iff X_a = U - aV < b = k/cb.  P(X_a < b) by Gil-Pelaez from the joint
# moment generating function M(s,t) = e^{-(s+c)/2} (C - (c+s)S)^{-1/2}, C = cosh mu, S = sinh(mu)/mu, mu^2 = c^2 - 2t
# (Proposition 1, case K), with the phase of the determinant unwrapped along the imaginary axis.
function cf_X(c, a, u::Vector{Float64})
    D = Vector{ComplexF64}(undef, length(u)); E = similar(D)
    for (k, uk) in enumerate(u)
        s = im * uk; t = -a * s
        mu = sqrt(complex(c * c - 2t)); C = cosh(mu); S = abs(mu) < 1e-8 ? one(mu) : sinh(mu) / mu
        D[k] = C - (c + s) * S; E[k] = exp(-(s + c) / 2)
    end
    ph = DFExact.unwrap!(angle.(D))
    [E[k] * abs(D[k])^(-0.5) * exp(-im * ph[k] / 2) for k in eachindex(u)]
end
function P_below(c, a, b; N = 60000, umax = 4000.0)
    # Gil-Pelaez: P(X < b) = 1/2 - (1/pi) int_0^inf Im[phi(u) e^{-iub}]/u du, trapezoidal rule in u = sinh(v) with the
    # endpoint u = 0, where the integrand tends to E X - b (taken from u = 1e-7)
    vmax = asinh(umax); h = vmax / N
    v = [k * h for k in 0:N]; u = sinh.(v); u[1] = 1e-7; w = cosh.(v) .* h; w[1] /= 2
    phi = cf_X(c, a, u)
    g = [imag(phi[k] * exp(-im * u[k] * b)) / u[k] * w[k] for k in 1:N+1]
    0.5 - sum(g) / pi
end
function envelope(cb; alpha = 0.05)
    a = cb / 2
    b = bisect(b -> P_below(0.0, a, b) - alpha, -50.0, 50.0)         # size at c = 0
    P_below(cb, a, b)                                                # power at c = cb
end

# ---- the line integral against the conditional integral for the coefficient statistic, zero start, a few points
for (cas, c, x) in (("DK", -5.0, -8.0391), ("DU", -10.0, -14.0936), ("DT", -15.0, -21.7112), ("DK", -20.0, -8.0391))
    Fs = rho_cdf(x, cas, c); Fd = DFExact.ltratio_cdf(0.0, cas, c; c0 = x, method = :double)
    @printf("check line vs conditional: %s c=%g c0=%g  %.10f  %.10f  diff %.1e\n", cas, c, x, Fs, Fd, Fs - Fd)
end
# ---- checks of the Gil-Pelaez routine against the engine: P(U - aV < 0) at c = 0 and c = -5
for (c, a) in ((0.0, -8.0), (-5.0, -8.0), (0.0, -2.0))
    p1 = P_below(c, a, 0.0); p2 = rho_cdf(a, "DK", c)
    @printf("check Gil-Pelaez vs engine: c=%g a=%g  %.10f  %.10f  diff %.1e\n", c, a, p1, p2, p1 - p2)
end
@printf("envelope at cb = -7 (ERS: 0.50): %.4f\n", envelope(-7.0))

cs = collect(0.0:-0.5:-30.0)
cv = Dict{String,Tuple{Float64,Float64}}()
for (cas, scas, pan) in (("DK", "SK", "K"), ("DU", "SU", "U"), ("DT", "ST", "T"))
    cvt = DFExact.ltratio_icdf(0.05, cas, 0.0)
    cvr = bisect(x -> rho_cdf(x, cas, 0.0) - 0.05, -40.0, -1.0)
    cv[pan] = (cvt, cvr)
    @printf("%s: 5%% critical values  t-ratio %.5f  coefficient %.4f\n", pan, cvt, cvr)
    open(joinpath(@__DIR__, "..", "..", "LaTeX", "figures", "power_$pan.dat"), "w") do io
        println(io, "c tau0 rho0 tauS rhoS env")
        for c in cs
            t0 = DFExact.ltratio_cdf(cvt, cas, c); r0 = rho_cdf(cvr, cas, c)
            if c < 0
                tS = DFExact.ltratio_cdf(cvt, scas, c); rS = rho_cdf(cvr, scas, c)
            else
                tS = t0; rS = r0
            end
            env = pan == "T" ? NaN : (c < 0 ? envelope(c) : 0.05)
            @printf(io, "%.1f %.6f %.6f %.6f %.6f %.6f\n", c, t0, r0, tS, rS, env)
        end
    end
end
# the points of 50% power (linear interpolation) and the powers at c = -5, -10, -20
for pan in ("K", "U", "T")
    rows = [parse.(Float64, split(l)) for l in readlines(joinpath(@__DIR__, "..", "..", "LaTeX", "figures", "power_$pan.dat"))[2:end]]
    M = reduce(hcat, rows)'
    names = ("tau0", "rho0", "tauS", "rhoS", "env")
    for (j, nm) in enumerate(names)
        y = M[:, j+1]; any(isnan, y) && continue
        k = findfirst(i -> y[i] >= 0.5, 1:length(y)); c50 = k === nothing ? NaN : M[k-1, 1] + (0.5 - y[k-1]) / (y[k] - y[k-1]) * (M[k, 1] - M[k-1, 1])
        at = [y[findfirst(==(cc), M[:, 1])] for cc in (-5.0, -10.0, -20.0)]
        @printf("%s %-5s 50%% at c = %.2f   power at c = -5, -10, -20: %.3f %.3f %.3f\n", pan, nm, c50, at...)
    end
end
