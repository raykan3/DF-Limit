"""
    DFExact

Exact density and distribution function of the limiting Dickey–Fuller t-ratio tau = U/sqrt(V),
Models M1–M3 with local-to-unity parameter c (zero initial value and drift), by the inversion
formulas of Theorem 1 with the s-integral done analytically: a port of df_exact2.py.

    density(q, model; c = 0.0, start = :zero)   # f(q); start = :stationary for the stationary start (c < 0)
    cdf(q, model; c = 0.0, start = :zero)       # P(tau <= q)
    quantile(prob, model; c = 0.0, start = :zero)
    moments(model; c = 0.0, start = :zero, pmax = 4)   # raw moments, one-dimensional integrals
    tstar(model, c)                             # the convergence abscissa in t

The line Re t = tau of the outer integral defaults to 1 for c <= 0 (where the abscissa t_* exceeds
1 in all three models) and to t_*/2 for c > 0, where t_* decreases to zero as c grows; `tau` can be
given explicitly.

For f = 0 the determinant D(s,t) is a polynomial in s of degree one (M1) or two (M2, M3) with
coefficients depending on t alone, so that the inverse two-sided Laplace transform in s is explicit
(a scaled chi-square(1) density for M1, the density of a difference of two scaled chi-square(1)
variables, a Bessel K_0, for M2 and M3).  The remaining integrals are a Gauss–Legendre rule along
paths in the complex plane on which the Gaussian factor decays, and a trapezoidal rule along
Re t = tau.  The t-loops run on NWORKERS[] tasks (start Julia with -t 6 or more; -t auto gives only the performance
cores on macOS): by default the performance cores plus a
third of the efficiency cores on macOS and the physical cores elsewhere, since using every hardware thread of a
machine with efficiency cores or hyperthreading makes it slower; set DFExact.NWORKERS[] to override.
"""
module DFExact

using SpecialFunctions, LinearAlgebra, QuadGK
export ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom
include("df_series.jl")

const Z0 = 8.0            # |z| below which D is evaluated by its Taylor series (cancellation in num/z^k)

# ---------- the number of tasks of the threaded loops ----------
function default_workers()
    n = Threads.nthreads()
    cores = try
        if Sys.isapple()
            p = parse(Int, strip(read(`sysctl -n hw.perflevel0.physicalcpu`, String)))
            e = try parse(Int, strip(read(`sysctl -n hw.perflevel1.physicalcpu`, String))) catch; 0 end
            p + e                                       # (the loops are scheduled dynamically: the efficiency cores help)
        else
            max(1, Sys.CPU_THREADS ÷ 2)                 # physical cores, assuming two hardware threads each
        end
    catch
        max(1, Sys.CPU_THREADS ÷ 2)
    end
    max(1, min(n, cores))
end
const NWORKERS = Ref(1)
function __init__()
    NWORKERS[] = default_workers()
    HAVE_ACCEL[] = Sys.isapple() && (try; Libdl.dlopen(ACCEL); true; catch; false; end)
    load_ivm()
    k0tables!()
end

"""
    tmap!(f, out, n)

out[j] = f(j) for j = 1..n on NWORKERS[] tasks, each taking every NWORKERS[]-th index (the cost of the
inner integrals varies slowly along the line, so the interleaving balances the tasks).
"""
function tmap!(f, out::AbstractVector, n::Int)
    nw = NWORKERS[]
    if nw <= 1
        for j in 1:n; out[j] = f(j); end
    else
        @sync for w in 1:nw
            Threads.@spawn for j in w:nw:n; out[j] = f(j); end
        end
    end
    out
end

# ---------- entire functions of z = mu^2 ----------
"""cosh mu, (1 - cosh mu)/z, sinh mu / mu, each entire in z, with series near z = 0."""
function pieces(z::Complex{T}) where {T}
    mu = sqrt(z)
    C = cosh(mu)
    if abs(z) < 1e-3
        S = 1 + z / 6 + z * z / 120 + z^3 / 5040
        Q = -one(T) / 2 - z / 24 - z * z / 720 - z^3 / 40320
    else
        S = sinh(mu) / mu; Q = (1 - C) / z
    end
    C, Q, S
end

# ---------- the stationary start ----------
# With y_0 = sigma sqrt(n) gamma_0 and gamma_0 ~ N(0, sigma_0^2), sigma_0^2 = -1/(2c), c < 0 (cases SK, SU, ST),
# the joint mgf is the average over gamma_0 of the fixed-start one.  For M2 and M3 the fixed start enters the
# Nabeya-Sorensen mgf through f = c gamma_0 (M2) or c^2 gamma_0 (M3) as log M = log M_0 + f^2 Q(s, t) with
# Q = e^c (t + s^2/2 + c s) g(z)/D, g = (2Q + S)/z (M2) or [-12 C + (24 - 4z) Q + (24 + z) S]/z^3 (M3) (df_mgf.py,
# checked there by Monte Carlo); averaging gives (1 + c Q)^{-1/2} (M2) and (1 + c^3 Q)^{-1/2} (M3), that is
# D_stat = D + c^k e^c (t + s^2/2 + c s) g(z), k = 1 or 3, again a polynomial of degree two in s.  For M1 the
# tilted pair (J(0), J(1)) is Gaussian with the covariance of Bao, Broda and Kan (DF-Test, Section 7.4) and
# R = (J(1)^2 - J(0)^2)/2, which gives D_stat = e^c [C - cS + S (2t + s^2)/(2c)]: the roots are +-, as time
# reversal of the stationary process requires.  The functions g are entire; near z = 0 their Taylor series.
const G2SER, G3SER = let
    rf = [1.0]; for m in 1:70; push!(rf, rf[end] / m); end          # rf[m+1] = 1/m!, by the recurrence (exact to rounding)
    cc(k) = k < 0 ? 0.0 : rf[2k+1]; qq(k) = k < 0 ? 0.0 : -rf[2k+3]; ss(k) = k < 0 ? 0.0 : rf[2k+2]
    g2 = [2qq(k + 1) + ss(k + 1) for k in 0:29]
    n(k) = -12cc(k) + 24qq(k) - 4qq(k - 1) + 24ss(k) + ss(k - 1)
    g3 = [n(k + 3) for k in 0:29]
    g2, g3
end
function gstat(model::Int, z::Complex{T}) where {T}
    if abs(z) < Z0
        ser = model == 2 ? G2SER : G3SER
        acc = zero(z); for j in length(ser):-1:1; acc = acc * z + ser[j]; end
        return acc
    end
    C, Q, S = pieces(z)
    model == 2 ? (2Q + S) / z : (-12 * (1 - z * Q) + Q * (24 - 4z) + S * (24 + z)) / z^3
end
Dstat_add(model::Int, c::T, s::Complex{T}, t::Complex{T}) where {T} = (model == 2 ? c : c^3) * exp(c) * (t + s * s / 2 + c * s) * gstat(model, c * c - 2t)
"""g(z) e^{-mu}, mu = sqrt(z): the entire function of gstat with its exponential growth taken out, so that nothing
overflows at the large tilts of the saddle-point searches (cosh mu overflows at |z| = 5e5)."""
function gstat_scaled(model::Int, z::Complex{T}) where {T}
    mu = sqrt(z)
    abs(z) < Z0 && return gstat(model, z) * exp(-mu)
    e2 = exp(-2mu)
    Ct = (1 + e2) / 2; St = (1 - e2) / (2mu); Qt = (exp(-mu) - Ct) / z          # cosh mu, sinh mu/mu, (1 - cosh mu)/z, times e^{-mu}
    model == 2 ? (2Qt + St) / z : (-12Ct + Qt * (24 - 4z) + St * (24 + z)) / z^3
end
kernel(model::Int, stat::Bool) = ((model == 1 || model == 4) && !stat) ? 1 : 2   # rank of the quadratic form R under the tilt (4: GLS-detrended, trend)

"""D(s, t) for f = 0 (log M = -s/2 + extra - log(D)/2, extra = -c/2 for M1 and 0 otherwise); with stat = true the
stationary-start determinant, for which extra = -c/2 - log(e^c)/2... is absorbed: D_stat includes the factor e^c for M1 too."""
function Dfun(model::Int, c::T, s::Complex{T}, t::Complex{T}; stat::Bool = false) where {T}
    lam = -s; lx = t; z = c * c - 2t
    C, Q, S = pieces(z)
    e = exp(c)
    if stat
        model == 1 && return e * (C - c * S + S * (2t + s * s) / (2c))
        return Dfun(model, c, s, t) + Dstat_add(model, c, s, t)
    end
    if model == 1
        return C - (c + s) * S
    elseif model == 2
        A = 2 * c * c * lam - 4 * c * lx - 2 * lam * lam
        B = -c^3 + c * c * lam - 2 * lx - lam * lam
        if abs(z) < Z0
            return e * sum(D1_SERIES[j + 1](lam, c, T) * z^j for j in 0:length(D1_SERIES)-1)
        end
        return e * (c * c * (1 - z * Q) + A * Q + B * S) / z
    else
        c2, c3, c4, c5, c6, c7 = c^2, c^3, c^4, c^5, c^6, c^7
        P1 = c6 - 12 * c4 * lam + 2 * lx * (-c4 + 12 * c3 - 12 * c2) + 48 * lam * lam * (c2 - 3 * c + 3)
        P2 = (4 * lam * (-c6 + 6 * c4) + 16 * lam * lam * (c4 - 3 * c3 - 3 * c2 + 18 * c - 18)
              + 8 * lx * (c5 - 3 * c4 + c4 * lam - 4 * lam * lam * (c2 - 3 * c + 3)) + 16 * lx * lx * (-c3 + 3 * c2 - 6 * c + 6))
        P3 = (-c7 + lam * (c6 + 24 * c4) + 4 * lam * lam * (-c4 + 3 * c3 - 27 * c2 + 72 * c - 72)
              + 2 * lx * (c5 - 4 * c4 - 12 * c3 + 12 * c2 - c4 * lam + 4 * lam * lam * (c2 - 3 * c + 3)) + 16 * lx * lx * (c2 - 3 * c + 3))
        if abs(z) < Z0
            return e * sum(D2_SERIES[j + 1](lam, c, T) * z^j for j in 0:length(D2_SERIES)-1)
        end
        return e * ((1 - z * Q) * P1 + Q * P2 + S * P3) / z^3
    end
end

function unwrap!(ph::Vector{T}) where {T}
    for j in 2:length(ph)
        d = ph[j] - ph[j-1]
        ph[j] -= 2T(pi) * round(d / (2T(pi)))
    end
    ph
end

"""P(t), s1(t), s2(t) (s2 unused for M1) and M0(t) = (-s1 s2)^{-1/2}, the total mass of phi_t for M2 and
M3 (unused for M1), on the grid t = tau + i w, w increasing from 0, with the branches of the square
roots and the labelling of the roots fixed by continuity along the grid.  For c > CGIRS with a zero start
the roots and P(t) come from the Girsanov forms (girsanov), which are free of the cancellation of D."""
function tparts(model::Int, c::T, t::Vector{Complex{T}}; stat::Bool = false) where {T}
    nt = length(t)
    P = Vector{Complex{T}}(undef, nt); s1 = similar(P); s2 = similar(P); M0 = similar(P)
    if model == 1 && !stat
        S = Vector{Complex{T}}(undef, nt)
        for j in 1:nt
            C, Q, Sj = pieces(c * c - 2t[j])
            S[j] = Sj
            # s1 = mu coth mu - c; for c > 0 without the cancellation of C - c S at large c: mu - c = -2t/(mu + c),
            # mu coth mu - mu = 2 mu e^{-2mu}/(1 - e^{-2mu})
            s1[j] = c > 0 ? eta1(c, t[j]) : (C - c * Sj) / Sj
        end
        ph = unwrap!(angle.(S))
        for j in 1:nt
            P[j] = exp(-c / 2) * abs(S[j])^(-T(1) / 2) * exp(-im * ph[j] / 2)
        end
        return P, s1, s2, M0
    end
    if model == 4 && !stat
        # the GLS-detrended trend case: M(s,t) = e^{-s/2} P(t) (s_1 - s)^{-1/2} as in case DK, with M_{V_g}(t) = e^{-c/2}
        # (cosh th |I - Q Sig|)^{-1/2} and s_1 = 1/(2 kap_1), 2 kap_1 = (1 - lam)^2 v' Sig (I - Q Sig)^{-1} v the variance of the
        # detrended end point under the tilt; branches continuous along the nodes (t[1] real)
        lam = lam_gls(CBAR_GLS); v = Complex{T}.(VGLS)
        lD = Vector{Complex{T}}(undef, nt); lch = similar(lD)
        for j in 1:nt
            G = girsanov(4, c, t[j])
            lD[j] = log(G.D); lch[j] = G.lch
            s1[j] = 1 / ((1 - lam)^2 * (transpose(v) * (G.Sg * (G.N \ v))))
            s2[j] = zero(t[j]); M0[j] = zero(t[j])
        end
        ph = unwrap!(imag.(lD)); ps = unwrap!(angle.(s1))
        for j in 1:nt
            P[j] = exp(-c / 2 - (lch[j] + real(lD[j]) + im * ph[j]) / 2 + (log(abs(s1[j])) + im * ps[j]) / 2)
        end
        return P, s1, s2, M0
    end
    sa = similar(P); sb = similar(P)
    if !stat && c > CGIRS
        # cases DU, DT at large c: D(s,t)/e^c cancels catastrophically (terms of size e^{2c} against a value near
        # one); the roots s_k = 1/nu_k of A2 nu^2 - A1 nu + A0 = 0 and M_V(t) = e^{-c/2} (cosh th det(I - Q Sig))^{-1/2}
        # come from the Girsanov forms, and P = M_V (-s1 s2)^{1/2}
        lD = Vector{Complex{T}}(undef, nt); lch = similar(lD)
        for j in 1:nt
            G = girsanov(model, c, t[j])
            disc = sqrt(G.A1 * G.A1 - 4 * G.A0 * G.A2)
            sw = sign(real(conj(G.A1) * disc)); sw == 0 && (sw = one(sw))
            rb = (G.A1 + sw * disc) / (2 * G.A0); rs = G.A2 / (G.A0 * rb)   # the larger root from the formula, the smaller from the product
            sa[j] = rb; sb[j] = rs
            lD[j] = log(G.D); lch[j] = G.lch
        end
        if real(sa[1]) < real(sb[1]); s1[1], s2[1] = sa[1], sb[1] else s1[1], s2[1] = sb[1], sa[1] end
        for j in 2:nt
            if abs(sa[j] - s1[j-1]) + abs(sb[j] - s2[j-1]) <= abs(sb[j] - s1[j-1]) + abs(sa[j] - s2[j-1])
                s1[j], s2[j] = sa[j], sb[j]
            else
                s1[j], s2[j] = sb[j], sa[j]
            end
        end
        ph = unwrap!(imag.(lD))
        m = [-s1[j] * s2[j] for j in 1:nt]
        phm = unwrap!(angle.(m))
        for j in 1:nt
            lm = log(abs(m[j])) + im * phm[j]
            M0[j] = exp(-lm / 2)
            P[j] = exp(-c / 2 - (lch[j] + real(lD[j]) + im * ph[j]) / 2 + lm / 2)
        end
        return P, s1, s2, M0
    end
    ma2 = Vector{Complex{T}}(undef, nt); rts = Vector{NTuple{3,Complex{T}}}(undef, nt)
    # the Taylor coefficients in z of D(., t)/e^c at s = 0, -1, 1 (lam = 0, 1, -1), computed once
    SER = model == 2 ? D1_SERIES : D2_SERIES
    coef = [[f(lam, c, T) for f in SER] for lam in (zero(T), one(T), -one(T))]
    Dser(k, z) = (acc = zero(z); for j in length(coef[k]):-1:1; acc = acc * z + coef[k][j]; end; exp(c) * acc)
    s0, sp, sm = complex(zero(T)), complex(-one(T)), complex(one(T))
    tmap!(j -> begin
        z = c * c - 2t[j]
        if model == 1                                                  # (the stationary start: no series needed)
            d0 = Dfun(1, c, s0, t[j]; stat = true); dp = Dfun(1, c, sp, t[j]; stat = true); dm = Dfun(1, c, sm, t[j]; stat = true)
        elseif abs(z) < Z0
            d0 = Dser(1, z); dp = Dser(2, z); dm = Dser(3, z)
            if stat
                d0 += Dstat_add(model, c, s0, t[j]); dp += Dstat_add(model, c, sp, t[j]); dm += Dstat_add(model, c, sm, t[j])
            end
        else
            d0 = Dfun(model, c, s0, t[j]; stat); dp = Dfun(model, c, sp, t[j]; stat); dm = Dfun(model, c, sm, t[j]; stat)
        end
        a2 = (dp + dm) / 2 - d0; a1 = (dp - dm) / 2; a0 = d0          # D = a2 lam^2 + a1 lam + a0, lam = -s
        disc = sqrt(a1 * a1 - 4 * a2 * a0)
        # the root of larger magnitude from the formula, the other from the product a0/a2 (the formula loses the
        # small root to cancellation when the roots differ by orders of magnitude, as at large c)
        sw = sign(real(conj(a1) * disc)); sw == 0 && (sw = one(sw))
        rb = (-a1 - sw * disc) / (2a2); rs = a0 / (a2 * rb)
        (-rb, -rs, -a2)
    end, rts, nt)
    for j in 1:nt; sa[j], sb[j], ma2[j] = rts[j]; end
    # at the real point s1 < 0 < s2; then continuity
    if real(sa[1]) < real(sb[1]); s1[1], s2[1] = sa[1], sb[1] else s1[1], s2[1] = sb[1], sa[1] end
    for j in 2:nt
        if abs(sa[j] - s1[j-1]) + abs(sb[j] - s2[j-1]) <= abs(sb[j] - s1[j-1]) + abs(sa[j] - s2[j-1])
            s1[j], s2[j] = sa[j], sb[j]
        else
            s1[j], s2[j] = sb[j], sa[j]
        end
    end
    abs(angle(ma2[1])) < 1e-8 || error("prefactor not positive at the real point")
    ph = unwrap!(angle.(ma2))
    for j in 1:nt
        P[j] = abs(ma2[j])^(-T(1) / 2) * exp(-im * ph[j] / 2)       # D = (-a2)(s - s1)(s2 - s), P = (-a2)^{-1/2}
    end
    m = [-s1[j] * s2[j] for j in 1:nt]                              # positive at the real point
    ph = unwrap!(angle.(m))
    for j in 1:nt
        M0[j] = abs(m[j])^(-T(1) / 2) * exp(-im * ph[j] / 2)
    end
    P, s1, s2, M0
end

# ---------- the Girsanov forms (zero start): everything free of cancellation for c > 0 ----------
# Under the law of J with a zero start, E_c Phi(J) = E{Phi(W) exp[c (W(1)^2 - 1)/2 - c^2/2 int W^2]} for Brownian
# motion W.  With th = (c^2 - 2t)^{1/2}, under the probability with density (cosh th)^{1/2} exp(-th^2/2 int W^2)
# the vector X = (W(1), int W[, int r W]) is Gaussian with covariance Sig(th) (tanh and sech in closed form),
# e^{tV} gains exp(X'QX/2) with Q = diag(c, -2t Gm), Gm the inverse Gram matrix of the deterministic terms, and
# R = X'KX/2.  Then M_V(t) = e^{-c/2} (cosh th det(I - Q Sig))^{-1/2}, and the weights nu = 2 kappa of R under
# the tilt solve A2 nu^2 - A1 nu + A0 = 0 with A2 = det(Sig^{-1} - Q) = det(I - Q Sig)/det Sig, A1 = tr{K adj(P')},
# A0 = tr{adj(K) P'}, P' = Sig^{-1} - Q.  Case DK: s1 = eta = th coth th - c, M_V = e^{-c/2} (th/sinh th)^{1/2} eta^{-1/2}.
# The entries of Sig are formed from e^{-2 th}, th - c as -2t/(th + c), and eta = (th - c) + 2 th e^{-2th}/(1 - e^{-2th});
# for |th| < 1 the Taylor series in q = th^2.  (The DF-Test paper's IIPAR, ported from Raymond Kan's MATLAB.)
const CGIRS = 2.0           # the Girsanov forms replace D(s,t) for c > CGIRS (zero start)
const GT = [1.0, -0.33333333333333333, 0.13333333333333333, -0.053968253968253968, 0.021869488536155203, -0.0088632355299021966, 0.003592128036572481, -0.0014558343870513183, 0.00059002744094558598, -0.00023912911424355248, 9.6915379569294503e-5, -3.9278323883316834e-5, 1.5918905069328965e-5, -6.4516892156554308e-6, 2.6147711512907546e-6, -1.0597268320104654e-6, 4.2949110782738059e-7, -1.7406618963571648e-7, 7.0546369464009683e-8, -2.8591366623052539e-8, 1.1587644432798852e-8, -4.6962953982309016e-9, 1.9033368339312759e-9, -7.7139336353590623e-10, 3.126339545892087e-10, -1.2670576930305401e-10, 5.1351914080393677e-11, -2.0812146867700474e-11, 8.4348454190943383e-12, -3.4185140868111558e-12, 1.3854715742948469e-12, -5.6151047924146801e-13, 2.2757162553728748e-13, -9.2231305851395317e-14, 3.7379940310967389e-14, -1.5149519187148605e-14, 6.1398688626168138e-15, -2.4883951222762789e-15, 1.0085085566354096e-15, -4.0873312268690138e-16, 1.6565329513786284e-16, -6.7136751750487117e-17, 2.7209500613044588e-17, -1.1027595233722371e-17, 4.4693160072374125e-18]   # tanh(th)/th = sum GT[k] q^{k-1}
const GE = [1.0, -0.5, 0.20833333333333333, -0.084722222222222222, 0.034350198412698413, -0.013922233245149912, 0.0056424968100315323, -0.0022868190951648293, 0.00092681292737742184, -0.0003756231338525945, 0.00015223432221797662, -6.1698246877700525e-5, 2.5005357609459247e-5, -1.0134289721572026e-5, 4.1072729198567004e-6, -1.664615015128028e-6, 6.7464305456632013e-7, -2.7342253129896734e-7, 1.1081397802278207e-7, -4.4911213669537136e-8, 1.820182931124577e-8, -7.3769235610848765e-9, 2.9897545073926764e-9, -1.2117018619561619e-9, 4.9108426750009155e-10, -1.9902895700495881e-10, 8.066339801136952e-11, -3.2691643852499818e-11, 1.3249424201396147e-11, -5.3697893706595739e-12, 2.1762936597810882e-12, -8.8201859824934e-13, 3.5746867347671487e-13, -1.4487659644686842e-13, 5.871627293628006e-14, -2.3796809091881837e-14, 9.6444834564008509e-15, -3.9087619176859166e-15, 1.5841615363041244e-15, -6.4203648775599251e-16, 2.6020758752402585e-16, -1.0545816304260601e-16, 4.2740583616893926e-17, -1.7322106086512011e-17, 7.0203851674541611e-18, -2.8452549391662691e-18]   # sech th = sum GE[k] q^{k-1}
polyq(cf, from, q) = (acc = zero(q); for k in length(cf):-1:from; acc = acc * q + cf[k]; end; acc)
const KDU = [1.0 -1.0; -1.0 0.0]
const KDT = [1.0 2.0 -6.0; 2.0 -12.0 12.0; -6.0 12.0 0.0]
adjK(model) = model == 2 ? [0.0 1.0; 1.0 1.0] : [-144.0 -72.0 -48.0; -72.0 -36.0 -24.0; -48.0 -24.0 -16.0]
det3(M) = M[1,1] * (M[2,2] * M[3,3] - M[2,3] * M[3,2]) - M[1,2] * (M[2,1] * M[3,3] - M[2,3] * M[3,1]) + M[1,3] * (M[2,1] * M[3,2] - M[2,2] * M[3,1])
function adj3(M)
    A = similar(M)
    A[1,1] = M[2,2] * M[3,3] - M[2,3] * M[3,2]; A[1,2] = -(M[1,2] * M[3,3] - M[1,3] * M[3,2]); A[1,3] = M[1,2] * M[2,3] - M[1,3] * M[2,2]
    A[2,1] = -(M[2,1] * M[3,3] - M[2,3] * M[3,1]); A[2,2] = M[1,1] * M[3,3] - M[1,3] * M[3,1]; A[2,3] = -(M[1,1] * M[2,3] - M[1,3] * M[2,1])
    A[3,1] = M[2,1] * M[3,2] - M[2,2] * M[3,1]; A[3,2] = -(M[1,1] * M[3,2] - M[1,2] * M[3,1]); A[3,3] = M[1,1] * M[2,2] - M[1,2] * M[2,1]
    A
end
"""eta = th coth th - c (= s_1 in case DK), th = (c^2 - 2t)^{1/2}, without cancellation at large c."""
function eta1(c::T, t::Complex{T}) where {T}
    mu = sqrt(c * c - 2t)
    abs(mu) < 1e-4 && return 1 + mu * mu / 3 - c
    e2 = exp(-2mu)
    -2t / (mu + c) + 2mu * e2 / (1 - e2)
end
"""The GLS detrending of Elliott, Rothenberg and Stock (1996), trend case: cbar and lambda = (1 - cbar)/(1 - cbar + cbar^2/3)."""
const CBAR_GLS = -13.5
lam_gls(cb) = (1 - cb) / (1 - cb + cb * cb / 3)
const VGLS = [1.0, 0.0, -3.0]                                             # V_cbar(1) = (1 - lam) (W(1) - 3 int r W)

"""
    girsanov(model, c, t)

The Girsanov forms at the point t (any complex t with cosh th != 0): th, lch = log cosh th, lsh = log(sinh th/th),
eta (case DK), and for models 2, 3 the covariance Sig, N = I - Q Sig, D = det N, dS = det Sig, and A2, A1, A0.
"""
function girsanov(model::Int, c::T, t::Complex{T}) where {T}
    q = c * c - 2t; th = sqrt(q)
    del = c > 0 ? -2t / (th + c) : th - c
    e2 = exp(-2th); em = exp(-th)
    if abs(th) < 1
        g1 = polyq(GT, 1, q); g2 = -polyq(GE, 2, q); g3 = -polyq(GT, 2, q); g4 = polyq(GE, 3, q); g5 = polyq(GT, 3, q)
        t1 = 1 - th * g1; lch = log(cosh(th))
        r = one(q); tm = one(q); ch = one(q)
        for k in 1:14; tm *= q / ((2k) * (2k + 1)); r += tm; end
        tm = one(q); for k in 1:14; tm *= q / ((2k - 1) * (2k)); ch += tm; end
        lsh = log(r); ep = ch / r - th
        N11 = 1 - c * g1
    else
        tn = (1 - e2) / (1 + e2); sc = 2em / (1 + e2)
        g1 = tn / th; g2 = (1 - sc) / th^2; g3 = (th - tn) / th^3; g4 = (th^2 - 2 + 2sc) / (2th^4); g5 = (th^3 / 3 - th + tn) / th^5
        t1 = 2 * e2 / (1 + e2); lch = th + log((1 + e2) / 2)
        ep = 2th * e2 / (1 - e2); lsh = th + log((1 - e2) / 2) - log(th)
        N11 = (del + c * t1) / th
    end
    eta = del + ep
    if model == 1
        return (th = th, del = del, eta = eta, lsh = lsh, lch = lch, Sg = Matrix{Complex{T}}(undef, 0, 0), N = Matrix{Complex{T}}(undef, 0, 0),
                dS = zero(q), D = zero(q), A2 = zero(q), A1 = zero(q), A0 = zero(q))
    elseif model == 2
        Sg = [g1 g2; g2 g3]; Qd = [c 0; 0 -2t]
        N = Matrix{Complex{T}}(I, 2, 2) - Qd * Sg; N[1,1] = N11
        dS = g1 * g3 - g2 * g2
        D = N11 * (1 + 2t * g3) + 2c * t * g2 * g2
        A2 = D / dS; A1 = (g1 - 2g2) / dS + 2t; A0 = -one(q)
    elseif model == 4
        # the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996) at cbar = CBAR_GLS: the detrended path is
        # J(r) - r L, L = lam J(1) + 3 (1 - lam) K, so that V_g = int W^2 - X' Gam_g X and R_g = (1 - lam)^2 (X_1 - 3 X_3)^2 / 2
        # (rank one), with X = (W(1), int W, int r W); the covariance under the tilt and the determinant as in case T
        lam = lam_gls(CBAR_GLS)
        Sg = [g1 g2 g3; g2 g3 g4; g3 g4 g5]
        Gm = [-lam * lam / 3 0 lam * lam; 0 0 0; lam * lam 0 3 * (1 - lam * lam)]
        Qd = Matrix{Complex{T}}([c 0 0; 0 0 0; 0 0 0]) .- 2t .* Gm
        N = Matrix{Complex{T}}(I, 3, 3) - Qd * Sg
        D = det3(N); dS = det3(Sg)
        A2 = D / dS; A1 = zero(q); A0 = zero(q)
    else
        Sg = [g1 g2 g3; g2 g3 g4; g3 g4 g5]; Qd = [c 0 0; 0 -8t 12t; 0 12t -24t]
        N = Matrix{Complex{T}}(I, 3, 3) - Qd * Sg; N[1,1] = N11
        D = det3(N); dS = det3(Sg)
        A2 = D / dS
        Si = adj3(Sg); Pp = Si ./ dS .- Qd
        Pp[1,1] = (N[1,1] * Si[1,1] + N[1,2] * Si[2,1] + N[1,3] * Si[3,1]) / dS     # (row 1 of N is cancellation-free)
        aP = adj3(Pp); aK = adjK(3)
        A1 = zero(q); A0 = zero(q)
        for i in 1:3, j in 1:3
            A1 += KDT[i,j] * aP[j,i]; A0 += aK[i,j] * Pp[j,i]
        end
    end
    (th = th, del = del, eta = eta, lsh = lsh, lch = lch, Sg = Sg, N = N, dS = dS, D = D, A2 = A2, A1 = A1, A0 = A0)
end

# ---------- the convergence abscissa in t ----------
"""
    tstar(model, c)

t_* = sup{t : E exp(tV) < inf}, the first zero of D(0, t) on the positive real axis (D(0, 0) = 1).
The line Re t = tau of the outer integrals must lie in (0, t_*) for M2 and M3: at t_* the root s_2
passes through zero and (-s_1 s_2)^{-1/2} has a branch point.  t_* exceeds 1 for c <= 0 (1.23, 4.93
and 19.7 for M1, M2 and M3 at c = 0, increasing as c decreases) and tends to zero as c grows
(about 4e-7 at c = 10, 1e-13 at c = 20: for c > CGIRS the sign of the Girsanov determinant, or of eta
in case DK, replaces D(0, t), whose terms of size e^{2c} cancel to a value of order e^{-c}).
"""
const TSTAR = Dict{Tuple{Int,Float64,Bool},Float64}()
const TSTARLOCK = ReentrantLock()
function tstar(model::Int, c::Float64; stat::Bool = false)
    lock(TSTARLOCK) do; get!(TSTAR, (model, c, stat)) do
        # case DK: the sign of D = eta S from the cancellation-free eta and the sign of S = sinh(mu)/mu (sin(th)/th for
        # 2t > c^2); eta alone changes sign at the poles of coth as well, and for c < 0 the first zero of D lies just below
        # the pole (c^2 + pi^2)/2, inside a stretch that the 5% steps of the bracketing can skip (c = -9.5: 61.6 for 49.2)
        sgnS(t) = (q = c * c - 2t; q >= 0 ? 1.0 : sign(sin(sqrt(-q))))
        g(t) = (model == 4 && !stat) ? (G = girsanov(4, c, complex(t)); real(exp(G.lch) * G.D)) :   # GLS: the sign of cosh(th) |I - Q Sig|
               (model == 1 && !stat) ? real(eta1(c, complex(t))) * sgnS(t) :
               (!stat && c > CGIRS) ? real(girsanov(model, c, complex(t)).D) :
               real(Dfun(model, c, complex(0.0), complex(t); stat))
        a = 1e-6                                                 # t_* ~ e^{-2c}: start below it
        while g(a) <= 0 && a > 1e-300; a /= 1e3; end
        g(a) > 0 || error("D(0, t) not positive near t = 0")
        b = a; fac = model == 4 ? 1.005 : 1.05
        while g(b) > 0
            a = b; b *= fac
            b > 1e4 && error("no zero of D(0, t) below t = 1e4")
        end
        for _ in 1:100
            m = (a + b) / 2
            g(m) > 0 ? (a = m) : (b = m)
            b - a < 1e-15 * b && break
        end
        (a + b) / 2
    end; end
end

"""The default line Re t = tau: 1 for c <= 0, otherwise t_*/2 (at most 1)."""
default_tau(model::Int, c::Float64; stat::Bool = false) = (c <= 0 && !stat) ? 1.0 : min(1.0, tstar(model, c; stat) / 2)

# ---------- the inverse s-transform phi~(X), two branches ----------
# ---------- e^z K_0(z) for Re z >= 0, the arguments that occur (checked: none with Re z < 0) ----------
# |z| >= 25: the asymptotic series sqrt(pi/(2z)) sum_k a_k z^{-k}, a_k = -a_{k-1} (2k-1)^2/(8k), 20 terms (the
# first omitted one is below 1e-17); |z| < 2: the power series K_0 = -(log(z/2) + gamma) I_0 + sum (z^2/4)^k H_k/(k!)^2,
# 14 terms; in between: Taylor expansions of K_0 about the centres of a grid of cells of side 1/4, the
# coefficients from K_0, K_1 at the centre (AMOS) and the recurrence of Bessel's equation, 12 terms (the ratio
# |z - z_0|/|z_0| is at most 0.09).  All three agree with AMOS to 1e-14 or better; 25-40 ns against 120-290.
const K0A = let a = zeros(21); a[1] = 1.0; for k in 1:20; a[k+1] = -a[k] * (2k - 1)^2 / (8k); end; a end
const K0H = 0.25; const K0N = 12; const K0R = 25.0
function k0cell(z0::ComplexF64, N::Int = max(K0N, 14))
    c = Vector{ComplexF64}(undef, N + 1)
    c[1] = besselk(0, z0); c[2] = -besselk(1, z0)
    for n in 0:N-2                                     # z^2 y'' + z y' - z^2 y = 0 about z0
        cm1 = n >= 1 ? c[n] : zero(ComplexF64); cm2 = n >= 2 ? c[n-1] : zero(ComplexF64)
        c[n+3] = (z0 * z0 * c[n+1] + 2z0 * cm1 + cm2 - 2z0 * (n + 1) * n * c[n+2] - z0 * (n + 1) * c[n+2]
                  - n * (n - 1) * c[n+1] - n * c[n+1]) / (z0 * z0 * (n + 2) * (n + 1))
    end
    c
end
# the cells (i, j), i = 0..99, j = 0..99, of the square 0 <= Re z < 25, 0 <= Im z < 25, built once (2 ms); Im z < 0
# by K_0(conj z) = conj K_0(z).  The coefficients of a cell are contiguous: K0C[n, cell], cell = i + 1 + K0NI j
# (2 MB, which fits the cache of a core cluster)
const K0NI = 100
const K0C = Matrix{ComplexF64}(undef, K0N + 1, K0NI * 100)      # filled by k0tables!() in __init__ (the library routine
                                                                # that seeds the cells is not available at the top level of a
                                                                # juliac build)
@inline k0cellof(z::ComplexF64) = (i = floor(Int, real(z) / K0H); j = floor(Int, imag(z) / K0H);
                                   (i + 1 + K0NI * j, z - complex((i + 0.5) * K0H, (j + 0.5) * K0H)))
# the fine cells of side 1/8 for 1 <= |z| < 2 (i = 0..15, j = 0..15), 14 terms: the ratio |z - z0|/|z0| is at most 0.094
const K0HF = 0.125; const K0NF = 14
const K0F = Matrix{ComplexF64}(undef, K0NF + 1, 16 * 16)
const K0READY = Ref(false)
function k0tables!()
    K0READY[] && return
    for i in 0:K0NI-1, j in 0:99; K0C[:, i + 1 + K0NI * j] .= k0cell(complex((i + 0.5) * K0H, (j + 0.5) * K0H), K0N); end
    for i in 0:15, j in 0:15
        z0 = complex((i + 0.5) * K0HF, (j + 0.5) * K0HF)
        K0F[:, i + 1 + 16j] .= abs(z0) > 0.8 ? k0cell(z0)[1:K0NF+1] : zero(ComplexF64)   # (cells below |z0| = 0.8 are never used)
    end
    K0READY[] = true
end
@inline k0cellof_f(z::ComplexF64) = (i = floor(Int, real(z) / K0HF); j = floor(Int, imag(z) / K0HF);
                                     (i + 1 + 16j, z - complex((i + 0.5) * K0HF, (j + 0.5) * K0HF)))
@inline upper(z::ComplexF64) = imag(z) < 0 ? conj(z) : z
@inline back(a::ComplexF64, z::ComplexF64) = imag(z) < 0 ? conj(a) : a
const EULERGAMMA = 0.5772156649015329
@inline function k0x(z::ComplexF64)
    a = abs(z)
    if real(z) < 0
        return besselkx(0, z)
    elseif a >= K0R
        w = 1 / z; acc = K0A[21]
        @inbounds for k in 20:-1:1; acc = acc * w + K0A[k]; end
        return sqrt(pi / (2z)) * acc
    elseif a < 2
        q4 = z * z / 4; term = one(ComplexF64); I0 = term; S = zero(ComplexF64); H = 0.0
        for k in 1:14
            term *= q4 / (k * k); H += 1 / k
            I0 += term; S += term * H
        end
        return exp(z) * (-(log(z / 2) + EULERGAMMA) * I0 + S)
    else
        c, u = k0cellof(upper(z))
        acc = @inbounds K0C[K0N+1, c]
        @inbounds for n in K0N:-1:1; acc = muladd(acc, u, K0C[n, c]); end
        return exp(z) * back(acc, z)
    end
end
# work arrays of k0_batch! (one set per task: see ExpBuf)
struct K0Work
    r1::Vector{Float64}; r2::Vector{Float64}; r3::Vector{Float64}
    i1::Vector{Int32}; i2::Vector{Int32}; i3::Vector{Int32}; i4::Vector{Int32}
end
K0Work(n) = K0Work(zeros(n), zeros(n), zeros(n), zeros(Int32, n), zeros(Int32, n), zeros(Int32, n), zeros(Int32, n))
const K0INVJ2 = [1 / (j * j) for j in 1:14]; const K0HJ = [sum(1 / i for i in 1:j) for j in 1:14]
"""
    k0_batch!(out, shift, z, n, W)

For k in 1:n (Re z_k >= 0): out[k] e^{shift[k]} = e^{z_k} K_0(z_k), with out[k] = K_0(z_k) and shift[k] = z_k
when |z_k| < 25 (the power series below |z| = 1, the fine Taylor cells up to |z| = 2, the cells of side 1/4
beyond) and out[k] = e^{z_k} K_0(z_k), shift[k] = 0 from |z| = 25 on (the asymptotic series, 17 terms); the
caller adds shift[k] to its own exponent, so that no exponential is taken here.  The series run as four interleaved Horner chains over the points of each kind (the
independent chains hide the latency of the complex arithmetic); the power series' logarithms are real logs and
atan2s, vectorized where the vector library is there.
"""
function k0_batch!(out::Vector{ComplexF64}, shift::Vector{ComplexF64}, z::Vector{ComplexF64}, n::Int, W::K0Work)
    n1 = n2 = n3 = n4 = 0; i1 = W.i1; i2 = W.i2; i3 = W.i3; i4 = W.i4
    @inbounds for k in 1:n
        a = abs2(z[k])
        if a < 1; n1 += 1; i1[n1] = k
        elseif a < 4; n2 += 1; i2[n2] = k
        elseif a < K0R * K0R; n3 += 1; i3[n3] = k
        else; n4 += 1; i4[n4] = k; end
        shift[k] = z[k]
    end
    # Taylor cells, four chains
    m = 1
    @inbounds while m + 3 <= n3
        k1 = i3[m]; k2 = i3[m+1]; k3 = i3[m+2]; k4 = i3[m+3]
        c1, u1 = k0cellof(upper(z[k1])); c2, u2 = k0cellof(upper(z[k2])); c3, u3 = k0cellof(upper(z[k3])); c4, u4 = k0cellof(upper(z[k4]))
        a1 = K0C[K0N+1, c1]; a2 = K0C[K0N+1, c2]; a3 = K0C[K0N+1, c3]; a4 = K0C[K0N+1, c4]
        for nn in K0N:-1:1
            a1 = muladd(a1, u1, K0C[nn, c1]); a2 = muladd(a2, u2, K0C[nn, c2]); a3 = muladd(a3, u3, K0C[nn, c3]); a4 = muladd(a4, u4, K0C[nn, c4])
        end
        out[k1] = back(a1, z[k1]); out[k2] = back(a2, z[k2]); out[k3] = back(a3, z[k3]); out[k4] = back(a4, z[k4])
        m += 4
    end
    @inbounds while m <= n3
        k1 = i3[m]; c1, u1 = k0cellof(upper(z[k1])); a1 = K0C[K0N+1, c1]
        for nn in K0N:-1:1; a1 = muladd(a1, u1, K0C[nn, c1]); end
        out[k1] = back(a1, z[k1]); m += 1
    end
    # the fine cells
    m = 1
    @inbounds while m + 3 <= n2
        k1 = i2[m]; k2 = i2[m+1]; k3 = i2[m+2]; k4 = i2[m+3]
        c1, u1 = k0cellof_f(upper(z[k1])); c2, u2 = k0cellof_f(upper(z[k2])); c3, u3 = k0cellof_f(upper(z[k3])); c4, u4 = k0cellof_f(upper(z[k4]))
        a1 = K0F[K0NF+1, c1]; a2 = K0F[K0NF+1, c2]; a3 = K0F[K0NF+1, c3]; a4 = K0F[K0NF+1, c4]
        for nn in K0NF:-1:1
            a1 = muladd(a1, u1, K0F[nn, c1]); a2 = muladd(a2, u2, K0F[nn, c2]); a3 = muladd(a3, u3, K0F[nn, c3]); a4 = muladd(a4, u4, K0F[nn, c4])
        end
        out[k1] = back(a1, z[k1]); out[k2] = back(a2, z[k2]); out[k3] = back(a3, z[k3]); out[k4] = back(a4, z[k4])
        m += 4
    end
    @inbounds while m <= n2
        k1 = i2[m]; c1, u1 = k0cellof_f(upper(z[k1])); a1 = K0F[K0NF+1, c1]
        for nn in K0NF:-1:1; a1 = muladd(a1, u1, K0F[nn, c1]); end
        out[k1] = back(a1, z[k1]); m += 1
    end
    # power series, 10 terms (|z| < 1): K_0 = -(log(z/2) + gamma) I_0 + S, four chains, then the logarithms
    if n1 > 0
        m = 1
        @inbounds while m + 3 <= n1
            k1 = i1[m]; k2 = i1[m+1]; k3 = i1[m+2]; k4 = i1[m+3]
            z1 = z[k1]; z2 = z[k2]; z3 = z[k3]; z4 = z[k4]
            q1 = z1 * z1 / 4; q2 = z2 * z2 / 4; q3 = z3 * z3 / 4; q4 = z4 * z4 / 4
            t1 = t2 = t3 = t4 = one(ComplexF64); I1 = I2 = I3 = I4 = one(ComplexF64); S1 = S2 = S3 = S4 = zero(ComplexF64)
            for j in 1:10
                inv = K0INVJ2[j]; H = K0HJ[j]
                t1 = t1 * q1 * inv; t2 = t2 * q2 * inv; t3 = t3 * q3 * inv; t4 = t4 * q4 * inv
                I1 += t1; I2 += t2; I3 += t3; I4 += t4
                S1 += t1 * H; S2 += t2 * H; S3 += t3 * H; S4 += t4 * H
            end
            out[k1] = I1; shift[k1] = S1; out[k2] = I2; shift[k2] = S2; out[k3] = I3; shift[k3] = S3; out[k4] = I4; shift[k4] = S4
            m += 4
        end
        @inbounds while m <= n1
            k1 = i1[m]; z1 = z[k1]; q1 = z1 * z1 / 4; t1 = one(ComplexF64); I1 = one(ComplexF64); S1 = zero(ComplexF64)
            for j in 1:10; t1 = t1 * q1 * K0INVJ2[j]; I1 += t1; S1 += t1 * K0HJ[j]; end
            out[k1] = I1; shift[k1] = S1; m += 1
        end
        # log(z/2) + gamma = (log|z|^2)/2 - log 2 + gamma + i atan2(Im z, Re z)
        la = W.r1; lb = W.r2; lc = W.r3
        @inbounds for m in 1:n1; zk = z[i1[m]]; la[m] = abs2(zk); lb[m] = imag(zk); lc[m] = real(zk); end
        if HAVE_ACCEL[]
            ccall((:vvlog, ACCEL), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ref{Int32}), la, la, Int32(n1))
            ccall((:vvatan2, ACCEL), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}), lb, lb, lc, Int32(n1))
        else
            @inbounds for m in 1:n1; la[m] = log(la[m]); lb[m] = atan(lb[m], lc[m]); end
        end
        @inbounds for m in 1:n1
            kk = i1[m]
            lg = complex(0.5 * la[m] - 0.6931471805599453 + EULERGAMMA, lb[m])
            out[kk] = shift[kk] - lg * out[kk]; shift[kk] = z[kk]
        end
    end
    # asymptotic series, 17 terms (the first omitted one is below 1e-16 at |z| = 25), Horner in w = 1/z, four
    # chains; sqrt(pi/(2z)) = sqrt(pi/2) conj(sqrt z)/|z|, sqrt z = p + i y/(2p), p = sqrt((|z| + x)/2) (Re z >= 0)
    if n4 > 0
        m = 1
        @inbounds while m + 3 <= n4
            k1 = i4[m]; k2 = i4[m+1]; k3 = i4[m+2]; k4 = i4[m+3]
            w1 = 1 / z[k1]; w2 = 1 / z[k2]; w3 = 1 / z[k3]; w4 = 1 / z[k4]
            a1 = a2 = a3 = a4 = complex(K0A[18])
            for j in 17:-1:1
                cj = K0A[j]
                a1 = muladd(a1, w1, cj); a2 = muladd(a2, w2, cj); a3 = muladd(a3, w3, cj); a4 = muladd(a4, w4, cj)
            end
            out[k1] = a1; out[k2] = a2; out[k3] = a3; out[k4] = a4
            m += 4
        end
        @inbounds while m <= n4
            k1 = i4[m]; w1 = 1 / z[k1]; a1 = complex(K0A[18])
            for j in 17:-1:1; a1 = muladd(a1, w1, K0A[j]); end
            out[k1] = a1; m += 1
        end
        @inbounds for m in 1:n4
            kk = i4[m]; zk = z[kk]; x = real(zk); y = imag(zk); r = sqrt(x * x + y * y)
            p = sqrt((r + x) / 2); qq = y / (2p); f = 1.2533141373155003 / r
            out[kk] *= complex(f * p, -f * qq); shift[kk] = zero(ComplexF64)
        end
    end
    out
end
phi_pos(::Val{1}, X, s1, s2) = exp(-s1 * X) / sqrt(pi * X)
phi_pos(::Val, X, s1, s2) = k0x((s2 - s1) / 2 * X) * exp(-s2 * X) / pi      # K_0(bX) e^{-(s1+s2)X/2}
phi_neg(::Val{1}, X, s1, s2) = zero(X)
phi_neg(::Val, X, s1, s2) = k0x(-(s2 - s1) / 2 * X) * exp(-s1 * X) / pi
# phi_pos without its exponential factor e^{-sd X}, sd = s1 (M1) or s2: algebraic off the support edge
phi_alg(::Val{1}, X, s1, s2) = 1 / sqrt(pi * X)
phi_alg(::Val, X, s1, s2) = k0x((s2 - s1) / 2 * X) / pi

"""Gauss–Legendre nodes and weights on [0, 1] (Golub–Welsch)."""
function gl(n::Int)
    b = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), b))
    x = (E.values .+ 1) ./ 2; w = (2 .* E.vectors[1, :] .^ 2) ./ 2
    x, w
end
const RULES = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const RULELOCK = ReentrantLock()
rule(n::Int) = lock(RULELOCK) do; get!(() -> gl(n), RULES, n); end     # (the caches are read from several tasks)

const EXPO = 60.0          # the Gaussian factors are followed down to e^{-EXPO}

# ---------- the inner integrals: the kernel transforms of the tilted density ----------
#
# The inverse transform phi_t(X), X = x + 1/2, is integrated against the kernels of the density and
# the distribution function formulas along paths in the complex plane on which the Gaussian factor
# decays.  Each inner integral is split into a part that is smooth in t and, for q < 0, an "edge"
# part, the contributions of a neighbourhood of the support edge X = 0 of phi_t, which carry the
# factor e^{-t v*^2}, v* = 1/(2|q|), and so oscillate in Im t.  The two parts are returned
# separately, because the outer integral treats them differently.  The split points are fixed
# (independent of t), so that each part is an analytic function of t; only the quadrature nodes
# inside a part adapt to t.  Node counts are set so that every part is converged to about 1e-13
# relative to the whole (checked by doubling, see the log).

# Gauss–Legendre on [a, b] with the substitution y = a + (b - a) u^p (p = 1, 2 or 4), which removes
# algebraic and logarithmic endpoint singularities at a.
const NODEFAC = Ref(1)      # multiplies every node count (set to 2 to check convergence)
@inline function glsum(f, a, b, n, p)
    u, gw = rule(n * NODEFAC[])
    n = n * NODEFAC[]
    acc = zero(ComplexF64)
    L = b - a
    @inbounds for k in 1:n
        if p == 1
            y = a + L * u[k]; dy = L
        elseif p == 2
            y = a + L * u[k]^2; dy = 2L * u[k]
        else
            y = a + L * u[k]^4; dy = 4L * u[k]^3
        end
        acc += f(y) * dy * gw[k]
    end
    acc
end

# The smooth/edge split for q < 0 is by an entire partition of unity chi = (1 + erf(kappa (m - .)))/2
# of the segment, m its midpoint and kappa such that chi is 1 and 0 at the ends to 1e-16.  Both parts
# are then path-independent integrals of analytic functions with the same real endpoints, hence
# analytic and conjugate symmetric in t, and only the edge part carries the oscillating factor
# e^{-t v*^2}.  The segment is bulged into the half-plane where the Gaussian factor decays, with unit
# slope at both ends (steepest descent there) and height H = 1.5/kappa, so that |chi| <= e^{2.25}
# on the path.
const KAPPA = 6.0           # kappa = KAPPA / (half-length): erfc(6)/2 = 1e-17 at the ends
cap(x) = x / sqrt(1 + x * x)                                   # bulge profile: cap(x) ~ x near 0, -> 1
# the Gaussian ray from r = 0 (the kink or the support edge) with the scales of the other factors
# resolved: a first piece on the scale `l` of the exponential and algebraic factors, a second one on
# which the algebraic factors (X^{-1/2}, the tail of K_0) still vary, then the rest; `osc` is the
# phase rate (per unit r) of the exponential factors along the ray, and the node counts of the
# outer pieces keep at most 0.85 rad per node.
nodes(n, osc, L) = max(n, 16 * cld(ceil(Int, 1.2 * osc * L), 16))
@inline function rayint(f, R, l, n1, n2, p; osc = 0.0)
    a = min(R, 8l)
    acc = glsum(f, 0.0, a, n1, p)
    while a < R                                                 # geometric panels of ratio 4: the algebraic factors
        b = min(R, 4a)                                          # are resolved on every scale up to R
        acc += glsum(f, a, b, nodes(n2, osc, b - a), 1)
        a = b
    end
    acc
end

"""
    gaussexp(A, t, q, sd, lA)

int_0^inf A(x) exp(-t x^2/q^2 - sd x) dx for A algebraic, varying on the scale lA, along a path on which
neither exponential factor grows and the phase stays bounded.  The ray x = r e^{-i arg(t)/2}, on which
the Gaussian decays, serves while the exponential decays or oscillates little along it (the callers
check); otherwise, with x0 = -sd q^2/(2t) the saddle of the exponent, u0 = -sd^2 q^2/(4t) = -z^2 its
depth and z = sd q/(2 sqrt t), the path is the segment from 0 to x0 and the steepest descent line
x0 + r e^{-i arg(t)/2} when Re z < 0 (the exponent is -u0 lam (2 - lam) on the segment and
-u0 - |t| r^2/q^2 on the line; the model integral with A = 1 is (q/2) sqrt(pi/t) exp(z^2) erfc(z),
which contains the saddle term 2 exp(z^2) exactly for Re z < 0, and the level curve from 0 then ends
in the other valley), and otherwise the level curve Im(t x^2/q^2 + sd x) = 0 from 0, on which the
exponent is -u for u = v^2 real: x = 2v^2/(sd + rt), rt = sqrt(sd^2 + 4 t v^2/q^2), dx = 2v dv/rt,
the principal branch continuous and ending in the right valley precisely when Re z >= 0.  The level
curve has the saddle at v = sqrt(u0), at an angle of at least pi/4 from the path (Re u0 <= 0 there,
or Re z = 0); v = vmin sinh(s) puts the scales of the factors on an equal footing and 13 nodes per
unit of s resolve them.  (Both paths stay in the upper half plane until they cross the positive real
axis, away from the branch cut of K_0 in the lower left quadrant.)
"""
function gaussexp(A, t::ComplexF64, q::Float64, sd::ComplexF64, lA::Float64; force::Int = 0, p::Int = 1)
    at = abs(t); e = exp(-im * angle(t) / 2)
    x0 = -sd * q * q / (2t); u0 = -sd * sd * q * q / (4t)
    if force == 1 || (force == 0 && real(sd * e) < 0)              # Re z < 0, z = sd q/(2 sqrt t): the saddle term is present
        ru = real(u0)
        lam_max = ru > EXPO ? 1 - sqrt(1 - EXPO / ru) : 1.0
        f1 = lam -> A(x0 * lam) * exp(-u0 * lam * (2 - lam)) * x0
        l = min(lA / abs(x0), 1 / (2abs(u0) + 1))
        acc = rayint(f1, lam_max, l, 48, 32, 1; osc = 2abs(imag(u0)))
        ru > EXPO && return acc
        f2 = r -> A(x0 + r * e) * exp(-u0 - at * r * r / (q * q)) * e
        return acc + rayint(f2, q * sqrt(EXPO / at), lA, 48, 32, 1)
    end
    v0 = abs(sd) * q / (2sqrt(at)); vA = sqrt(lA * abs(sd))
    vmin = max(min(v0, vA, 1.0) / 2, 1e-4)
    S = asinh(sqrt(EXPO) / vmin)
    f = s -> begin
        v = vmin * sinh(s); dv = vmin * cosh(s)
        rt = sd * sqrt(1 + 4t * v * v / (q * q * sd * sd))
        x = 2v * v / (sd + rt)
        A(x) * exp(-v * v) * 2v / rt * dv
    end
    p == 1 && return glsum(f, 0.0, S, 16 * cld(ceil(Int, 13S), 16), 1)
    S1 = min(1.0, S)                                                # p = 4: an algebraic or logarithmic singularity of A at x = 0
    acc = glsum(f, 0.0, S1, 32, p)
    S > 1 && (acc += glsum(f, 1.0, S, 16 * cld(ceil(Int, 13(S - 1)), 16), 1))
    acc
end

# the ray x = r e^{-i arg t/2} is kept when the exponential e^{-sd x} grows along it by at most e^2 and
# turns by at most 60 rad over the Gaussian's range
ray_ok(sd, e, q, at) = max(0.0, -real(sd * e))^2 * q * q / (4at) <= 2 && abs(imag(sd * e)) * q * sqrt(EXPO / at) <= 60

"""
    inner_density(model, t, q, s1, s2, M0) -> (smooth, edge)

I_f(t) = int_0^inf 2 v^2 e^{-t v^2} phi_t(q v + 1/2) dv.  For q > 0 the ray v = r e^{-i arg(t)/2}
(all smooth).  For q < 0 the bulged segment [0, v*], v* = 1/(2|q|), split by the partition into the
smooth part (the Gaussian bump at v = 0) and the edge part (the neighbourhood of X = 0), plus, for
M2 and M3, the ray beyond v* (X < 0), also edge.
"""
inner_density(model::Int, args...; kw...) = inner_density(Val(model), args...; kw...)
function inner_density(V::Val{model}, tj::ComplexF64, q::Float64, s1::ComplexF64, s2::ComplexF64, M0::ComplexF64 = zero(ComplexF64); part::Int = 0) where {model}
    th = angle(tj) / 2; at = abs(tj); e = exp(-im * th); w = abs(imag(tj))
    vs = 1 / (2abs(q))
    if q > 0 || 3 * real(tj) * vs * vs / 16 > 40                      # (for q < 0: the edge part is negligible, only the bump at v = 0 remains)
        part == 2 && return zero(ComplexF64), zero(ComplexF64)
        sd = model == 1 ? s1 : s2                                     # decay rate of phi_pos, X -> +inf
        lA = 1 / (abs(model == 1 ? s1 : (s2 - s1) / 2) + 2)           # scale of X^{-1/2} or of the scaled K_0, in X
        if q < 0 || ray_ok(sd, e, q, at)
            R = q > 0 ? sqrt(EXPO / at) : min(vs, sqrt(EXPO / at))
            f = r -> (v = r * e; 2 * v * v * exp(-at * r * r) * phi_pos(V, q * v + 0.5, s1, s2) * e)
            return rayint(f, R, lA / abs(q), 48, 32, 1; osc = abs(q) * abs(imag(sd * e))), zero(ComplexF64)
        end
        A = x -> 2 * x * x / q^3 * phi_alg(V, 0.5 + x, s1, s2) * exp(-sd / 2)   # (x = q v)
        return gaussexp(A, tj, q, sd, lA), zero(ComplexF64)
    end
    sg = imag(tj) >= 0 ? 1.0 : -1.0
    kap = KAPPA / (vs / 2); H = 1.5 / kap
    pathd = d -> begin                                                 # parametrized by d = v* - r, so that X = |q| d is exact near the edge
        r = vs - d
        a = cap(r / H); b = cap(d / H)
        g = H * a * b                                                  # the bulge: v = r - i g sg
        dg = b / sqrt(1 + (r / H)^2)^3 - a / sqrt(1 + (d / H)^2)^3     # d g / d r
        v = r - im * g * sg
        dv = 1 - im * dg * sg
        X = abs(q) * (d + im * g * sg)                                 # = q v + 1/2 = |q| (v* - v)
        2 * v * v * exp(-tj * v * v) * phi_pos(V, X, s1, s2) * dv
    end
    chi = d -> (1 + erf(kap * (vs / 2 - d - im * H * cap((vs - d) / H) * cap(d / H) * sg))) / 2   # chi(v) = (1 + erf(kappa (v* - v - v*/2)))/2: 1 near the edge
    Is = Ie = zero(ComplexF64)
    if part != 2
        # smooth part: (1 - chi) F, the Gaussian bump at r = 0 of width ~ (2w)^{-1/2}
        fs = r -> pathd(vs - r) * (1 - chi(vs - r))
        r1 = w > 0 ? min(vs, sqrt(EXPO / (1.9w))) : vs
        Is = glsum(fs, 0.0, r1, 64, 1)
        r1 < vs && (Is += glsum(fs, r1, vs, 48, 1))
    end
    part == 1 && return Is, Ie
    # edge part: chi F, parametrized by d; magnitude ~ e^{-2 w v* d} near d = 0
    fe = d -> pathd(d) * chi(d)
    delta = w > 0 ? min(vs, EXPO / (1.9w * vs)) : vs
    Ie = glsum(fe, 0.0, delta / 8, 48, 4) + glsum(fe, delta / 8, delta, 64, 1)   # the singularity, then the damped oscillation
    delta < vs && (Ie += glsum(fe, delta, vs, 48, 1))
    if model != 1
        rho = real((s2 - 3s1) / 2 * e) * abs(q)                        # decay rate of phi_neg along the ray
        R = min(sqrt(EXPO / at), EXPO / (2at * vs * cos(th)))             # the Gaussian and its linear term e^{-2 t v* r e}
        rho > 0 && (R = min(R, EXPO / rho))
        f = r -> (v = vs + r * e; X = -abs(q) * r * e; 2 * v * v * exp(-tj * v * v) * phi_neg(V, X, s1, s2) * e)
        Ie += rayint(f, R, 1 / (rho + 2at * vs * cos(th)), 64, 32, 4; osc = abs(q) * abs(imag(s1 * e)) + 2at * vs * abs(sin(th)))
    end
    Is, Ie
end

"""
    inner_cdf(model, t, q, s1, s2, M0) -> (smooth, edge)

I_F(t) = int phi_t(X) W_q(X - 1/2, t) dX without the factor 1/t.  Smooth: the constant part of
W_q (closed form for M1, the total mass minus a short integral for M2, M3), and the Gaussian part on
the ray from X = 1/2 (q > 0) or, for q < 0, the part of the Gaussian integral over the bulged
segment (0, 1/2) near X = 1/2.  Edge (q < 0): the part near X = 0 and, for M2 and M3, the ray X < 0.
"""
inner_cdf(model::Int, args...; kw...) = inner_cdf(Val(model), args...; kw...)
function inner_cdf(V::Val{model}, tj::ComplexF64, q::Float64, s1::ComplexF64, s2::ComplexF64, M0::ComplexF64 = zero(ComplexF64); part::Int = 0) where {model}
    th = angle(tj) / 2; at = abs(tj); e = exp(-im * th); w = abs(imag(tj))
    Is = Ie = zero(ComplexF64)
    vs2 = 1 / (4q * q)
    if part != 2
        # (i) the part of W_q that is constant in x
        if model == 1
            z = sqrt(s1 / 2)
            Is = erf(z) / sqrt(s1)
        else
            l = 1 / (abs(s2 - s1) + 2)                                # the scale on which K_0 reaches its exponential form
            if abs(s2) >= abs(s1) / 4                                 # total mass minus the integral over X > 1/2, where phi_pos ~ e^{-s2 X},
                e2 = exp(-im * angle(s2))                             # on the ray X = 1/2 + r e2 on which it decays without oscillating
                Is = M0 - rayint(r -> phi_pos(V, 0.5 + r * e2, s1, s2) * e2, 36 / abs(s2), l, 40, 32, 1)
            else                                                      # |s2| small (c > 0, tau near t_*; the ray above would be long): the mass over
                e1 = exp(-im * angle(-s1))                            # X < 1/2 directly, with the ray X = -r e1 on which phi_neg ~ e^{-s1 X} decays
                Is = rayint(X -> phi_pos(V, X, s1, s2), 0.5, l, 40, 32, 4) +
                     rayint(r -> phi_neg(V, -r * e1, s1, s2) * e1, 36 / abs(s1), l, 40, 32, 4)
            end
        end
    end
    # (ii) the Gaussian part
    if q > 0 || 3 * real(tj) * vs2 / 16 > 40
        part == 2 && return Is, Ie
        sd = model == 1 ? s1 : s2
        lA = 1 / (abs(model == 1 ? s1 : (s2 - s1) / 2) + 2)
        if q < 0 || ray_ok(sd, e, q, at)
            R = q > 0 ? abs(q) * sqrt(EXPO / at) : min(0.5, abs(q) * sqrt(EXPO / at))
            f = r -> (X = 0.5 + sign(q) * r * e; phi_pos(V, X, s1, s2) * exp(-at * r * r / (q * q)) * sign(q) * e)
            return Is + rayint(f, R, lA, 48, 32, 1; osc = abs(imag(sd * e))), Ie
        end
        A = x -> phi_alg(V, 0.5 + x, s1, s2) * exp(-sd / 2)
        return Is + gaussexp(A, tj, q, sd, lA), Ie
    end
    sg = imag(tj) >= 0 ? 1.0 : -1.0
    kap = KAPPA / 0.25; H = 1.5 / kap
    path = Xr -> begin
        a = cap(Xr / H); b = cap((0.5 - Xr) / H)
        g = H * a * b
        dg = b / sqrt(1 + (Xr / H)^2)^3 - a / sqrt(1 + ((0.5 - Xr) / H)^2)^3
        X = Xr + im * g * sg
        dX = 1 + im * dg * sg
        x = X - 0.5
        -phi_pos(V, X, s1, s2) * exp(-tj * x * x / (q * q)) * dX
    end
    chi = Xr -> (1 + erf(kap * (0.25 - Xr - im * H * cap(Xr / H) * cap((0.5 - Xr) / H) * sg))) / 2   # chi(X): 1 near X = 0
    if part != 2
        # smooth part: (1 - chi) times the integrand, the bump at X = 1/2 (x = 0) of width ~ |q| (2w)^{-1/2}
        fs = y -> path(0.5 - y) * (1 - chi(0.5 - y))                 # y = 1/2 - Xr
        eps_ = w > 0 ? min(0.5, abs(q) * sqrt(EXPO / (1.9w))) : 0.5
        Is += glsum(fs, 0.0, eps_, 64, 1)
        eps_ < 0.5 && (Is += glsum(fs, eps_, 0.5, 48, 1))
    end
    part == 1 && return Is, Ie
    # edge part: chi times the integrand; magnitude ~ e^{-w X / q^2} near X = 0
    fe = Xr -> path(Xr) * chi(Xr)
    delta = w > 0 ? min(0.5, EXPO * q * q / (0.95w)) : 0.5
    Ie = glsum(fe, 0.0, delta / 8, 48, 4) + glsum(fe, delta / 8, delta, 64, 1)   # the singularity, then the damped oscillation
    delta < 0.5 && (Ie += glsum(fe, delta, 0.5, 48, 1))
    if model != 1
        rho = real((s2 - 3s1) / 2 * e)                                 # decay rate of phi_neg along the ray
        R = min(abs(q) * sqrt(EXPO / at), EXPO * q * q / (at * cos(th)))  # the Gaussian and its linear term e^{-t r e / q^2}
        rho > 0 && (R = min(R, EXPO / rho))
        f = r -> (X = -r * e; x = X - 0.5; -phi_neg(V, X, s1, s2) * exp(-tj * x * x / (q * q)) * e)
        Ie += rayint(f, R, 1 / (rho + at * cos(th) / (q * q)), 64, 32, 4; osc = abs(imag(s1 * e)) + at * abs(sin(th)) / (q * q))
    end
    Is, Ie
end

# ---------- batched evaluation of the chi-square kernel (M1, zero start, c <= 0): fixed node layouts ----------
# With node layouts that do not depend on t, every piece of the M1 transforms is a sum over nodes of a cached
# weight C_k times a single complex exponential exp(A_k t + B_k s_1(t)): the Gaussian factor and the
# exponential of the kernel combine, and the algebraic factors, the partition and the Jacobians are cached.
# The layouts are geometric panels that resolve every scale that occurs along the line (the narrowest
# Gaussian width is |q|(2 w)^{-1/2} at w = WMAX), with Gauss-Legendre nodes per panel.  The exponentials of
# all nodes are taken at once, by Accelerate's vForce on macOS and by a plain loop elsewhere.
using Libdl
const ACCEL = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
const HAVE_ACCEL = Ref(false)                                  # set in __init__: Accelerate's vForce on macOS
# IntelVectorMath (MKL) where available: x86-64 Linux and Windows; loaded at run time if it is in the environment
const HAVE_IVM = Ref(false)
const IVM_EXP = Ref{Any}(nothing); const IVM_SIN = Ref{Any}(nothing); const IVM_COS = Ref{Any}(nothing)
function load_ivm()
    HAVE_ACCEL[] && return
    (Sys.ARCH == :x86_64 && !Sys.isapple()) || return
    try
        m = Base.require(Main, :IntelVectorMath)
        IVM_EXP[] = getfield(m, :exp!); IVM_SIN[] = getfield(m, :sin!); IVM_COS[] = getfield(m, :cos!)
        HAVE_IVM[] = true
    catch
        HAVE_IVM[] = false
    end
end
vecmath() = HAVE_ACCEL[] ? "Accelerate" : HAVE_IVM[] ? "IntelVectorMath" : "none"
struct ExpBuf
    a::Vector{Float64}; b::Vector{Float64}; e::Vector{Float64}; sn::Vector{Float64}; cs::Vector{Float64}
    z::Vector{ComplexF64}; k::Vector{ComplexF64}; sh::Vector{ComplexF64}; kw::K0Work
end
ExpBuf(n) = ExpBuf(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(ComplexF64, n), zeros(ComplexF64, n), zeros(ComplexF64, n), K0Work(n))
# a pool of buffers, so that the tasks of the batched loops allocate nothing: take one of at least n entries, give it back
const BUFPOOL = ExpBuf[]; const BUFLOCK = ReentrantLock()
function takebuf(n::Int)
    lock(BUFLOCK) do
        for (i, b) in enumerate(BUFPOOL)
            length(b.a) >= n && return splice!(BUFPOOL, i)
        end
        ExpBuf(max(n, 1024))
    end
end
givebuf(b::ExpBuf) = lock(BUFLOCK) do; push!(BUFPOOL, b); nothing; end
# sum_k C_k exp(A_k t + B_k s) for one (t, s), with the exponentials of all k at once
# the exponents z_k = A_k t + B_k s into buf.a, buf.b; returns the number of nodes above the cutoff
@inline function exponents!(A::Vector{ComplexF64}, B::Vector{ComplexF64}, t::ComplexF64, s::ComplexF64, buf::ExpBuf)
    n = length(A); neff = 0
    @inbounds @simd for k in 1:n
        z = A[k] * t + B[k] * s
        buf.a[k] = real(z); buf.b[k] = imag(z)
    end
    @inbounds for k in 1:n; buf.a[k] >= -CUTOFF[] && (neff = k); end    # the nodes are ordered from the dominant end
    neff
end
"""sum_k C_k K_0(bK X_k) exp(A_k t + B_k s): the K_0 kernel, the exponential of the kernel folded into the exponent."""
function expsumk(A::Vector{ComplexF64}, B::Vector{ComplexF64}, C::Vector{ComplexF64}, t::ComplexF64, s::ComplexF64, buf::ExpBuf, K::Vector{ComplexF64}, bK::ComplexF64)
    n = exponents!(A, B, t, s, buf)
    acc = zero(ComplexF64)
    @inbounds for k in 1:n; buf.z[k] = bK * K[k]; end
    k0_batch!(buf.k, buf.sh, buf.z, n, buf.kw)
    @inbounds @simd for k in 1:n; buf.a[k] += real(buf.sh[k]); buf.b[k] += imag(buf.sh[k]); end
    if HAVE_ACCEL[] || HAVE_IVM[]
        vexpsincos!(buf, n)
        @inbounds @simd for k in 1:n
            acc += C[k] * buf.k[k] * complex(buf.e[k] * buf.cs[k], buf.e[k] * buf.sn[k])
        end
    else
        @inbounds for k in 1:n
            acc += C[k] * buf.k[k] * exp(complex(buf.a[k], buf.b[k]))
        end
    end
    acc
end
function expsum(A::Vector{ComplexF64}, B::Vector{ComplexF64}, C::Vector{ComplexF64}, t::ComplexF64, s::ComplexF64, buf::ExpBuf)
    n = exponents!(A, B, t, s, buf)
    acc = zero(ComplexF64)
    if HAVE_ACCEL[] || HAVE_IVM[]
        vexpsincos!(buf, n)
        @inbounds @simd for k in 1:n
            acc += C[k] * complex(buf.e[k] * buf.cs[k], buf.e[k] * buf.sn[k])
        end
    else
        @inbounds for k in 1:n
            acc += C[k] * exp(complex(buf.a[k], buf.b[k]))
        end
    end
    acc
end

# e = exp(a), (sn, cs) = sincos(b) for the first n entries of the buffer
function vexpsincos!(buf::ExpBuf, n::Int)
    if HAVE_ACCEL[]
        ccall((:vvexp, ACCEL), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ref{Int32}), buf.e, buf.a, Int32(n))
        ccall((:vvsincos, ACCEL), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}), buf.sn, buf.cs, buf.b, Int32(n))
    else
        a = view(buf.a, 1:n); b = view(buf.b, 1:n)
        IVM_EXP[](view(buf.e, 1:n), a); IVM_SIN[](view(buf.sn, 1:n), b); IVM_COS[](view(buf.cs, 1:n), b)
    end
end

struct Layout
    A::Vector{ComplexF64}; B::Vector{ComplexF64}; C::Vector{ComplexF64}
    X::Vector{ComplexF64}       # for the K_0 kernel: K_0 is taken at b X_k (X > 0) or -b X_k (X < 0); empty for M1
    neg::Bool                   # the X < 0 branch (exponent in s_1 instead of s_2)
end
Layout(A, B, C) = Layout(A, B, C, ComplexF64[], false)
emptyL() = Layout(ComplexF64[], ComplexF64[], ComplexF64[])
# nodes and weights of geometric panels on [0, L]: the first panel [0, a1] with the substitution x = a1 u^p,
# then panels of ratio 2 up to L, n nodes each
const RATIO = Ref(4.0)        # growth factor of the geometric panels
function geopanels(a1::Float64, L::Float64, n::Int, p::Int; ratio::Float64 = RATIO[])
    u, gw = rule(n)
    xs = Float64[]; ws = Float64[]
    for k in 1:n
        if p == 1; push!(xs, a1 * u[k]); push!(ws, a1 * gw[k])
        else; push!(xs, a1 * u[k]^p); push!(ws, a1 * p * u[k]^(p - 1) * gw[k]); end
    end
    a = a1
    while a < L
        b = min(L, ratio * a)
        append!(xs, a .+ (b - a) .* u); append!(ws, (b - a) .* gw)
        a = b
    end
    xs, ws
end
const NPAN = Ref(32)          # nodes per panel of the M1 layouts
const NPANK = Ref(48)         # nodes per panel of the K_0-kernel layouts (smooth bump and edge segment)
const NPANR = Ref(32)         # ... of the rays (X < 0 and the constant part's ray over X > 1/2)
const CUTOFF = Ref(45.0)    # nodes whose exponent is below -CUTOFF are skipped (e^{-45} = 3e-20)
const E0 = cis(-pi / 8)      # fixed direction of the rays from X = 1/2 (q > 0): Re(t e0^2) >= |t| cos(pi/4) on the line
"""Layouts of the smooth and the edge part of the M1 transforms at q, for the distribution function (dens = false) or
the density."""
function layouts_m1(q::Float64, dens::Bool)
    sg = 1.0
    if q > 0
        R = abs(q) * sqrt(85.0)                                 # the Gaussian is spent beyond r = |q| sqrt(85/tau) (tau = 1)
        xs, ws = geopanels(0.001 * abs(q), R, NPAN[], 1)
        A = ComplexF64[]; B = ComplexF64[]; C = ComplexF64[]
        for (r, w) in zip(xs, ws)
            if dens
                v = r * E0 / abs(q); X = q * v + 0.5                 # r = |q| v
                push!(A, -v * v); push!(B, -X); push!(C, 2 * v * v / sqrt(pi * X) * E0 / abs(q) * w)
            else
                X = 0.5 + r * E0
                push!(A, -(r * E0)^2 / (q * q)); push!(B, -X); push!(C, 1 / sqrt(pi * X) * E0 * w)
            end
        end
        return Layout(A, B, C), emptyL()
    end
    if !dens
        kap = KAPPA / 0.25; H = 1.5 / kap
        path = Xr -> begin
            a = cap(Xr / H); b = cap((0.5 - Xr) / H); g = H * a * b
            dg = b / sqrt(1 + (Xr / H)^2)^3 - a / sqrt(1 + ((0.5 - Xr) / H)^2)^3
            X = Xr + im * g * sg; dX = 1 + im * dg * sg
            X, dX
        end
        chi = Xr -> (1 + erf(kap * (0.25 - Xr - im * H * cap(Xr / H) * cap((0.5 - Xr) / H) * sg))) / 2
        # smooth: y = 1/2 - Xr from 0, weight 1 - chi; edge: Xr from 0 with the X^{-1/2} singularity, weight chi
        ys, wy = geopanels(0.001 * abs(q), 0.5, NPAN[], 1)
        As = ComplexF64[]; Bs = ComplexF64[]; Cs = ComplexF64[]
        for (y, w) in zip(ys, wy)
            X, dX = path(0.5 - y); x = X - 0.5
            push!(As, -x * x / (q * q)); push!(Bs, -X); push!(Cs, -dX / sqrt(pi * X) * (1 - chi(0.5 - y)) * w)
        end
        xs, wx = geopanels(0.001 * abs(q), 0.5, NPAN[], 4)
        Ae = ComplexF64[]; Be = ComplexF64[]; Ce = ComplexF64[]
        for (Xr, w) in zip(xs, wx)
            X, dX = path(Xr); x = X - 0.5
            push!(Ae, -x * x / (q * q)); push!(Be, -X); push!(Ce, -dX / sqrt(pi * X) * chi(Xr) * w)
        end
        return Layout(As, Bs, Cs), Layout(Ae, Be, Ce)
    end
    vs = 1 / (2abs(q)); kap = KAPPA / (vs / 2); H = 1.5 / kap
    pathd = d -> begin
        r = vs - d; a = cap(r / H); b = cap(d / H); g = H * a * b
        dg = b / sqrt(1 + (r / H)^2)^3 - a / sqrt(1 + (d / H)^2)^3
        v = r - im * g * sg; dv = 1 - im * dg * sg; X = abs(q) * (d + im * g * sg)
        v, dv, X
    end
    chi = d -> (1 + erf(kap * (vs / 2 - d - im * H * cap((vs - d) / H) * cap(d / H) * sg))) / 2
    rs, wr = geopanels(0.001, vs, NPAN[], 1)                          # smooth: r = vs - d from the bump at v = 0
    As = ComplexF64[]; Bs = ComplexF64[]; Cs = ComplexF64[]
    for (r, w) in zip(rs, wr)
        v, dv, X = pathd(vs - r)
        push!(As, -v * v); push!(Bs, -X); push!(Cs, 2 * v * v / sqrt(pi * X) * dv * (1 - chi(vs - r)) * w)
    end
    ds, wd = geopanels(0.001, vs, NPAN[], 4)                          # edge: d from the support edge
    Ae = ComplexF64[]; Be = ComplexF64[]; Ce = ComplexF64[]
    for (d, w) in zip(ds, wd)
        v, dv, X = pathd(d)
        push!(Ae, -v * v); push!(Be, -X); push!(Ce, 2 * v * v / sqrt(pi * X) * dv * chi(d) * w)
    end
    Layout(As, Bs, Cs), Layout(Ae, Be, Ce)
end
const E1 = cis(pi / 8)       # fixed direction of the ray from X = 1/2 for the constant part (Re(s_2 e1) > 0 for c <= 0)
"""Layouts for the K_0 kernel (M2, M3, the stationary cases) at q and tau: the smooth part, the edge part (the
neighbourhood of X = 0 and the ray X < 0), and the ray X > 1/2 of the constant part; all with c <= 0."""
function layouts_k0(q::Float64, dens::Bool, tau::Float64)
    sg = 1.0
    # the constant part's ray X = 1/2 + r e1 over X > 1/2 (distribution function only): phi_pos ~ e^{-s2 X}
    rs, wr = geopanels(0.002, 36.0 / 2.5, NPANR[], 1)                   # |s_2| >= 2.5 at the real point for c <= 0
    Ac = ComplexF64[]; Bc = ComplexF64[]; Cc = ComplexF64[]; Xc = ComplexF64[]
    for (r, w) in zip(rs, wr)
        X = 0.5 + r * E1
        push!(Ac, 0); push!(Bc, -X); push!(Cc, E1 * w / pi); push!(Xc, X)
    end
    Lc = dens ? emptyL() : Layout(Ac, Bc, Cc, Xc, false)
    if q > 0
        R = abs(q) * sqrt(85.0 / tau)
        xs, ws = geopanels(0.001 * abs(q), R, NPANK[], 1)
        A = ComplexF64[]; B = ComplexF64[]; C = ComplexF64[]; Xs = ComplexF64[]
        for (r, w) in zip(xs, ws)
            if dens
                v = r * E0 / abs(q); X = q * v + 0.5
                push!(A, -v * v); push!(B, -X); push!(C, 2 * v * v * E0 / abs(q) * w / pi); push!(Xs, X)
            else
                X = 0.5 + r * E0
                push!(A, -(r * E0)^2 / (q * q)); push!(B, -X); push!(C, E0 * w / pi); push!(Xs, X)
            end
        end
        return [Layout(A, B, C, Xs, false)], Layout[], [Lc]
    end
    # q < 0: the bulged segment (0, 1/2) with the partition, and the ray X < 0
    if !dens
        kap = KAPPA / 0.25; H = 1.5 / kap
        path = Xr -> (a = cap(Xr / H); b = cap((0.5 - Xr) / H); g = H * a * b;
                      dg = b / sqrt(1 + (Xr / H)^2)^3 - a / sqrt(1 + ((0.5 - Xr) / H)^2)^3;
                      (Xr + im * g * sg, 1 + im * dg * sg))
        chi = Xr -> (1 + erf(kap * (0.25 - Xr - im * H * cap(Xr / H) * cap((0.5 - Xr) / H) * sg))) / 2
        ys, wy = geopanels(0.001 * abs(q), 0.5, NPANK[], 1)
        As = ComplexF64[]; Bs = ComplexF64[]; Cs = ComplexF64[]; Xs = ComplexF64[]
        for (y, w) in zip(ys, wy)
            X, dX = path(0.5 - y); x = X - 0.5
            push!(As, -x * x / (q * q)); push!(Bs, -X); push!(Cs, -dX / pi * (1 - chi(0.5 - y)) * w); push!(Xs, X)
        end
        xs, wx = geopanels(0.001 * abs(q), 0.5, NPANK[], 4)
        Ae = ComplexF64[]; Be = ComplexF64[]; Ce = ComplexF64[]; Xe = ComplexF64[]
        for (Xr, w) in zip(xs, wx)
            X, dX = path(Xr); x = X - 0.5
            push!(Ae, -x * x / (q * q)); push!(Be, -X); push!(Ce, -dX / pi * chi(Xr) * w); push!(Xe, X)
        end
        # the ray X = -r e0, X < 0: -phi_neg(X) e^{-t x^2/q^2} e0, x = X - 1/2 = -(1/2 + r e0)
        rs, wr = geopanels(1e-4 * q * q, abs(q) * sqrt(85.0 / tau), NPANR[], 4)
        An = ComplexF64[]; Bn = ComplexF64[]; Cn = ComplexF64[]; Xn = ComplexF64[]
        for (r, w) in zip(rs, wr)
            X = -r * E0; x = X - 0.5
            push!(An, -x * x / (q * q)); push!(Bn, -X); push!(Cn, -E0 / pi * w); push!(Xn, X)
        end
        return [Layout(As, Bs, Cs, Xs, false)], [Layout(Ae, Be, Ce, Xe, false), Layout(An, Bn, Cn, Xn, true)], [Lc]
    end
    vs = 1 / (2abs(q)); kap = KAPPA / (vs / 2); H = 1.5 / kap
    pathd = d -> (r = vs - d; a = cap(r / H); b = cap(d / H); g = H * a * b;
                  dg = b / sqrt(1 + (r / H)^2)^3 - a / sqrt(1 + (d / H)^2)^3;
                  (r - im * g * sg, 1 - im * dg * sg, abs(q) * (d + im * g * sg)))
    chi = d -> (1 + erf(kap * (vs / 2 - d - im * H * cap((vs - d) / H) * cap(d / H) * sg))) / 2
    rs, wr = geopanels(0.001, vs, NPANK[], 1)
    As = ComplexF64[]; Bs = ComplexF64[]; Cs = ComplexF64[]; Xs = ComplexF64[]
    for (r, w) in zip(rs, wr)
        v, dv, X = pathd(vs - r)
        push!(As, -v * v); push!(Bs, -X); push!(Cs, 2 * v * v / pi * dv * (1 - chi(vs - r)) * w); push!(Xs, X)
    end
    ds, wd = geopanels(0.001, vs, NPANK[], 4)
    Ae = ComplexF64[]; Be = ComplexF64[]; Ce = ComplexF64[]; Xe = ComplexF64[]
    for (d, w) in zip(ds, wd)
        v, dv, X = pathd(d)
        push!(Ae, -v * v); push!(Be, -X); push!(Ce, 2 * v * v / pi * dv * chi(d) * w); push!(Xe, X)
    end
    # the ray beyond v*: v = vs + r e0, X = -|q| r e0: 2 v^2 e^{-t v^2} phi_neg(X) e0
    rs, wr = geopanels(1e-4, sqrt(85.0 / tau), NPANR[], 4)
    An = ComplexF64[]; Bn = ComplexF64[]; Cn = ComplexF64[]; Xn = ComplexF64[]
    for (r, w) in zip(rs, wr)
        v = vs + r * E0; X = -abs(q) * r * E0
        push!(An, -v * v); push!(Bn, -X); push!(Cn, 2 * v * v / pi * E0 * w); push!(Xn, X)
    end
    [Layout(As, Bs, Cs, Xs, false)], [Layout(Ae, Be, Ce, Xe, false), Layout(An, Bn, Cn, Xn, true)], [Lc]
end
const LAYOUTS = Dict{Tuple{Float64,Bool,Int,Float64},Tuple{Layout,Layout}}()
const LAYOUTSK = Dict{Tuple{Float64,Bool,Float64,Int,Int,Float64},Tuple{Vector{Layout},Vector{Layout},Vector{Layout}}}()
const LAYLOCK = ReentrantLock()
layouts(q, dens) = lock(LAYLOCK) do; get!(() -> layouts_m1(q, dens), LAYOUTS, (q, dens, NPAN[], RATIO[])); end
layoutsk(q, dens, tau) = lock(LAYLOCK) do; get!(() -> layouts_k0(q, dens, tau), LAYOUTSK, (q, dens, tau, NPANK[], NPANR[], RATIO[])); end
"""The inner values of all nodes: out[j] = const(t_j) + sum_k C_k exp(A_k t_j + B_k s1_j), on NWORKERS tasks."""
function batchvals(L::Layout, ts::Vector{ComplexF64}, s1::Vector{ComplexF64}, constpart)
    J = length(ts); out = Vector{ComplexF64}(undef, J); nw = NWORKERS[]
    next = Threads.Atomic{Int}(1); ch = max(1, CHUNK[])
    @sync for w in 1:nw
        Threads.@spawn begin
            buf = takebuf(length(L.A))
            while true
                j0 = Threads.atomic_add!(next, ch); j0 > J && break
                for j in j0:min(j0 + ch - 1, J)
                    out[j] = constpart(ts[j], s1[j]) + (isempty(L.A) ? zero(ComplexF64) : expsum(L.A, L.B, L.C, ts[j], s1[j], buf))
                end
            end
            givebuf(buf)
        end
    end
    out
end
# the K_0 kernel: phi_pos(X) = k0x(b X) e^{-s2 X}/pi for Re X > 0, phi_neg(X) = k0x(-b X) e^{-s1 X}/pi for Re X < 0,
# b = (s2 - s1)/2; a layout holds nodes of one sign of Re X except the edge layout, split at the first X < 0 node
const CHUNK = Ref(2)        # nodes per unit of work of the dynamic schedule of the batched loops (0: static round robin)
function batchvalsk(Ls::Vector{Layout}, ts::Vector{ComplexF64}, s1::Vector{ComplexF64}, s2::Vector{ComplexF64})
    J = length(ts); out = Vector{ComplexF64}(undef, J); nw = NWORKERS[]
    nmax = maximum(length(L.A) for L in Ls; init = 1)
    next = Threads.Atomic{Int}(1); ch = CHUNK[]
    @sync for w in 1:nw
        Threads.@spawn begin
            buf = takebuf(nmax)
            if ch <= 0
                for j in w:nw:J; out[j] = kvalue(Ls, ts[j], s1[j], s2[j], buf); end
            else
                while true                                        # chunks handed out in order: the cores that are faster take more
                    j0 = Threads.atomic_add!(next, ch); j0 > J && break
                    for j in j0:min(j0 + ch - 1, J); out[j] = kvalue(Ls, ts[j], s1[j], s2[j], buf); end
                end
            end
            givebuf(buf)
        end
    end
    out
end
@inline function kvalue(Ls::Vector{Layout}, t::ComplexF64, s1::ComplexF64, s2::ComplexF64, buf::ExpBuf)
    b = (s2 - s1) / 2
    v = zero(ComplexF64)
    for L in Ls
        isempty(L.A) && continue
        v += L.neg ? expsumk(L.A, L.B, L.C, t, s1, buf, L.X, -b) : expsumk(L.A, L.B, L.C, t, s2, buf, L.X, b)
    end
    v
end

# ---------- the outer integrals ----------
# f(q) = (1/pi) int_0^inf Re[P(t) I_f(t)] dw and P(tau <= q) = (1/pi) int_0^inf Re[P(t) I_F(t)/t] dw,
# t = tau + i w.  The smooth part of the integrand decays like exp(-sqrt(w)/2) and is analytic in a
# strip: w = tau sinh(u) and the trapezoidal rule in u (a few dozen nodes).  The edge part
# oscillates like e^{-i w v*^2}: the trapezoidal rule in w with a step set by v*^2 (and by the
# distance tau of the pole at t = 0), up to where exp(-sqrt(w)/2) is negligible.
const WMAX = 4900.0         # exp(-sqrt(WMAX)/2) = 6e-16
const FILONW = Ref(300.0)   # the Filon panels are at most 25/y long up to w = FILONW/y (the transient of the partition)
const TIMING = Ref(false)
using Printf

"""
    filon_panel(G, a, b, y, N)

int_a^b G(w) e^{-i y w} dw for smooth G: G is interpolated by Legendre polynomials of degree < N at the
Gauss–Legendre nodes of [a, b] and the exponential is integrated exactly, by the moments
int_{-1}^1 P_n(x) e^{-i Omega x} dx = 2 (-i)^n j_n(Omega), Omega = y (b - a)/2.
"""
const FILONCACHE = Dict{Int,Tuple{Matrix{Float64},Vector{Float64}}}()      # per N: the Legendre values at the nodes, the weights
function filon_setup(N::Int)
    lock(RULELOCK) do; get!(FILONCACHE, N) do
        u, gw = rule(N); x = 2 .* u .- 1; ww = 2 .* gw
        P = Matrix{Float64}(undef, N, N)                              # P[n+1, k] = P_n(x_k)
        P[1, :] .= 1; P[2, :] .= x
        for n in 1:N-2; P[n+2, :] .= ((2n + 1) .* x .* P[n+1, :] .- n .* P[n, :]) ./ (n + 1); end
        (P, ww)
    end; end
end
"""j_n(x) for n = 0..N-1, x >= 0: upward recurrence for x > N (stable), else downward (Miller) from n = N + 40
normalized by j_0 = sin(x)/x."""
function sphbessel!(j::Vector{Float64}, N::Int, x::Float64)
    if x < 1e-12
        fill!(j, 0.0); j[1] = 1.0; return j
    end
    if x > N
        j[1] = sin(x) / x; N > 1 && (j[2] = (sin(x) - x * cos(x)) / (x * x))
        for n in 1:N-2; j[n+2] = (2n + 1) / x * j[n+1] - j[n]; end
    else
        M = N + 40; jp = 0.0; jc = 1e-300
        tmp = Vector{Float64}(undef, M + 1); tmp[M+1] = jc
        for n in M-1:-1:0
            jn = (2n + 3) / x * jc - jp; jp = jc; jc = jn; tmp[n+1] = jc
            if abs(jc) > 1e200; tmp ./= jc; jp /= jc; jc = 1.0; end
        end
        f = (sin(x) / x) / tmp[1]
        for n in 0:N-1; j[n+1] = tmp[n+1] * f; end
    end
    j
end
function filon_panel(Gvals::AbstractVector{ComplexF64}, a::Float64, b::Float64, y::Float64, N::Int)
    P, ww = filon_setup(N)
    L = b - a; Omega = y * L / 2
    jn = sphbessel!(Vector{Float64}(undef, N), N, Omega)
    acc = zero(ComplexF64); mi = one(ComplexF64)                         # (-i)^n
    @inbounds for n in 0:N-1
        cn = zero(ComplexF64)
        for k in 1:N; cn += ww[k] * Gvals[k] * P[n+1, k]; end
        acc += (2n + 1) * cn * mi * jn[n+1]
        mi *= -im
    end
    L / 2 * exp(-im * y * (a + b) / 2) * acc
end

"""
    saddle_phase(model, q, ts, s1, s2)

The phase of the saddle factor exp(-u0) of the Gaussian part of the inner functions (q > 0) at the
points ts of the line, with u0 = -sd^2 q^2/(4t) = -z^2, z = sd q/(2 sqrt(t)), the depth of the
saddle of the exponent -t x^2/q^2 - sd x.  The model integral int_0^inf exp(-t x^2/q^2 - sd x) dx =
(q/2) sqrt(pi/t) exp(z^2) erfc(z) shows the factor present for Re z < 0, that is when the
exponential grows along the ray x = r e^{-i arg(t)/2} (never for c <= 0, where the saddle is uphill
at the real point and Re z >= 0 along the line), with the transition of erfc around Re z = 0.  Zero
where the factor is absent or negligible.
"""
function saddle_phase(kern::Int, q::Float64, ts::Vector{ComplexF64}, s1::Vector{ComplexF64}, s2::Vector{ComplexF64})
    sd = kern == 1 ? s1 : s2
    psi = fill(NaN, length(ts))                                   # NaN: the factor is negligible there
    for j in eachindex(ts)
        z = sd[j] * q / (2sqrt(ts[j]))
        # the saddle term 2 exp(z^2) of exp(z^2) erfc(z) is present for Re z < 0 (beyond the Stokes line
        # Re z = 0 of the asymptotics of erfc, where it is still of half size but the phase is small) and is
        # of size exp(Re z^2); for c <= 0, Re z >= 0 along the whole line
        (real(z) < 0 && real(z * z) > -30) && (psi[j] = imag(z * z))
    end
    psi
end

"""The changes of the phase psi between consecutive points, zero where the factor is negligible at either."""
phase_steps(psi) = [isnan(psi[j]) || isnan(psi[j+1]) ? 0.0 : abs(psi[j+1] - psi[j]) for j in 1:length(psi)-1]

"""
    outer(model, c, q, tau, inner, weight; hu, N)

The outer integral.  Smooth part: w = tau sinh(u), trapezoidal rule in u with step hu, or, when the
phase of the saddle factor (q > 0, see saddle_phase) changes too fast for it, Gauss–Legendre panels.  Edge part
(q < 0): the factor e^{-t v*^2} is taken out, the rest is interpolated on geometric panels
[0, 1/2], [1/2, 1], [1, 2], ..., [WEDGE/2, WEDGE] with N Gauss–Legendre nodes each, and the
oscillation is integrated exactly (Filon).
"""
function outer(model::Int, c::Float64, q::Float64, tau::Float64, inner::Function, weight::Function;
               hu::Float64 = 0.075, N::Int = 20, cde::Float64 = tau, parts::Bool = false, edge_hw = nothing, pfac::Int = 1, stat::Bool = false,
               batch = nothing)
    kern = kernel(model, stat)
    # with a batch (the fixed layouts of M1), the inner values of a node set come from batch(ts, s1, part)

    # the smooth part
    t0 = time_ns()
    umax = asinh(WMAX / cde)
    us = collect(0.0:hu:umax)
    ts = [tau + im * cde * sinh(u) for u in us]
    P, s1, s2, M0 = tparts(model, c, ts; stat)
    t1 = time_ns()
    # For q > 0 the Gaussian part of the inner function carries the factor exp(-u0), u0 = -sd^2 q^2/(4t)
    # the depth of the saddle of its exponent, whenever the saddle is downhill (Re u0 >= 0); its phase
    # grows like q^2 along the line.  The trapezoidal rule serves while the phase changes by at most
    # 0.25 rad between its nodes; beyond that (large q, or c > 0 with tau small), Gauss-Legendre panels
    # with the node counts set by the phase change across each panel.
    psi = saddle_phase(kern, q, ts, s1, s2)
    if q > 0 && maximum(phase_steps(psi)) > 0.25
        w0 = min(0.5, tau / 2)
        edges0 = [0.0; w0 * 2.0 .^ (0:ceil(Int, log2(WMAX / w0)))]
        edges0[end] = WMAX
        K = 8                                                      # the phase change across a panel, summed over K pieces
        wk = Float64[]
        for p in 1:length(edges0)-1
            append!(wk, range(edges0[p], edges0[p+1]; length = K + 1)[1:end-1])
        end
        push!(wk, WMAX)
        te0 = [tau + im * w for w in wk]
        _, s1e, s2e, _ = tparts(model, c, te0; stat)
        steps = phase_steps(saddle_phase(kern, q, te0, s1e, s2e))
        wn = Float64[]; wt = Float64[]
        for p in 1:length(edges0)-1
            a, b = edges0[p], edges0[p+1]
            dp = sum(steps[(p-1)*K+k] for k in 1:K)
            m = max(1, ceil(Int, dp / 50))                        # sub-panels of at most 50 rad, 1/3 rad per node
            n = max(24, 16 * cld(ceil(Int, 3dp / m), 16)) * NODEFAC[]
            u, gw = rule(n)
            for k in 1:m
                aa = a + (b - a) * (k - 1) / m; bb = a + (b - a) * k / m
                append!(wn, aa .+ (bb - aa) .* u); append!(wt, (bb - aa) .* gw)
            end
        end
        ts = [tau + im * w for w in [0.0; wn]]                 # the real point first, for the branches
        P, s1, s2, M0 = tparts(model, c, ts; stat)
        gs = Vector{Float64}(undef, length(wn))
        if batch === nothing
            tmap!(j -> real(P[j+1] * inner(ts[j+1], s1[j+1], s2[j+1], M0[j+1], 1)[1] * weight(ts[j+1])) * wt[j], gs, length(wn))
        else
            sv = batch(ts[2:end], s1[2:end], s2[2:end], M0[2:end], 1)
            for j in eachindex(wn); gs[j] = real(P[j+1] * sv[j] * weight(ts[j+1])) * wt[j]; end
        end
        val = sum(gs) / pi
    else
        gs = Vector{Float64}(undef, length(ts))
        if batch === nothing
            tmap!(j -> real(P[j] * inner(ts[j], s1[j], s2[j], M0[j], 1)[1] * weight(ts[j])) * cde * cosh(us[j]), gs, length(ts))
        else
            sv = batch(ts, s1, s2, M0, 1)
            for j in eachindex(ts); gs[j] = real(P[j] * sv[j] * weight(ts[j])) * cde * cosh(us[j]); end
        end
        gs[1] *= 0.5
        val = sum(gs) * hu / pi
    end
    t2 = time_ns()
    TIMING[] && @printf("  tparts(smooth) %.2f ms, smooth loop %.2f ms (%d nodes)\n", (t1 - t0) / 1e6, (t2 - t1) / 1e6, length(ts))
    # the edge part, of size e^{-3 tau y/16} at most (the transition region of the partition)
    y = 1 / (4q * q)
    (q > 0 || 3tau * y / 16 > 40) && return (parts ? (val, 0.0) : val)
    if edge_hw !== nothing                                      # the trapezoidal rule in w, for checking
        ws = collect(0.0:edge_hw:WMAX); te = [tau + im * w for w in ws]
        P, s1, s2, M0 = tparts(model, c, te; stat)
        ge = Vector{Float64}(undef, length(te))
        tmap!(j -> real(P[j] * inner(te[j], s1[j], s2[j], M0[j], 2)[2] * weight(te[j])), ge, length(te))
        ge[1] *= 0.5
        ev = sum(ge) * edge_hw / pi
        return parts ? (val, ev) : val + ev
    end
    # panels: geometric, starting on the scale min(1/2, 1/y, tau/2) of the pole and of the transient of the
    # transition region of the partition, which oscillates at the frequency 3y/4 and decays at the rate
    # y/8: up to w = 300/y the panels are at most 25/y long
    w0 = min(0.5, 1 / y, tau / 2)                                # (tau/2: the pole at t = 0 must be resolved by the first panels)
    edges0 = [0.0; w0 * 2.0 .^ (0:ceil(Int, log2(WMAX / w0)))]
    edges0[end] = WMAX
    edges = Float64[0.0]
    for p in 1:length(edges0)-1
        a, b = edges0[p], edges0[p+1]
        m = b <= FILONW[] / y ? ceil(Int, (b - a) * y / 25) : 1
        append!(edges, range(a, b; length = m + 1)[2:end])
    end
    if pfac > 1                                                  # refined panels, for checking
        e2 = Float64[]
        for p in 1:length(edges)-1
            append!(e2, range(edges[p], edges[p+1]; length = pfac + 1)[1:end-1])
        end
        edges = [e2; edges[end]]
    end
    u, _ = rule(N)
    wn = Float64[]
    for p in 1:length(edges)-1
        append!(wn, edges[p] .+ (edges[p+1] - edges[p]) .* u)
    end
    te = [tau + im * w for w in [0.0; wn]]                     # the real point first, for the branches
    t3 = time_ns()
    P, s1, s2, M0 = tparts(model, c, te; stat)
    t4 = time_ns()
    G = Vector{ComplexF64}(undef, length(wn))
    if batch === nothing
        tmap!(j -> (t = te[j+1]; P[j+1] * inner(t, s1[j+1], s2[j+1], M0[j+1], 2)[2] * weight(t) * exp(t * y)), G, length(wn))
    else
        ev = batch(te[2:end], s1[2:end], s2[2:end], M0[2:end], 2)
        for j in eachindex(wn); t = te[j+1]; G[j] = P[j+1] * ev[j] * weight(t) * exp(t * y); end
    end
    t5 = time_ns()
    acc = zero(ComplexF64)
    for p in 1:length(edges)-1
        acc += filon_panel(view(G, (p-1)*N+1:p*N), edges[p], edges[p+1], y, N)
    end
    t6 = time_ns()
    TIMING[] && @printf("  tparts(edge) %.2f ms, edge loop %.2f ms (%d nodes), filon %.2f ms\n", (t4 - t3) / 1e6, (t5 - t4) / 1e6, length(wn), (t6 - t5) / 1e6)
    ev = real(exp(-tau * y) * acc) / pi
    parts ? (val, ev) : val + ev
end

function startflag(start::Symbol, c::Float64)
    start == :zero && return false
    start == :stationary || error("start must be :zero or :stationary")
    c < 0 || error("the stationary start needs c < 0")
    true
end

const BATCHED = Ref(true)   # use the fixed layouts for M1 (zero start, c <= 0): 4-5x faster, agrees to 1e-13
# the fixed layouts serve c <= 0 and |q| above about 0.034: for smaller |q| the bulge of the segment path (of height
# 1.5/kappa ~ 1/(8|q|) in v) lets the Gaussian factor grow like e^{2 w r g} and the segment integral cancels
# catastrophically; the adaptive code then takes the ray from X = 1/2 (its criterion 3 tau y/16 > 40)
usebatch(c, q, tau) = BATCHED[] && c <= 0 && !(3tau / (64q * q) > 40)
const BATCHK0 = Ref(true)   # the fixed layouts for the K_0 kernel (ratio 4, 48/32 nodes: 3e-13 against the adaptive rules)
function m1batch(q::Float64, dens::Bool, tau::Float64)
    Ls, Le = layouts(q, dens)
    y = 1 / (4q * q)
    edgeneg = q > 0 || 3tau * y / 16 > 40                    # the edge part is omitted (as in the scalar code)
    (ts, s1, s2, M0, part) -> begin
        if part == 1
            cst = dens ? ((t, s) -> zero(ComplexF64)) : ((t, s) -> erf(sqrt(s / 2)) / sqrt(s))
            batchvals(Ls, ts, s1, cst)
        else
            edgeneg ? zeros(ComplexF64, length(ts)) : batchvals(Le, ts, s1, (t, s) -> zero(ComplexF64))
        end
    end
end
function k0batch(q::Float64, dens::Bool, tau::Float64)
    Ls, Le, Lc = layoutsk(q, dens, tau)
    y = 1 / (4q * q)
    edgeneg = q > 0 || 3tau * y / 16 > 40
    (ts, s1, s2, M0, part) -> begin
        if part == 1
            v = batchvalsk(Ls, ts, s1, s2)
            dens || (v .+= M0 .- batchvalsk(Lc, ts, s1, s2))        # the constant part: M0 minus the ray over X > 1/2
            v
        else
            edgeneg ? zeros(ComplexF64, length(ts)) : batchvalsk(Le, ts, s1, s2)
        end
    end
end
function density(qin::Float64, model::Int; c::Float64 = 0.0, tau = nothing, start::Symbol = :zero, c0::Float64 = 0.0, g0::Float64 = 0.0, gen::Bool = false, kw...)
    q = qin == 0 ? 1e-12 : qin
    stat = startflag(start, c)
    tau === nothing && (tau = default_tau(model, c; stat))
    (c0 != 0 || g0 != 0 || gen) && return value_gen(1, q, model, c, c0, tau, stat; g0, kw...)
    V = Val(kernel(model, stat))
    batch = !usebatch(c, q, tau) ? nothing : (kernel(model, stat) == 1 ? (tau == 1.0 ? m1batch(q, true, tau) : nothing) : (BATCHK0[] ? k0batch(q, true, tau) : nothing))
    outer(model, c, q, tau, (t, s1, s2, M0, part) -> inner_density(V, t, q, s1, s2, M0; part = part), t -> one(t); stat, batch, kw...)
end

function cdf(qin::Float64, model::Int; c::Float64 = 0.0, tau = nothing, start::Symbol = :zero, c0::Float64 = 0.0, g0::Float64 = 0.0, gen::Bool = false, kw...)
    q = qin == 0 ? 1e-12 : qin
    stat = startflag(start, c)
    tau === nothing && (tau = default_tau(model, c; stat))
    (c0 != 0 || g0 != 0 || gen) && return value_gen(2, q, model, c, c0, tau, stat; g0, kw...)
    V = Val(kernel(model, stat))
    batch = !usebatch(c, q, tau) ? nothing : (kernel(model, stat) == 1 ? (tau == 1.0 ? m1batch(q, false, tau) : nothing) : (BATCHK0[] ? k0batch(q, false, tau) : nothing))
    outer(model, c, q, tau, (t, s1, s2, M0, part) -> inner_cdf(V, t, q, s1, s2, M0; part = part), t -> 1 / t; stat, batch, kw...)
end

# the old scheme (trapezoidal rule in w for the whole integrand), kept for checking
function density_uniform(qin::Float64, model::Int; c::Float64 = 0.0, tau = nothing, W::Float64 = WMAX, hw::Float64 = 0.1, start::Symbol = :zero, c0::Float64 = 0.0)
    q = qin == 0 ? 1e-12 : qin
    stat = startflag(start, c); kern = kernel(model, stat)
    tau === nothing && (tau = default_tau(model, c; stat))
    c0 != 0 && return value_gen(1, q, model, c, c0, tau, stat; hw)
    w = collect(0.0:hw:W); t = tau .+ im .* w
    P, s1, s2, M0 = tparts(model, c, t; stat)
    g = Vector{Float64}(undef, length(t))
    tmap!(j -> real(P[j] * sum(inner_density(kern, t[j], q, s1[j], s2[j], M0[j]))), g, length(t))
    g[1] *= 0.5
    sum(g) * hw / pi
end
function cdf_uniform(qin::Float64, model::Int; c::Float64 = 0.0, tau = nothing, W::Float64 = WMAX, hw::Float64 = 0.1, start::Symbol = :zero, c0::Float64 = 0.0)
    q = qin == 0 ? 1e-12 : qin
    stat = startflag(start, c); kern = kernel(model, stat)
    tau === nothing && (tau = default_tau(model, c; stat))
    c0 != 0 && return value_gen(2, q, model, c, c0, tau, stat; hw)
    w = collect(0.0:hw:W); t = tau .+ im .* w
    P, s1, s2, M0 = tparts(model, c, t; stat)
    g = Vector{Float64}(undef, length(t))
    tmap!(j -> real(P[j] * sum(inner_cdf(kern, t[j], q, s1[j], s2[j], M0[j])) / t[j]), g, length(t))
    g[1] *= 0.5
    sum(g) * hw / pi
end

"""Quantile by Newton's method on the distribution function with the density as derivative."""
function quantile(prob::Float64, model::Int; c::Float64 = 0.0, q0 = nothing, tol = 1e-9, start::Symbol = :zero, c0::Float64 = 0.0)
    q = q0 === nothing ? (model == 1 ? -0.5 : model == 2 ? -1.6 : model == 4 ? -2.4 : -2.2) : q0
    lo, hi = q - 1.0, q + 1.0                                          # a bracket, then Newton safeguarded by bisection
    while cdf(lo, model; c, start, c0) > prob; lo -= 2(q - lo); end
    while cdf(hi, model; c, start, c0) < prob; hi += 2(hi - q); end
    for it in 1:60
        F = cdf(q, model; c, start, c0); f = density(q, model; c, start, c0)
        F > prob ? (hi = q) : (lo = q)
        qn = q - (F - prob) / f
        (lo < qn < hi) || (qn = (lo + hi) / 2)
        abs(qn - q) < tol && return qn
        q = qn
    end
    error("quantile search did not converge")
end

# ---------- moments ----------
"""
    moments_dfx(model; c = 0.0, start = :zero, pmax = 4)   (the first implementation, kept as an independent check of `moments`)

The first pmax raw moments of the limit, E tau^p = Gamma(p/2)^{-1} int_0^inf s^{p/2-1} m_p(s) ds with
m_p(s) = d^p/du^p M(u, -s) at u = 0 and M(u, t) = E exp(uU + tV) = e^{-u/2} D(u, t)^{-1/2}.  The derivatives
are Cauchy integrals on a circle of 0.4 times the distance from 0 to the nearest root of D(., -s), with
64 nodes and the branch of D^{-1/2} continued around the circle from its real point; the s-integral is
the trapezoidal rule in log s on [e^{-60}, e^8] with the step 0.05.
"""
function moments_dfx(model::Int; c::Float64 = 0.0, start::Symbol = :zero, pmax::Int = 4, h::Float64 = 0.05, N::Int = 64, rfrac::Float64 = 0.4)
    stat = startflag(start, c)
    extra = (model == 1 && !stat) ? -c / 2 : 0.0                     # log M = -u/2 + extra - log(D)/2
    ls = collect(-60.0:h:8.0)
    acc = zeros(pmax)
    th = 2pi .* (0:N-1) ./ N
    for l in ls
        s = exp(l); t = complex(-s)
        _, s1, s2, _ = tparts(model, c, [t]; stat)
        rad = rfrac * (kernel(model, stat) == 1 ? abs(s1[1]) : min(abs(s1[1]), abs(s2[1])))
        u = rad .* cis.(th)
        d = [Dfun(model, c, uk, t; stat) for uk in u]
        ph = unwrap!(angle.(d))
        M = [exp(-u[k] / 2 + extra) * abs(d[k])^(-0.5) * exp(-im * ph[k] / 2) for k in 1:N]
        for p in 1:pmax
            mp = real(sum(M[k] * cis(-p * th[k]) for k in 1:N)) / N / rad^p * factorial(p)
            acc[p] += s^(p / 2) * mp                                  # s^{p/2-1} m_p(s) ds, ds = s d(log s)
        end
    end
    [acc[p] * h / gamma(p / 2) for p in 1:pmax]
end

# =====================================================================================
# Moments: any c, c0 and start (Proposition 3; a port of Raymond Kan's TR_LIMMOM)
#
# E tau^p = Gamma(p/2)^{-1} int_0^inf s^{p/2-1} m_p(s) ds, m_p(s) = d^p/du^p M(u, -s - c0 u) at u = 0, where the moment
# generating function M(u, t) = E exp(u U + t V) is used near the real axis only, where it is smooth and positive.  The
# derivatives in u are Cauchy integrals on a circle of radius r(s) inside the domain of analyticity (trapezoidal rule,
# Nu nodes), with log D continued along the circle from the real point u = r (a circle on which it does not close
# encloses a zero of D, and its radius is halved); the integral over s is the trapezoidal rule in log s, with an error
# estimate from the rule with half the step.  With a zero start the moment generating function comes from the Girsanov
# forms, log E exp(u R - u/2 + v V) = -u/2 - c/2 - [log(sinh th/th) + log(eta - u)]/2 in case DK and
# -u/2 - c/2 - [log cosh th + log det(I - (Q + u K) Sig)]/2 in cases DU, DT, th = (c^2 - 2v)^{1/2}, free of the
# cancellation of terms of order e^c for c >> 0; a fixed start adds g0^2 Q(u, v) (Proposition 2), with D in Q written in
# the same forms; the stationary start uses D_S.  For c > 0 the radius r(s) is also at least 0.4 times the distance to
# the nearest real zero of the argument of the logarithm, which for c0 = c >> 0 lies far beyond 1/(2 kap_1) (the terms
# of order u of R and c0 V cancel), where a circle of radius 0.4/(2 kap_1) leaves the derivatives below rounding.

"""the logarithm of Z continued along a closed circle starting at a point where Z > 0; ok: the steps are resolved and
the circle closes."""
function circlog(Z::Vector{ComplexF64})
    n = length(Z)
    lz = log.(Z)
    ok = abs(imag(lz[1])) < 1e-8
    ph = Vector{Float64}(undef, n); ph[1] = imag(lz[1])
    for k in 2:n
        d = angle(exp(im * (imag(lz[k]) - imag(lz[k-1]))))
        abs(d) < pi / 3 || (ok = false)
        ph[k] = ph[k-1] + d
    end
    ok &= abs(ph[end] + angle(Z[1] / Z[end]) - ph[1]) < 1e-6
    [real(lz[k]) + im * ph[k] for k in 1:n], ok
end

"""log E exp(u R - u/2 + v V) with a zero start by the Girsanov forms, at the points of a circle (U) with the arguments V of
V; ok: the logarithm closes along the circle."""
function zmgf(model::Int, c::Float64, U::Vector{ComplexF64}, V::Vector{ComplexF64})
    n = length(U)
    L = Vector{ComplexF64}(undef, n); arg = Vector{ComplexF64}(undef, n); base = Vector{ComplexF64}(undef, n)
    for k in 1:n
        G = girsanov(model, c, V[k])
        if model == 1
            arg[k] = G.eta - U[k]; base[k] = -U[k] / 2 - c / 2 - G.lsh / 2
        else
            K = model == 2 ? KDU : KDT
            M = G.N - U[k] .* (K * G.Sg)
            arg[k] = model == 2 ? M[1,1] * M[2,2] - M[1,2] * M[2,1] : det3(M)
            base[k] = -U[k] / 2 - c / 2 - G.lch / 2
        end
    end
    lz, ok = circlog(arg)
    [base[k] - lz[k] / 2 for k in 1:n], ok
end

"""the argument of the logarithm in zmgf at real u, v (real there), with the sign of cosh th."""
function zarg(model::Int, c::Float64, u::Float64, v::Float64)
    G = girsanov(model, c, complex(v))
    model == 1 && return real(G.eta - u)
    K = model == 2 ? KDU : KDT
    M = G.N - u .* (K * G.Sg)
    d = model == 2 ? M[1,1] * M[2,2] - M[1,2] * M[2,1] : det3(M)
    real(d) * sign(real(exp(G.lch)))
end

"""smallest |u| > 0 on the real axis at which the argument of the logarithm in zmgf changes sign (or stops being
finite), from u = +-r0 2^k, k = 0..80, then bisection; r0/0.4 when none is found."""
function zrad(model::Int, c::Float64, c0::Float64, s::Float64, r0::Float64)
    R = Inf
    for sg in (1.0, -1.0)
        lo = 0.0; hi = Inf; u = sg * r0
        for _ in 0:80
            d = zarg(model, c, u, -s - c0 * u)
            if !(d > 0) || !isfinite(d)
                hi = abs(u); break
            end
            lo = abs(u); u *= 2
        end
        if isfinite(hi)
            for _ in 1:30
                m = (lo + hi) / 2; d = zarg(model, c, sg * m, -s - c0 * sg * m)
                (d > 0 && isfinite(d)) ? (lo = m) : (hi = m)
            end
        end
        R = min(R, lo)
    end
    R > 0 ? R : r0 / 0.4
end

"""The initial-value term Q(u, v) of Proposition 2 (log M_{g0} = log M_0 + g0^2 Q), in the Girsanov forms: cases DU, DT
Q = c^{2k} (v + u^2/2 + c u) g(z)/{cosh th det(I - (Q + u K) Sig)}, z = c^2 - 2v, k = 1, 2; case DK
Q = v/eta - u/2 + u/{2 S^2 eta (eta - u)}, S = sinh th/th."""
function qstart(model::Int, c::Float64, u::ComplexF64, v::ComplexF64)
    G = girsanov(model, c, v)
    if model == 1
        S = exp(G.lsh)
        return v / G.eta - u / 2 + u / (2 * S * S * G.eta * (G.eta - u))
    end
    K = model == 2 ? KDU : KDT
    M = G.N - u .* (K * G.Sg)
    d = model == 2 ? M[1,1] * M[2,2] - M[1,2] * M[2,1] : det3(M)
    # g(z)/cosh th = g(z) e^{-th} 2/(1 + e^{-2 th}) with z = c^2 - 2v = th^2: no overflow at large tilts
    (model == 2 ? c^2 : c^4) * (v + u * u / 2 + c * u) * gstat_scaled(model, c * c - 2v) * 2 / ((1 + exp(-2 * G.th)) * d)
end

"""
    moments(model; c = 0.0, c0 = 0.0, g0 = 0.0, start = :zero, pmax = 4) -> m

E tau^p, p = 1..pmax, for the limit tau = (U - c0 V)/sqrt(V) with a zero, a fixed (g0) or a stationary start.  Options:
Nu (64) nodes on the circle, h (0.05) the step in log s, tlo (min(-60, -2c - 40)) and thi (8) the range of log s,
rfac (0.4).  `moments_err` returns the moments with the error estimates of the s-integrals.
"""
function moments_err(model::Int; c::Float64 = 0.0, c0::Float64 = 0.0, g0::Float64 = 0.0, start::Symbol = :zero, pmax::Int = 4,
                     Nu::Int = 64, h::Float64 = 0.05, tlo::Float64 = min(-60.0, -2c - 40), thi::Float64 = 8.0, rfac::Float64 = 0.4)
    stat = startflag(start, c)
    stat && g0 != 0 && error("a fixed start g0 applies to the zero-start cases only")
    kern = kernel(model, stat)
    smax = tstar(model, c; stat)                                    # the right end of the real domain of E exp(sV)
    svals = exp.(collect(tlo:h:thi)); ns = length(svals)
    zs = !stat                                                      # zero or fixed start: the Girsanov forms
    r = Vector{Float64}(undef, ns)
    for (i, s) in enumerate(svals)
        _, s1, s2, _ = tparts(model, c, [complex(-s)]; stat)        # the weights of R under the real tilt -s: r = 0.4 min |s_k|
        r[i] = rfac * (kern == 1 ? abs(s1[1]) : min(abs(s1[1]), abs(s2[1])))
        c0 != 0 && (r[i] = min(r[i], 0.4 * (s + smax) / abs(c0)))   # the argument -s - c0 u of V must stay in the domain
        zs && c > 0 && g0 == 0 && (r[i] = max(r[i], rfac * zrad(model, c, c0, s, r[i])))
        (stat || g0 != 0) && (r[i] = min(r[i], 0.25))               # the start term must stay analytic
    end
    w = [cis(2pi * k / Nu) for k in 0:Nu-1]
    extra = (model == 1 && !stat) ? -c / 2 : 0.0
    acc = zeros(pmax, ns)
    for i in 1:ns
        s = svals[i]; ok = false; L = ComplexF64[]
        for _ in 1:8
            U = r[i] .* w; V = [-s - c0 * u for u in U]
            if zs
                L, ok = zmgf(model, c, U, V)
                g0 != 0 && (L .+= g0^2 .* [qstart(model, c, U[k], V[k]) for k in 1:Nu])
            else
                d = [Dfun(model, c, U[k], V[k]; stat) for k in 1:Nu]
                lz, ok = circlog(d)
                L = [-U[k] / 2 + extra - lz[k] / 2 for k in 1:Nu]
            end
            ok && break
            r[i] /= 2
        end
        ok || error("moments: no circle in u free of zeros of D was found at s = $s")
        f = exp.(L)
        for j in 1:pmax
            Mj = factorial(j) / (Nu * r[i]^j) * real(sum(f[k] * w[k]^(-j) for k in 1:Nu))
            acc[j, i] = s^(j / 2) * Mj                               # s^{p/2-1} m_p(s) ds, ds = s d(log s)
        end
    end
    m = [h * sum(acc[j, :]) / gamma(j / 2) for j in 1:pmax]
    err = [abs(h * sum(acc[j, :]) - 2h * sum(acc[j, 1:2:ns])) / gamma(j / 2) for j in 1:pmax]
    m, err
end
moments(model::Int; kw...) = moments_err(model; kw...)[1]

"""Mean and standard deviation of the limit."""
function meansd(model::Int; c::Float64 = 0.0, start::Symbol = :zero, c0::Float64 = 0.0, g0::Float64 = 0.0)
    m = moments(model; c, start, c0, g0, pmax = 2)
    m[1], sqrt(m[2] - m[1]^2)
end

# =====================================================================================
# The user-level interface: the same arguments as the MATLAB programs ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom
#
# Model: y_t = alpha + delta t + phi y_{t-1} + sigma e_t, phi = 1 + c/n, no drift, and t the least squares t-ratio of the
# hypothesis phi = 1 + c0/n (c0 = 0: the Dickey-Fuller tests of phi = 1) from the regression of y_t on y_{t-1} and the
# deterministic terms.  As n -> inf, t => (U - c0 V)/sqrt(V), U = int J~ dJ, V = int J~^2, J the Ornstein-Uhlenbeck process
# dJ = cJ dr + dW started at g0 = y_0/(sigma sqrt(n)), J~ its residual on the deterministic terms.  Six cases, selected by
# a two-letter code (D: deterministic, i.e. fixed, start; S: stationary start):
#   "DK"  regression of y_t on y_{t-1}          (tau),      start g0
#   "DU"  regression on 1 and y_{t-1}            (tau_mu),   start g0
#   "DT"  regression on 1, t and y_{t-1}         (tau_tau),  start g0
#   "SK", "SU", "ST"  the same regressions with y_0 from the stationary distribution, J(0) ~ N(0, -1/(2c)), c < 0
const CASES = ("DK", "DU", "DT", "SK", "SU", "ST", "DG")                 # DG: GLS-detrended, trend, zero start (Section 7 of the paper)
"""the method of `method = :auto` at the point x: the single integral at c0 = 0 (its fixed node layouts make it the faster
one by 2-20x, zero or stationary start, any c) and with a fixed start g0 != 0 (the noncentral kernel, 2-6x faster than
the double integral and without its limit c <= 2); with c0 != 0 and a zero start the double integral, except for
c0 > 0 at x >= 0, where X(v) has no positive root, the single integral has no breakpoint and takes 0.2-5 ms in C and
about 1 ms here against 1-60 ms for the double integral.  Elsewhere with c0 != 0 the double integral is the faster
one at every c, measured point by point over c in [-5, 5], c0 in {-5, -1, 1, 3} and x in [-4, 32] in the three cases
(by 2-10x in C; with c0 < 0 the single integral's saddle part costs up to 1 s at x ~ 8 with c0 = -1, and here its
general kernel is 5-90x slower).  The tails below 1e-3 are recomputed by the double integral in any case
(ltratio_dist)."""
automethod(model::Int, c::Float64, c0::Float64, g0::Float64, x::Float64) =
    (model == 1 && c0 > 0 && g0 == 0 && x == -sqrt(2c0)) ? :single :       # the infinite density of case DK: the single integral returns Inf there without a flag
    (c0 != 0 && g0 == 0 && !(c0 > 0 && x >= 0)) ? :double : :single
"""
    ltratio_args(cas, c; c0, g0, method) -> (model, stat, c0, g0, method)

Validation of the arguments: the case ("DK", "DU", "DT", "SK", "SU", "ST"), the local-to-unity parameter c (finite; a
stationary start requires c < 0), the local null value c0, the fixed start g0 (fixed-start cases only) and the method
(:auto, the default: the faster representation at each x; :single; :double).
"""
function ltratio_args(cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto)
    cas = uppercase(String(cas))
    cas in CASES || throw(ArgumentError("the case must be one of DK, DU, DT, SK, SU, ST, DG"))
    isfinite(c) || throw(ArgumentError("c must be a finite real scalar"))
    stat = cas[1] == 'S'
    stat && !(c < 0) && throw(ArgumentError("a stationary start (cases SK, SU, ST) requires c < 0"))
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    isfinite(g0) || throw(ArgumentError("g0 must be a finite real scalar"))
    g0 != 0 && stat && throw(ArgumentError("g0 (a fixed start) applies to cases DK, DU, DT only"))
    method in (:auto, :single, :double) || throw(ArgumentError("method must be :auto, :single or :double"))
    model = cas == "DG" ? 4 : findfirst(==(cas[2]), "KUT")
    if model == 4                                                     # the GLS-detrended trend case: the line integral only
        method == :double && throw(ArgumentError("case DG has the line integral representation only"))
        g0 != 0 && throw(ArgumentError("g0 (a fixed start) applies to cases DK, DU, DT only"))
        method = :single
    end
    model, stat, Float64(c0), Float64(g0), method
end

function ltratio_dist(kind::Int, x, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, kw...)
    model, stat, c0, g0, method = ltratio_args(cas, c; c0, g0, method)
    start = stat ? :stationary : :zero
    xs = x isa Number ? [Float64(x)] : Float64.(vec(collect(x)))
    F = [isnan(xk) ? NaN : xk == Inf ? (kind == 2 ? 1.0 : 0.0) : xk == -Inf ? 0.0 : 0.0 for xk in xs]
    fin = findall(isfinite, xs)
    dbl = [k for k in fin if method == :double || (method == :auto && automethod(model, Float64(c), c0, g0, xs[k]) == :double)]
    sgl = setdiff(fin, dbl)
    single(k) = kind == 2 ? cdf(xs[k], model; c = Float64(c), c0, g0, start, kw...) : density(xs[k], model; c = Float64(c), c0, g0, start, kw...)
    if !isempty(dbl)
        Fd, info = tr_limdist(xs[dbl], model, Float64(c); c0, g0, start, pdf = kind == 1, warn = method == :double, kw...)
        for (j, k) in enumerate(dbl)
            # a point chosen by the rule at which the double integral reports that its error estimate is not met gets the
            # single integral's value (with method = :double the flagged value is returned, with a warning)
            F[k] = (method == :auto && info.status[j] != 0) ? single(k) : Fd[j]
        end
    end
    for k in sgl; F[k] = single(k); end
    # the tails: the single integral's error is absolute (1e-14 to 1e-21, not reduced by finer rules), so with method = :auto
    # the values of the single integral below 1e-3 (a density in absolute value, a distribution function in the lower tail)
    # are recomputed by the double integral, which keeps its relative accuracy there (as the MATLAB programs do)
    if method == :auto
        tl = [k for k in sgl if (kind == 1 ? abs(F[k]) : F[k]) < 1e-3]
        if !isempty(tl)
            Ft, info = tr_limdist(xs[tl], model, Float64(c); c0, g0, start, pdf = kind == 1, warn = false, kw...)
            for (j, k) in enumerate(tl); info.status[j] == 0 && (F[k] = Ft[j]); end   # (where the double integral flags the point, the single integral's value stays)
        end
    end
    kind == 1 && (F[F .< 0] .= 0.0)                                   # a density below the error of the method
    x isa Number ? F[1] : reshape(F, size(x))
end

"""
    ltratio_cdf(x, cas, c = 0; c0 = 0, g0 = 0, method = :auto)

The limit as n -> inf of P[t <= x] for the least squares t-ratio t of the hypothesis phi = 1 + c0/n under phi = 1 + c/n,
case `cas` ("DK", "DU", "DT": fixed start g0, default 0; "SK", "SU", "ST": stationary start, c < 0; "DG": the GLS-detrended
trend case of Elliott, Rothenberg and Stock, cbar = -13.5, zero start, line integral only).  x may be an array.
Non-finite x: 0 at -Inf, 1 at Inf, NaN at NaN.  method: :auto (the faster of the two representations at each x, see
`automethod`; the values of the single integral below 1e-3 are recomputed by the double integral, and a point at which
the double integral reports that its error estimate is not met gets the single integral's value), :single (the single
integral of Theorem 2; with a fixed start the noncentral kernel) or :double (the double integral of Theorem S1 of the Supplementary Material, whose
flagged values are returned with a warning).  Further keywords go to the engine (tau, hu, N, pfac, ...).

    ltratio_cdf(-2.86, "DU")                        # P[tau_mu <= -2.86] at the unit root
    ltratio_cdf(-1.94087, "DK", -5)                 # power of the 5% test of tau at c = -5
    ltratio_cdf(-3.41, "ST", -10)                   # trend case, stationary start
    ltratio_cdf(0, "DK", 2; c0 = 2)                 # t-ratio centered at the true value
"""
ltratio_cdf(x, cas::AbstractString, c::Real = 0; kw...) = ltratio_dist(2, x, cas, c; kw...)
"""
    ltratio_pdf(x, cas, c = 0; c0 = 0, g0 = 0, method = :auto)

The limiting density of the t-ratio; arguments as in `ltratio_cdf`.  In case DK with c0 > 0 the density is infinite at
x = -sqrt(2 c0), where the two support edges of the kernel merge (a logarithmic singularity), and Inf is returned there.
"""
ltratio_pdf(x, cas::AbstractString, c::Real = 0; kw...) = ltratio_dist(1, x, cas, c; kw...)

"""
    ltratio_icdf(p, cas, c = 0; tol = 1e-10, c0 = 0, g0 = 0, method = :auto)

The limit of the p-quantile of the t-ratio, p in (0, 1) (an array), by Newton's method on `ltratio_cdf` with `ltratio_pdf`
as the derivative, safeguarded by bisection, started at the normal approximation from the first two moments.

    ltratio_icdf([0.01, 0.05, 0.10], "DU")          # the asymptotic critical values of tau_mu
"""
function ltratio_icdf(p, cas::AbstractString, c::Real = 0; tol::Real = 1e-10, c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, kw...)
    model, = ltratio_args(cas, c; c0, g0, method)
    if model == 4                                                     # (no moments for case DG: a rough start, the search is safeguarded)
        mu = -1.9 + 0.1min(c, 0.0); sd = 0.8
    else
        m = ltratio_mom(2, cas, c; c0, g0)
        mu = m[1]; sd = sqrt(max(m[2] - m[1]^2, 1e-6))
    end
    one(pk) = begin
        0 < pk < 1 || throw(ArgumentError("p must lie in (0, 1)"))
        x = mu + sd * sqrt(2) * erfinv(2pk - 1)
        lo = x - sd; hi = x + sd
        while ltratio_cdf(lo, cas, c; c0, g0, method, kw...) > pk; lo -= 2(x - lo); end
        while ltratio_cdf(hi, cas, c; c0, g0, method, kw...) < pk; hi += 2(hi - x); end
        for _ in 1:60
            F = ltratio_cdf(x, cas, c; c0, g0, method, kw...); f = ltratio_pdf(x, cas, c; c0, g0, method, kw...)
            F > pk ? (hi = x) : (lo = x)
            xn = x - (F - pk) / f
            (lo < xn < hi && isfinite(xn)) || (xn = (lo + hi) / 2)
            abs(xn - x) < tol && return xn
            x = xn
        end
        @warn "ltratio_icdf: Newton's method did not converge at p = $pk"
        x
    end
    p isa Number ? one(p) : map(one, p)
end

"""
    ltratio_mom(p, cas, c = 0; c0 = 0, g0 = 0) -> m

m[j] = E[t^j], j = 1..p, for the limit (U - c0 V)/sqrt(V) of the t-ratio under phi = 1 + c/n (Proposition 3; `moments`).

    ltratio_mom(4, "DU")[2]                         # 3.0559953450 (Nabeya, 1999)
"""
function ltratio_mom(p::Int, cas::AbstractString, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, kw...)
    p >= 1 || throw(ArgumentError("p must be a positive integer"))
    model, stat, c0, g0, _ = ltratio_args(cas, c; c0, g0)
    model == 4 && throw(ArgumentError("the moments are not available for case DG"))
    moments(model; c = Float64(c), c0, g0, start = stat ? :stationary : :zero, pmax = p, kw...)
end

# =====================================================================================
# The local null value c0: the general kernel
#
# The t-ratio for phi0 = 1 + c0/n has the limit (U - c0 V)/sqrt(V).  In the derivation of Theorem 2 only the
# region changes: {U <= x sqrt(V)} becomes {U <= x sqrt(V) + c0 V}, so with V = v^2 the kernel runs along the
# parabola u = x v + c0 v^2 instead of the line u = x v, with the same Jacobian 2 v^2.  With
# X(v) = c0 v^2 + x v + 1/2 the transforms are
#   density:  I_f(t) = int_0^inf 2 v^2 e^{-t v^2} phi_t(X(v)) dv,
#   cdf:      I_F(t) = int_0^inf 2 t v e^{-t v^2} Psi_t(X(v)) dv
#                    = Psi_t(1/2) + int_0^inf e^{-t v^2} (x + 2 c0 v) phi_t(X(v)) dv,
# Psi_t the distribution function of phi_t, and f = (1/2 pi i) int P I_f dt, F = (1/2 pi i) int P I_F dt/t as
# before.  The support edge X = 0 of phi_t (case DK), or its logarithmic singularity, is met at the positive
# roots of X(v): none, one or two of them.  Each root b contributes a part carrying exp(-t b^2), which
# oscillates in Im t with the period 2 pi/b^2 and gets its own Filon part in the outer integral; for c0 < 0
# there is always one root, also for x > 0.  Between the roots X < 0 (phi_neg; nothing in case DK).  For
# c0 > 0 and x < 0 without real roots, the vertex v0 = -x/(2 c0) of the parabola is a breakpoint too: the
# complex roots v0 +- i l of X are at the distance l = sqrt(Xmin/c0) from the real axis, and the paths of the
# quadrature, which leave the real axis, must not cross the cuts of X^{-1/2} or log X that run from them away
# from the real axis; the segment [0, v0] is bulged and the ray starts at v0, as at an edge, and the part
# near v0 carries exp(-t v0^2).  Along each ray from a breakpoint b the exponent of the kernel and of phi_t is
#   -t v^2 - s_d X(v) = -t b^2 - s_d X(b) - beta r - alpha r^2,   v = b + r,
# alpha = t + c0 s_d, beta = 2 alpha b + s_d x, s_d = s_1 (X < 0, or case DK) or s_2, so the ray and saddle
# paths of gaussexp apply with (alpha, beta) in place of (t/q^2, s_d).  (Raymond Kan's extension of the
# MATLAB port, ported back.)

"""The breakpoints of the v-integrals of inner_gen for the kernel along u = q v + c0 v^2: X(v) = c0 v^2 + q v + x0
(x0 = 1/2, or (1 + g0^2)/2 in case DK with a fixed start, where the support edge of R + g0^2/2 is at 0) = c0 (v - r1)(v - r2) (or q (v - r1) for c0 = 0, cancellation-free near the roots), its positive roots b (edges,
kindb = 1) and, for c0 > 0, q < 0 without real roots, the vertex (kindb = 0); y = b^2 the frequencies of the parts;
ell the scale of the singularity near each breakpoint (the distance of the nearest other root, real or complex),
ell0 the same at v = 0; sgn the sign of X on the segments [0, b_1], ..., [b_ne, inf); negl the parts that are
negligible on the line Re t = tau; vsad the saddle of the exponent on the first segment (NaN when irrelevant)."""
mutable struct Breaks
    q::Float64; c0::Float64
    rr::Vector{ComplexF64}; lead::Float64
    b::Vector{Float64}; y::Vector{Float64}; kindb::Vector{Int}; ell::Vector{Float64}; ne::Int
    ell0::Float64; sgn::Vector{Int}; negl::Vector{Bool}; dbl::Bool; vsad::Float64
end
"""X at v = b + dv, cancellation-free near a root b: lead * prod_k (dv + (b - r_k))."""
@inline function xat(R::Breaks, dv, b)
    X = complex(R.lead) * one(dv)
    for r in R.rr; X *= (dv + (b - r)); end
    X
end
dXv(R::Breaks, v) = 2R.c0 * v + R.q
function breakpts(q::Float64, c0::Float64, tau::Float64; x0::Float64 = 0.5)
    if c0 == 0
        rr = [complex(-x0 / q)]; lead = q; r = [-x0 / q]
    else
        d = q * q - 4c0 * x0
        sq = d >= 0 ? complex(sqrt(d)) : im * sqrt(-d)
        rr = [(-q - sq) / (2c0), (-q + sq) / (2c0)]; lead = c0
        r = d >= 0 ? sort(real.(rr)) : Float64[]
    end
    b = [x for x in r if x > 0]; noroot = isempty(b)
    # a root so far out that e^{-t v^2} is below e^-40 there on the line Re t = tau (3 tau b^2/4 > 40) is dropped:
    # the part near it is negligible and the segment up to it would be integrated at a cost that grows with b
    # (for c0 = 0 and q -> 0- the root -1/(2 q) runs off to infinity)
    # (for c0 = 0 only: with c0 != 0 X changes sign at the root, and the segment beyond it has the other form, phi_neg for
    # phi_pos, with the other exponential factor, whose Gaussian decay alpha = t + c0 s_d the near side's form lacks;
    # dropping it ran the near side's form out to infinity, "Re alpha <= 0 on a ray" or a wrong value: Raymond Kan,
    # November 2026, e.g. SU, c = -1.2, c0 = -1.2 at x = 9.  The edge part of a far root is skipped as negligible, negl.)
    c0 == 0 && (b = [x for x in b if 3tau * x * x / 4 <= 40])
    kindb = ones(Int, length(b)); ell = fill(Inf, length(b)); dbl = false
    if length(b) == 2 && b[2] - b[1] <= 1e-9 * b[2]              # a double root (x = -sqrt(2 c0), c0 > 0)
        b = [(b[1] + b[2]) / 2]; kindb = [1]; ell = [Inf]; dbl = true
    elseif length(b) == 2
        ell .= b[2] - b[1]
    end
    if c0 > 0 && q < 0 && noroot
        b = [-q / (2c0)]; kindb = [0]; ell = [sqrt(max(x0 - q * q / (4c0), 0.0) / c0)]
    end
    for k in eachindex(b)                                        # the distance of the nearest other root
        dr = [abs(b[k] - r) for r in rr]; dr = [x for x in dr if x > 1e-9 * max(b[k], 1.0)]
        isempty(dr) || (ell[k] = min(ell[k], minimum(dr)))
    end
    ell = max.(ell, 1e-8 .* max.(b, 1.0))                        # (a double root: the density is infinite there)
    ell0 = minimum(abs.(rr))                                     # the distance of the nearest root from v = 0
    ne = length(b)
    ed = [0.0; b; Inf]; sgn = ones(Int, ne + 1)
    R = Breaks(q, c0, rr, lead, b, b .^ 2, kindb, ell, ne, ell0, sgn, falses(ne), dbl, NaN)
    for j in 1:ne+1
        vm = isfinite(ed[j+1]) ? (ed[j] + ed[j+1]) / 2 : ed[j] + 1
        sv = sign(real(xat(R, vm, 0.0))); sv != 0 && (R.sgn[j] = Int(sv))
    end
    mid = [(ed[j] + ed[j+1]) / 2 for j in 1:ne]
    R.negl = [3tau * m * m / 4 > 40 for m in mid]                 # the edge part is below e^-40 there
    R
end

"""phi_t without its exponential factor e^{-s_d X}: algebraic, bounded away from X = 0."""
phialg(::Val{1}, X, sgn, s1, s2) = 1 / sqrt(pi * X)
phialg(::Val, X, sgn, s1, s2) = sgn > 0 ? k0x((s2 - s1) / 2 * X) / pi : k0x(-(s2 - s1) / 2 * X) / pi
kernof(::Val{k}) where {k} = k

# ---------- the fixed start g0 != 0 through the single integral: the noncentral kernels ----------
# Under the tilt e^{tV} with a fixed start, R = U + 1/2 is a noncentral form of the same rank (Proposition 2): in case DK
# R + g0^2/2 = J(1)^2/2 with J(1) ~ N(m, 1/s1), m = g0/F_0, and in cases DU, DT R = (X1^2 - X2^2)/2 with X1 ~ N(m1, 1/s2),
# X2 ~ N(m2, -1/s1) (s2 the root with positive real part, s1 the other), m_k^2 = g0^2 mu_k from (partial).  The
# transform is M_{g0}(0,t) = M_0(0,t) e^{g0^2 Q(0,t)}: pref = e^{g0^2 Q(0,t)} multiplies phi_t, and phi_t itself is the
# density of R with the fixed start, normalized as the central one (total mass M0 = (-s1 s2)^{-1/2}, or s1^{-1/2}).
# Case DK: phi_t(X~) = pref e^{-s1 m^2/2} e^{-s1 X~} cosh(m s1 sqrt(2 X~))/sqrt(pi X~) with X~ = X + g0^2/2 (the code's X,
# with x0 = (1 + g0^2)/2 in the parabola), elementary; the edge is at X~ = 0.
# Cases DU, DT: integrating out X2 (l = |X2|),
#   phi_t(X) = pref (2/pi) e^{(s1 m2^2 - s2 m1^2)/2} e^{-s2 X} G(X; a, m2 s1, m1 s2)        (X > 0),
#   phi_t(X) = pref (2/pi) e^{(s1 m2^2 - s2 m1^2)/2} e^{-s1 X} G(-X; a, m1 s2, m2 s1)      (X < 0),
#   G(Y; a, b1, b2) = int_0^inf e^{-a l^2/2} cosh(b1 l) cosh(b2 sqrt(2Y + l^2)) (2Y + l^2)^{-1/2} dl,  a = s2 - s1,
# analytic in Y on Re Y > 0 (and beyond), G(Y; a, 0, 0) = e^{aY/2} K_0(aY/2)/2, so that m = 0 gives back the K_0 form.
# The kink at X = 0 is where it was, and the exponential factors e^{-s_d X} are those of the central kernel, so the paths
# and partitions of inner_gen apply unchanged; the noncentral factor grows at most like e^{lam} along them, lam/2 =
# m^2 s/2 the Poisson parameters, which are below 1 on the line for |c g0| <= 5 and below 10 for c <= 5, g0 = 1 (the
# pieces of the paths stop where the central exponent is below -EXPO; the error is of the order e^{lam - EXPO}).
"""the noncentral kernel at one point t of the line: m1, m2 the means (case DK: m1 = m), pref = exp(g0^2 Q(0,t)), g0"""
struct NCK{K}
    m1::ComplexF64; m2::ComplexF64; pref::ComplexF64; g0::Float64
end
kernof(::NCK{k}) where {k} = k
const NLNC = Ref(40)       # nodes of the l-rule of G (40 agrees with 384 to 1e-14; 32 loses 3e-13)
"""G(Y; a, b1, b2): l = al sinh T, al = sqrt|2Y|, Gauss-Legendre in T up to where the Gaussian, against the growth of the
cosh factors, is below e^{-41.4}; sqrt(2Y + l^2) formed as al sqrt(2Y/al^2 + sinh^2 T), the principal branch, continuous
in l since Im(2Y) is constant along the rule and Re(2Y + l^2) > 0 where it vanishes."""
const NLMAX = 256         # bounds the node counts of the l-rule
"""the nodes and weights of the l-rule for int_0^lm e^{-a l^2/2 + bb l + i(...)} (...) dl with the edge scale al (the
factor (2Y + l^2)^{-1/2} varies on it): l = al sinh T with Gauss-Legendre nodes in T, at least Nl, more where the peak of
the Gaussian times the growth, at l* = bb/Re a with the width sigma = (Re a)^{-1/2}, is narrow in T (at most NLMAX).  (At
large noncentrality, with the peak separated from the edge scale, the integral runs along the saddle line instead: ncsaddle.)"""
function lrule(al::Float64, lm::Float64, bb::Float64, ra::Float64, ai::Float64, bi::Float64, Nl::Int)
    lstar = bb / ra; sig = 1 / sqrt(ra)
    if lstar - 8sig > 2al
        l1 = lstar - 8sig; lhi = min(lm, lstar + 8sig)
        dphi = abs(ai) * (lhi * lhi - l1 * l1) / 2 + bi * (lhi - l1)
        nB = clamp(8 * cld(ceil(Int, 1.2dphi + 24), 8), Nl, NLMAX)
        zl, wl = rule(Nl); zB, wB = rule(nB)
        tm = asinh(l1 / al)
        ls = Vector{Float64}(undef, Nl + nB); ws = Vector{Float64}(undef, Nl + nB)
        @inbounds for k in 1:Nl; T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; end
        @inbounds for k in 1:nB; ls[Nl+k] = l1 + (lhi - l1) * zB[k]; ws[Nl+k] = (lhi - l1) * wB[k]; end
        return ls, ws
    end
    tm = asinh(lm / al)
    dT = sig / sqrt(al * al + lstar * lstar)                      # the width of the peak in T
    n = clamp(8 * cld(ceil(Int, 3tm / dT), 8), Nl, NLMAX)
    zl, wl = rule(n)
    ls = Vector{Float64}(undef, n); ws = Vector{Float64}(undef, n)
    @inbounds for k in 1:n; T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; end
    ls, ws
end
function ncG(Y::ComplexF64, a::ComplexF64, b1::ComplexF64, b2::ComplexF64)
    # along the ray l = e s, e = e^{-i arg(a)/2}, the Gaussian is e^{-|a| s^2/2} (no oscillation) and r = e (2Y e^{i arg a} +
    # s^2)^{1/2}: the same integral with the arguments (Y e^{i arg a}, |a|, b1 e, b2 e), whose cut in Y lies where a Y is
    # negative real, as the cut of K_0((s2 - s1) X/2) of the central kernel does, so the paths of inner_gen, which avoid that
    # cut, serve; with the real-axis rule the cut would lie at negative real Y instead
    ph = angle(a); e = exp(-im * ph / 2)
    Y = Y * exp(im * ph); a = complex(abs(a)); b1 = b1 * e; b2 = b2 * e
    ra = max(real(a), 1e-300); bb = abs(b1) + abs(b2)
    lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra
    al = max(sqrt(abs(2Y)), 1e-8 * lm)
    ls, ws = lrule(al, lm, abs(real(b1)) + abs(real(b2)), ra, imag(a), abs(imag(b1)) + abs(imag(b2)), NLNC[])
    Y2 = 2Y / (al * al)
    # the terms as e^{E_k - Eref} with Eref the largest real exponent: G = S e^{Eref}, S bounded (the exponents reach
    # 2 lam at large noncentrality, beyond the range of a double)
    n = length(ls)
    Es = Vector{ComplexF64}(undef, n); rs = Vector{ComplexF64}(undef, n); fs = Vector{ComplexF64}(undef, n)
    Eref = -Inf
    @inbounds for k in 1:n
        l = ls[k]; sh = l / al
        r = al * sqrt(Y2 + sh * sh)
        z1 = b1 * l; real(z1) < 0 && (z1 = -z1)
        z2 = b2 * r; real(z2) < 0 && (z2 = -z2)
        # cosh z = e^z (1 + e^{-2z})/2 for Re z >= 0: one exponential of the combined exponent
        Es[k] = -a * l * l / 2 + z1 + z2; rs[k] = r; fs[k] = (1 + exp(-2z1)) * (1 + exp(-2z2)) / 4
        Eref = max(Eref, real(Es[k]))
    end
    S = zero(ComplexF64)
    @inbounds for k in 1:n; S += exp(Es[k] - Eref) * fs[k] / rs[k] * ws[k]; end
    S, Eref
end
"""P[R <= 1/2] under the tilt with a fixed start, cases DU, DT: 2 int_0^inf phibar(l; m2, w2) Fbar(sqrt(1 + l^2); m1, w1) dl,
X1 ~ N(m1, w1 = 1/s2), X2 ~ N(m2, w2 = -1/s1), by the l-rule (the Gaussian in l peaks at |m2|)"""
function ncP(s1::ComplexF64, s2::ComplexF64, m1::ComplexF64, m2::ComplexF64, Nl::Int)
    w1 = 1 / s2; w2 = -1 / s1
    ra = max(real(1 / w2), 1e-300); bb = abs(m2 / w2)
    lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra
    ls, ws = lrule(1.0, lm, abs(real(m2 / w2)), ra, imag(1 / w2), abs(imag(m2 / w2)), Nl)
    sq1 = sqrt(2w1); c2 = 1 / (2sqrt(2pi * w2))
    P = zero(ComplexF64)
    @inbounds for k in eachindex(ls)
        l = ls[k]; u = sqrt(1 + l * l)
        f2 = (exp(-(l - m2)^2 / (2w2)) + exp(-(l + m2)^2 / (2w2))) * c2
        F1 = (erf((u - m1) / sq1) + erf((u + m1) / sq1)) / 2
        P += f2 * F1 * ws[k]
    end
    2P
end
phialg(K::NCK{1}, X, sgn, s1, s2) = K.pref * exp(-s1 * K.m1 * K.m1 / 2) * cosh(K.m1 * s1 * sqrt(2X)) / sqrt(pi * X)
"""phialg of cases DU, DT as (S, E): phialg = S e^E, S bounded"""
function phialg_scaled(K::NCK{2}, X, sgn, s1, s2)
    a = s2 - s1
    cst = K.pref * (2 / pi) * exp((s1 * K.m2 * K.m2 - s2 * K.m1 * K.m1) / 2)
    S, E = sgn > 0 ? ncG(X, a, K.m2 * s1, K.m1 * s2) : ncG(-X, a, K.m1 * s2, K.m2 * s1)
    cst * S, E
end
function phialg(K::NCK{2}, X, sgn, s1, s2)
    S, E = phialg_scaled(K, X, sgn, s1, s2)
    S * exp(E)
end
# The noncentral density is a Gaussian in sqrt(2 X~) centered at the mean (X~ = X + g0^2/2 in case DK, X or -X in cases
# DU, DT): e^{-s_d X~} cosh(b sqrt(2 X~)) with b = m s_d, so that its mass sits at X~ = m^2/2, away from the edge, by the
# factor e^{lam}, lam = m^2 s_d/2.  The paths of inner_gen are built from the exponent; for the noncentral kernels it is
# E = -t v^2 - s_d X + ncexp, ncexp = log cosh(b sqrt(2 X~)), and the factor carried by the nodes is phialg e^{-ncexp},
# which is bounded.  (For the central kernels ncexp = 0.)
"""log cosh z without overflow: z_s + log(1 + e^{-2 z_s}) - log 2 with Re z_s >= 0"""
@inline function logcosh(z::ComplexF64)
    zs = real(z) >= 0 ? z : -z
    zs + log1p(exp(-2zs)) - 0.6931471805599453
end
"""the coefficient b of the noncentral exponent on the side sgn (kern 1: one side)"""
ncb(K::NCK{1}, sgn, s1, s2) = K.m1 * s1
ncb(K::NCK{2}, sgn, s1, s2) = sgn > 0 ? K.m1 * s2 : K.m2 * s1
ncexp(::Val, X, sgn, s1, s2) = zero(ComplexF64)
ncexp(K::NCK, X, sgn, s1, s2) = logcosh(ncb(K, sgn, s1, s2) * sqrt(2 * (sgn > 0 ? X : -X)))
"""d ncexp / dv = b^2 (tanh z / z) (+-dX/dv), z = b sqrt(2 X~): finite at the edge"""
ncexp_dv(::Val, X, sgn, s1, s2, dXdv) = zero(ComplexF64)
function ncexp_dv(K::NCK, X, sgn, s1, s2, dXdv)
    b = ncb(K, sgn, s1, s2); Xt = sgn > 0 ? X : -X; z = b * sqrt(2Xt)
    tz = abs(z) < 1e-6 ? one(z) : tanh(z) / z
    b * b * tz * (sgn > 0 ? dXdv : -dXdv)
end
"""phialg without the noncentral growth, as (S, E) with phired = S e^E and S bounded (the exponent E reaches the size of
lam at large noncentrality): the central kernels and case DK with E = 0; cases DU, DT by the l-rule of ncG, or along the
saddle line (ncsaddle) when the Gaussian of X_2 is separated from the edge scale"""
const USE_SADDLE = Ref(false)   # the saddle line of ncsaddle (experimental; the composite l-rule serves)
phired_scaled(V, X, sgn, s1, s2) = (phialg(V, X, sgn, s1, s2) * exp(-ncexp(V, X, sgn, s1, s2)), 0.0)
function phired_scaled(K::NCK{2}, X, sgn, s1, s2)
    w2 = -1 / s1; Xt = sgn > 0 ? X : -X
    USE_SADDLE[] && abs(K.m2) - 8sqrt(abs(w2)) > 2sqrt(abs(2Xt)) && return ncsaddle(K, X, sgn, s1, s2)
    S, E = phialg_scaled(K, X, sgn, s1, s2)
    S, E - ncexp(K, X, sgn, s1, s2)
end
phired(V, X, sgn, s1, s2) = ((S, E) = phired_scaled(V, X, sgn, s1, s2); S * exp(E))
"""the square roots of q at the nodes of a path, continuous: the root with positive real part at the first node, then the
root nearer the previous value (the nodes in path order)"""
function sqrt_cont(q::AbstractVector{ComplexF64})
    u = similar(q)
    u0 = sqrt(q[1]); real(u0) < 0 && (u0 = -u0); u[1] = u0
    @inbounds for k in 2:length(q); r = sqrt(q[k]); u[k] = abs(r - u[k-1]) <= abs(r + u[k-1]) ? r : -r; end
    u
end
"""the nodes of a line through a saddle, s in [-S, S], in two chains from the center outward (for the branch tracking), with
the Gauss-Legendre weights; n nodes"""
function saddle_nodes(n::Int, S::Float64)
    z, w = rule(n)
    s = S .* (2 .* z .- 1); ws = 2S .* w
    ip = findall(>=(0), s); im_ = findall(<(0), s)
    sort!(ip; by = k -> s[k]); sort!(im_; by = k -> -s[k])
    s, ws, ip, im_
end
"""phired of cases DU, DT at large noncentrality, as (S, E): the integral over X_2 along the line through the saddle of its
Gaussian, on which that Gaussian is e^{-s^2/2} and nothing cancels (on the real axis the terms exceed the integral by e^{lam}).
X > 0: l = m_2 + w_2^{1/2} s, u = (l^2 + 2X)^{1/2} continuous from s = 0; X < 0: u = u_s + d s with u_s = (m_2^2 - 2X~)^{1/2}
and d = (m_2/u_s) w_2^{1/2}, l = (u^2 + 2X~)^{1/2}.  The node count from the change of the exponent along the line."""
function ncsaddle(K::NCK{2}, X, sgn, s1, s2)
    w1 = 1 / s2; w2 = -1 / s1; m1 = K.m1; m2 = K.m2
    real(m2) < 0 && (m2 = -m2)
    sq2 = sqrt(w2); Xt = sgn > 0 ? X : -X
    pre = K.pref / (sqrt(-s1) * sqrt(s2))                            # the mass M0
    Sl = 9.0
    if sgn > 0
        us0 = sqrt(m2 * m2 + 2X); real(us0) < 0 && (us0 = -us0)
        real(m1 * us0 / w1) < 0 && (m1 = -m1)
        cst = s2 * X - ncexp(K, X, sgn, s1, s2)
        Eof = (s, u) -> -s * s / 2 - (u - m1)^2 / (2w1) + cst
        # the node count from the change of the exponent along the line and the width of the Gaussian of X_1 in s
        sc = range(-Sl, Sl; length = 33); uc = sqrt_cont([(m2 + sq2 * s)^2 + 2X for s in sc])
        Ec = [Eof(sc[k], uc[k]) for k in eachindex(sc)]
        dph = sum(abs(Ec[k+1] - Ec[k]) for k in 1:length(sc)-1)
        n = clamp(8 * cld(ceil(Int, 1.2dph + 24 + 6Sl * abs(sq2) / sqrt(abs(w1))), 8), 64, 2NLMAX)
        s, ws, ip, im_ = saddle_nodes(n, Sl)
        u = Vector{ComplexF64}(undef, n)
        u[ip] = sqrt_cont([(m2 + sq2 * s[k])^2 + 2X for k in ip]); u[im_] = sqrt_cont([(m2 + sq2 * s[k])^2 + 2X for k in im_])
        E = [Eof(s[k], u[k]) for k in 1:n]; Eref = maximum(real, E)
        acc = zero(ComplexF64)
        @inbounds for k in 1:n; acc += ws[k] * exp(E[k] - Eref) * (1 + exp(-2m1 * u[k] / w1)) / u[k]; end
        return pre / (2pi * sqrt(w1)) * acc, Eref
    end
    us = sqrt(m2 * m2 - 2Xt); real(us) < 0 && (us = -us)
    d = m2 / us * sq2
    real(m1 * us / w1) < 0 && (m1 = -m1)
    cst = s1 * X - ncexp(K, X, sgn, s1, s2)
    Eof = (u, l) -> -(u - m1)^2 / (2w1) - (l - m2)^2 / (2w2) + cst
    sc = range(-Sl, Sl; length = 33); ucs = [us + d * s for s in sc]; lc = sqrt_cont([uu * uu + 2Xt for uu in ucs])
    Ec = [Eof(ucs[k], lc[k]) for k in eachindex(sc)]
    dph = sum(abs(Ec[k+1] - Ec[k]) for k in 1:length(sc)-1)
    n = clamp(8 * cld(ceil(Int, 1.2dph + 24 + 6Sl * abs(d) / sqrt(abs(w1))), 8), 64, 2NLMAX)
    s, ws, ip, im_ = saddle_nodes(n, Sl)
    u = [us + d * s[k] for k in 1:n]; l = Vector{ComplexF64}(undef, n)
    l[ip] = sqrt_cont([u[k]^2 + 2Xt for k in ip]); l[im_] = sqrt_cont([u[k]^2 + 2Xt for k in im_])
    E = [Eof(u[k], l[k]) for k in 1:n]; Eref = maximum(real, E)
    acc = zero(ComplexF64)
    @inbounds for k in 1:n; acc += ws[k] * d * exp(E[k] - Eref) * (1 + exp(-2m1 * u[k] / w1)) * (1 + exp(-2m2 * l[k] / w2)) * u[k] / l[k]; end
    pre / (2pi * sqrt(w1) * sqrt(w2)) * acc, Eref
end
"""the kernel of the fixed start g0 at the point t: the means and pref from the Girsanov forms (case DK) or from the
partial fractions (partial) of Q(s,t) (cases DU, DT; the determinant form of Q for c <= CGIRS, the Girsanov form beyond)"""
function nckernel(model::Int, c::Float64, t::ComplexF64, g0::Float64, s1::ComplexF64, s2::ComplexF64)
    if model == 1
        G = girsanov(1, c, t)
        return NCK{1}(g0 * exp(-G.lsh) / G.eta, zero(ComplexF64), exp(g0 * g0 * t / G.eta), g0)
    end
    Q0, m1s, m2s = qpieces(model, c, t, g0, 1 / s2, -1 / s1, c > CGIRS ? qstart : qstart_d)
    NCK{2}(sqrt(m1s), sqrt(m2s), exp(Q0), g0)
end

"""Psi_t(1/2) = int_{-inf}^{1/2} phi_t(X) dX: closed form in case DK, otherwise the total mass minus a short
integral over X > 1/2 (or the integral over X < 1/2 when |s2| is small); with a fixed start, the noncentral probability
P[J(1)^2 <= 1 + g0^2] (case DK) or P[R <= 1/2] by the l-rule of Pkap (cases DU, DT), times the mass."""
psi_half(::Val{1}, s1, s2, M0) = erf(sqrt(s1 / 2)) / sqrt(s1)
function psi_half(K::NCK{1}, s1, s2, M0)
    r = sqrt(1 + K.g0 * K.g0); w = sqrt(2 / s1)
    K.pref / sqrt(s1) * (erf((r - K.m1) / w) + erf((r + K.m1) / w)) / 2
end
psi_half(K::NCK{2}, s1, s2, M0) = K.pref * M0 * ncP(s1, s2, K.m1, K.m2, NLNC[])
function psi_half(V::Val, s1, s2, M0)
    l = 1 / (abs(s2 - s1) + 2)
    if abs(s2) >= abs(s1) / 4
        e2 = exp(-im * angle(s2))
        return M0 - rayint(r -> phi_pos(V, 0.5 + r * e2, s1, s2) * e2, 36 / abs(s2), l, 40, 32, 1)
    end
    e1 = exp(-im * angle(-s1))
    rayint(X -> phi_pos(V, X, s1, s2), 0.5, l, 40, 32, 4) + rayint(r -> phi_neg(V, -r * e1, s1, s2) * e1, 36 / abs(s1), l, 40, 32, 4)
end

"""Gauss-Legendre nodes and weights on [a, b] with y = a + (b - a) u^p."""
function glnodes(a::Float64, b::Float64, n::Int, p::Int)
    n = n * NODEFAC[]
    u, gw = rule(n)
    L = b - a
    y = Vector{Float64}(undef, n); wt = Vector{Float64}(undef, n)
    for k in 1:n
        if p == 1
            y[k] = a + L * u[k]; wt[k] = L * gw[k]
        elseif p == 2
            y[k] = a + L * u[k]^2; wt[k] = 2L * u[k] * gw[k]
        else
            y[k] = a + L * u[k]^4; wt[k] = 4L * u[k]^3 * gw[k]
        end
    end
    y, wt
end

"""int_0^inf A(r) exp(-alpha r^2 - beta r) dr, Re alpha > 0: along the ray r e^{-i arg(alpha)/2}, on which the Gaussian
decays, while the exponential grows by at most e^2 and turns by at most 150 rad along it; otherwise by the saddle or
level-curve path of gaussexp.  The cap is 150 rad, not the 60 of `ray_ok`: the level-curve path, taken when the saddle
z = beta / (2 sqrt(alpha)) has Re z >= 0, passes the branch point of its square root at the distance |z| from the
origin, where the Gaussian is e^{-|z|^2}; at 60 rad it took over at |Im z| ~ 4, where that is 1e-7, and the inner
transform lost seven digits on a short stretch of the line (1e-10 in the distribution function in case DK with c0 > 0
at x >= 0); at 150 rad the level curve begins at |Im z| ~ 10, where the branch point is below 1e-40."""
function rayexp(A, alpha::ComplexF64, beta::ComplexF64, lA::Float64, p::Int)
    real(alpha) > 0 || error("Re alpha <= 0 on a ray")
    e = exp(-im * angle(alpha) / 2); aa = abs(alpha)
    if max(0.0, -real(beta * e))^2 / (4aa) <= 2 && abs(imag(beta * e)) * sqrt(EXPO / aa) <= 150
        Rr = sqrt(EXPO / aa)
        real(beta * e) > 0 && (Rr = min(Rr, EXPO / real(beta * e)))
        f = r -> A(r * e) * exp(-aa * r * r - beta * e * r) * e
        return rayint(f, Rr, lA, 48, 32, p; osc = abs(imag(beta * e)))
    end
    gaussexp(A, alpha, 1.0, beta, lA; p = p)
end

const NGROW = 16; const NLEV = 5; const NCON = 16   # node counts of the segment pieces (the least nodes of a growing piece,
                                                    # the grading levels of the connection, the least nodes of each of its sub-pieces)
"""nodes and weights for int_0^L: the first piece [0, first] (shortened, by halving, until the exponent falls by at most 4,
rises by at most 4 (the noncentral kernels rise away from the edge) and its phase changes by at most 40 rad across it) with
nmin nodes and the substitution p, then pieces growing by the factor 4,
with the node counts set by the change of the phase of the exponent across each piece (at most 0.85 rad per node, in
sub-pieces of at most 160 nodes), until the exponent is below -EXPO."""
function pieces_nodes(Ef, L::Float64, first::Float64, p::Int, nmin::Int)
    first = min(first, L)
    for _ in 1:60
        E = [Ef(first * i / 8) for i in 0:8]
        ((real(E[end]) > -4 && maximum(real(Ek) for Ek in E) - real(E[1]) < 4 && sum(abs(imag(E[i+1]) - imag(E[i])) for i in 1:8) < 40) || first <= 1e-12 * L) && break
        first /= 2
    end
    y, wt = glnodes(0.0, first, nmin, p)
    a = first
    while a < L
        bn = min(L, 4a)
        E = [Ef(a + (bn - a) * i / 8) for i in 0:8]
        all(real(Ek) < -EXPO for Ek in E) && break
        dph = sum(abs(E[i+1] - E[i]) for i in 1:8)                   # the change of the exponent (phase and modulus)
        n = max(NGROW, 16 * cld(ceil(Int, 1.2dph + 16), 16))
        m = cld(n, 160); n = 16 * cld(cld(n, m), 16)                  # at most 160 nodes per sub-piece
        for k in 1:m
            y2, w2 = glnodes(a + (bn - a) * (k - 1) / m, a + (bn - a) * k / m, n, 1)
            append!(y, y2); append!(wt, w2)
        end
        real(E[end]) < -EXPO && break
        a = bn
    end
    y, wt
end

"""nodes and weights for int_0^1: Gauss-Legendre in sub-pieces graded geometrically toward both ends (NLEV levels, the
shortest 2^-NLEV), with the node counts from the change of the exponent (phase and modulus, sampled at 9 points of each
sub-piece, so that a stationary point of the phase inside is seen), skipping sub-pieces on which the exponent is below
max(E1, E2) - EXPO (up to 2: the descent pieces end where the exponent has fallen by EXPO), E1 and E2 the exponents at
the two ends of the segment."""
function phase_nodes(Ef, E1::ComplexF64, E2::ComplexF64)
    g = [2.0^(-k) for k in NLEV:-1:1]
    ss = [0.0; g; [1 - g[k] for k in length(g)-1:-1:1]; 1.0]
    ns = length(ss) - 1
    Emax = max(real(E1), real(E2)) - EXPO + 2
    y = Float64[]; wt = Float64[]
    for k in 1:ns
        E = [Ef(ss[k] + (ss[k+1] - ss[k]) * i / 8) for i in 0:8]
        maximum(real(Ek) for Ek in E) < Emax && continue
        dph = sum(abs(E[i+1] - E[i]) for i in 1:8)
        n = max(NCON, 8 * cld(ceil(Int, 1.2dph + 16), 8))              # (+16: a stationary point of the phase inside)
        m = cld(n, 160); n = 8 * cld(cld(n, m), 8)
        for i in 1:m
            y2, w2 = glnodes(ss[k] + (ss[k+1] - ss[k]) * (i - 1) / m, ss[k] + (ss[k+1] - ss[k]) * i / m, n, 1)
            append!(y, y2); append!(wt, w2)
        end
    end
    y, wt
end

"""the unit direction u of steepest descent of E(v0 + z) = E(v0) + gam z - alpha z^2 at an end, kept within deg degrees
of the direction dir (+1: into the segment from its left end, -1: from its right end) and, when gam = 0, vertical toward
-i sg; and the length l along it: where the exponent has fallen by EXPO, or the turning point of its quadratic part, or
the depth dep, or the half-length L/2 of the segment."""
function descent(gam::ComplexF64, alpha::ComplexF64, dir::Int, L::Float64, dep::Float64, sg::Float64, deg::Float64)
    u = abs(gam) > 0 ? -conj(gam) / abs(gam) : complex(0.0, -sg)
    cmax = cosd(deg)
    if real(u) * dir < cmax                                       # clamp to deg degrees from dir
        si = sign(imag(u)); si == 0 && (si = -sg)
        u = complex(dir * cmax, si * sind(deg))
    end
    g1 = real(gam * u); g2 = -real(alpha * u * u)                  # Re E(v0 + s u) - E(v0) = g1 s + g2 s^2
    l = L / 2
    abs(imag(u)) > 0 && (l = min(l, dep / abs(imag(u))))
    if g1 < 0
        g2 > 0 && (l = min(l, -g1 / (2g2)))                        # the turning point of the parabola
        disc = g1 * g1 - 4g2 * EXPO                                # g2 s^2 + g1 s + EXPO = 0: fallen by EXPO
        if g2 == 0
            l = min(l, -EXPO / g1)
        elseif disc >= 0
            sE = (-g1 - sqrt(disc)) / (2g2); sE > 0 && (l = min(l, sE))
        end
    end
    max(l, 1e-12 * L), u
end

"""The integral over the segment [a, b] of W(v) e^{-t v^2} phi_t(X(v)), split by the partition chi into the part near b (vr)
and the part near a (vs: the bump at a = 0, or vl: the edge at a).  The path has the fixed endpoints a and b (so that each
part is independent of it and analytic in t) and three straight pieces: from a in the direction of steepest descent of
the exponent E = -t v^2 - s_d X(v) at a, from b likewise, both kept within 80 degrees of the segment and within the depth
L/8 (|chi| <= e^{2.25} there) and stopped where the quadratic term turns the exponent up, and the straight connection
between their ends.  Near an end where e^{-s_d X} is still large and the Gaussian oscillates, the descent direction may
point into the other half plane than the Gaussian's; the connection then crosses the real axis where both factors are
small.  The nodes come from pieces_nodes (geometric pieces, node counts from the phase of the exponent, stopping where the
exponent is below -60) and, for the connection, phase_nodes; the integrand and the partition are evaluated once at all
of them."""
function segint(V, tj::ComplexF64, sd::ComplexF64, sgn::Int, s1::ComplexF64, s2::ComplexF64, W, R::Breaks, a::Float64, b::Float64,
                chi, pa::Int, pb::Int, ella::Float64, ellb::Float64, ws::Bool, wl::Bool, wr::Bool, sg::Float64, ka::Int, kb::Int, dep::Float64)
    L = b - a; c0 = R.c0
    # the integrand at the offset z from the end o (a or b), X formed from z without cancellation
    Ez = (z, o) -> (X = xat(R, z, o); -tj * (o + z) * (o + z) - sd * X + ncexp(V, X, sgn, s1, s2))
    Fz = (z, o) -> (X = xat(R, z, o); (S, E) = phired_scaled(V, X, sgn, s1, s2); W(o + z) * S * exp(Ez(z, o) + E))
    Eb = Ez(0.0, b); Ea = Ez(0.0, a)
    alpha = tj + c0 * sd
    # the descent directions, kept within 80 degrees of the segment, or 45 degrees at a pseudo-edge, whose branch points
    # v0 -+ i ell have their cuts running vertically away from the real axis; the first pieces on the scale of the nearest
    # singularity (ell/2 when it lies on the real axis, ell at a pseudo-edge, where it is at the perpendicular distance
    # >= 0.7 ell from the path)
    dega = 80.0; fa = ella / 2; ka == 0 && (dega = 45.0; fa = ella)
    degb = 80.0; fb = ellb / 2; kb == 0 && (degb = 45.0; fb = ellb)
    la, ua = descent(-2tj * a - sd * dXv(R, a) + ncexp_dv(V, xat(R, 0.0, a), sgn, s1, s2, dXv(R, a)), alpha, 1, L, dep, sg, dega)
    lb, ub = descent(-2tj * b - sd * dXv(R, b) + ncexp_dv(V, xat(R, 0.0, b), sgn, s1, s2, dXv(R, b)), alpha, -1, L, dep, sg, degb)
    pa_ = a + la * ua; pb_ = b + lb * ub
    # the three pieces: the integrand at the nodes (from the offsets along the two descent pieces, so that X is exact
    # near the ends) and the partition
    # the integrand is evaluated only where the wanted parts of the partition (chi near b, 1 - chi near a) exceed 1e-15:
    # the edge part skips the nodes near a, the smooth part those near b
    needa = ws || wl; needb = wr
    keep(vc) = (ch = chi(vc); (needa && abs(1 - ch) > 1e-15) || (needb && abs(ch) > 1e-15))
    fv = ComplexF64[]; v = ComplexF64[]
    s, w = pieces_nodes(s -> Ez(s * ua, a) - Ea, la, min(la / 8, fa), pa, 48 + 16 * (pa == 1))
    for k in eachindex(s); vc = a + s[k] * ua; keep(vc) && (push!(fv, Fz(s[k] * ua, a) * (w[k] * ua)); push!(v, vc)); end
    s, w = pieces_nodes(s -> Ez(s * ub, b) - Eb, lb, min(lb / 8, fb), pb, 48)
    for k in eachindex(s); vc = b + s[k] * ub; keep(vc) && (push!(fv, -Fz(s[k] * ub, b) * (w[k] * ub)); push!(v, vc)); end   # from b toward the segment: reversed
    # the connection: the straight line, unless the exponent rises along it above the ends (the Gaussian grows in the half
    # plane a descent piece has entered under the exponential factor, as with Re alpha ~ 0 and a steep -beta v); then the
    # polyline through the real axis, pa_ -> Re pa_ -> Re pb_ -> pb_, on which the exponent is that of the integrand itself
    if abs(pb_ - pa_) > 0
        Emax = max(real(Ea), real(Eb)) + 2
        legs = [pa_, pb_]
        if maximum(real(Ez(pa_ + k / 32 * (pb_ - pa_) - a, a)) for k in 0:32) > Emax
            legs = [pa_, complex(real(pa_)), complex(real(pb_)), pb_]
        end
        for k in 1:length(legs)-1
            dv = legs[k+1] - legs[k]
            abs(dv) == 0 && continue
            s, w = phase_nodes(s -> Ez(legs[k] + s * dv - a, a), Ea, Eb)
            for i in eachindex(s)
                vc = legs[k] + s[i] * dv
                keep(vc) || continue
                push!(fv, (abs(vc - b) < abs(vc - a) ? Fz(vc - b, b) : Fz(vc - a, a)) * (w[i] * dv)); push!(v, vc)
            end
        end
    end
    vs = vl = vr = zero(ComplexF64)
    if ws || wl
        val = zero(ComplexF64); for k in eachindex(v); val += fv[k] * (1 - chi(v[k])); end
        ws ? (vs = val) : (vl = val)
    end
    if wr
        val = zero(ComplexF64); for k in eachindex(v); val += fv[k] * chi(v[k]); end
        vr = val
    end
    vs, vl, vr
end

"""the length of a straight piece from the start along which Ef (real part, 0 at the start) has fallen below -EXPO:
from R0 (the quadratic estimate), grown by 1.5 while Ef is above -EXPO"""
function raylen(Ef, R0::Float64)
    Rr = R0
    for _ in 1:60
        real(Ef(Rr)) < -EXPO && break
        Rr *= 1.5
    end
    Rr
end
"""the ray of the noncentral kernel when the full exponent rises along the real axis by more than 2 above its value at the
edge (the mass of the noncentral density sits at X~ = m^2/2, beyond the breakpoint): on the rotated ray the integrand then
oscillates with a magnitude of up to e^{lam} and cancels, while the saddle of the full exponent E(rho) = -alpha rho^2 -
beta rho + ncexp(X(rho)) lies near the real axis.  The path is the straight piece from 0 to the saddle rho_s (Newton on E'
from the maximum of Re E on the real axis, E'' by a difference), then straight pieces in the direction of steepest descent
at their start, renewed where the descent along a piece ends, until E has fallen by EXPO; the nodes from pieces_nodes with
the full exponent.  Returns nothing when there is no such rise (the rotated ray serves)."""
function ray_saddle(Efull, dEfull, Ared, first::Float64, p::Int, Rmax::Float64)
    E0 = real(Efull(0.0 + 0im))
    rs = [Rmax * 1.25^(k - 80) for k in 0:80]
    Es = [real(Efull(complex(r))) for r in rs]
    kmax = argmax(Es)
    Es[kmax] - E0 > 2 || return nothing
    # the maximum on the real axis refined (Newton on Re E' along the axis, within the scan's cell), so that the saddle is
    # judged against the true height of the bump
    rr = rs[kmax]; Emax = Es[kmax]
    for _ in 1:20
        h = 1e-6 * rr; d1 = real(dEfull(complex(rr))); d2 = real(dEfull(complex(rr + h)) - dEfull(complex(rr - h))) / (2h)
        d2 < 0 || break
        rn = rr - d1 / d2; (rn > rs[kmax] / 1.25 && rn < rs[kmax] * 1.25) || break
        abs(rn - rr) < 1e-10 * rr && (rr = rn; break)
        rr = rn
    end
    Emax = max(Emax, real(Efull(complex(rr))))
    rho = complex(rr)
    for _ in 1:30                                                     # Newton on E' = 0
        d1 = dEfull(rho); h = 1e-6 * abs(rho)
        d2 = (dEfull(rho + h) - dEfull(rho - h)) / (2h)
        abs(d2) == 0 && break
        step = -d1 / d2
        abs(step) > abs(rho) && (step *= abs(rho) / abs(step))
        rho += step
        abs(step) < 1e-10 * abs(rho) && break
    end
    # the saddle serves only if it is the bump of the real axis slightly displaced: a saddle far above it (Re E_s beyond
    # the real-axis maximum) or far from it is one the real-axis path does not cross, and the rotated ray serves instead
    if !(isfinite(rho) && real(rho) > 0 && real(Efull(rho)) > E0 + 1 && real(Efull(rho)) <= Emax + 1 && abs(imag(rho)) <= 0.5abs(real(rho)))
        # a small rise: the rotated ray serves (its cancellation is at most e^{rise}); a large one: the real-axis maximum
        # is the pivot, and the descent pieces start from it
        Emax - E0 > 10 || return nothing
        rho = complex(rr)
    end
    val = zero(ComplexF64)
    # the piece from 0 to the saddle
    E_s = Efull(rho)
    s, w = pieces_nodes(lam -> Efull(lam * rho), 1.0, min(1 / 8, first / abs(rho)), p, 48)
    for k in eachindex(s); (S, E) = Ared(s[k] * rho); val += S * exp(Efull(s[k] * rho) + E) * w[k]; end
    val *= rho
    # the descent from the saddle: straight pieces, each in the direction of steepest descent at its start
    z0 = rho
    h = 1e-6 * abs(rho); d2 = (dEfull(rho + h) - dEfull(rho - h)) / (2h)
    d = abs(d2) > 0 ? exp(im * (pi - angle(d2)) / 2) : one(rho); real(d) < 0 && (d = -d)
    for piece in 1:16
        Ez0 = Efull(z0)
        Ef = r -> Efull(z0 + r * d) - Ez0
        # the length: where the exponent has fallen by EXPO, or where the descent along the line ends (Re Ef turns up)
        # the length: along the line while the exponent falls; stop where it has fallen by EXPO, or at the last point before
        # it rises (the direction is renewed there from the gradient), or at 10 Rmax
        L = 1e-3 * abs(rho); Lend = NaN; Eprev = 0.0
        for it in 0:200
            Ec = real(Ef(L))
            if Ec < -EXPO; Lend = L; break; end
            if isnan(Ec) || Ec > Eprev; Lend = it == 0 ? L : L / 1.4; break; end
            if L > 10Rmax; Lend = L; break; end
            Eprev = Ec; L *= 1.4
        end
        isnan(Lend) && (Lend = L)
        s, w = pieces_nodes(Ef, Lend, min(Lend / 8, first), 1, 48)
        for k in eachindex(s); (S, E) = Ared(z0 + s[k] * d); val += S * exp(Ef(s[k]) + Ez0 + E) * w[k] * d; end
        z0 += Lend * d
        real(Efull(z0) - E_s) < -EXPO && break
        g = dEfull(z0); abs(g) == 0 && break
        d = -conj(g) / abs(g)                                           # the new direction of steepest descent
    end
    val
end
"""int_0^inf A(rho) exp(-alpha rho^2 - beta rho) drho for the noncentral kernel K, A = W phialg along the ray from a, whose
growth e^{ncexp} the quadratic exponent does not see: the straight ray rho = r e in the direction e of the Gaussian's decay,
or, when the exponential grows along it by more than e^2 (the saddle term of the exponent, Re z < 0), the segment to the
saddle x0 = -beta/(2 alpha) and the descent line x0 + r e from it; the nodes from pieces_nodes with the full exponent, the
first piece of length `first` with the substitution p (4 at an edge)."""
function rayexp_nc(K::NCK, R::Breaks, a::Float64, sgn::Int, s1, s2, W, alpha::ComplexF64, beta::ComplexF64, first::Float64, p::Int)
    e = exp(-im * angle(alpha) / 2); aa = abs(alpha); be = beta * e
    Ared = rho -> ((S, E) = phired_scaled(K, xat(R, rho, a), sgn, s1, s2); (W(a + rho) * S, E))     # (S, E): the factor S e^E
    Efull = rho -> -alpha * rho * rho - beta * rho + ncexp(K, xat(R, rho, a), sgn, s1, s2)
    dEfull = rho -> -2alpha * rho - beta + ncexp_dv(K, xat(R, rho, a), sgn, s1, s2, dXv(R, a + rho))
    val = zero(ComplexF64)
    if max(0.0, -real(be))^2 / (4aa) <= 2
        Ef = r -> Efull(r * e)
        Rr = raylen(Ef, (-real(be) + sqrt(real(be)^2 + 4aa * EXPO)) / (2aa))
        vs = ray_saddle(Efull, dEfull, Ared, first, p, Rr)
        vs === nothing || return vs
        s, w = pieces_nodes(Ef, Rr, min(Rr / 8, first), p, 48)
        for k in eachindex(s); (S, E) = Ared(s[k] * e); val += S * exp(Ef(s[k]) + E) * w[k]; end
        return val * e
    end
    x0 = -beta / (2alpha)
    Ef1 = lam -> Efull(lam * x0)                                   # the segment to the saddle, rho = lam x0
    s, w = pieces_nodes(Ef1, 1.0, min(1 / 8, first / abs(x0)), p, 48)
    for k in eachindex(s); (S, E) = Ared(s[k] * x0); val += S * exp(Ef1(s[k]) + E) * w[k]; end
    val *= x0
    E0 = Efull(x0)
    Ef2 = r -> Efull(x0 + r * e) - E0                              # the descent line from the saddle
    Rr = raylen(Ef2, sqrt(EXPO / aa))
    s, w = pieces_nodes(Ef2, Rr, min(Rr / 8, first), 1, 48)
    v2 = zero(ComplexF64)
    for k in eachindex(s); (S, E) = Ared(x0 + s[k] * e); v2 += S * exp(Ef2(s[k]) + E0 + E) * w[k]; end
    val + v2 * e
end

"""
    inner_gen(V, t, R, s1, s2, M0, kind, part) -> (Is, Ie)

The transforms of the general kernel (see above), kind 1 (density) or 2 (cdf).  Is: the smooth part (Psi_t(1/2) for the
cdf, the bump of e^{-t v^2} at v = 0, and the ray from 0 when there is no breakpoint); Ie[k]: the part near the breakpoint
R.b[k], carrying exp(-t b_k^2), from the segment(s) adjacent to it and, for the last one, the ray beyond.  part: 0 all,
1 the smooth part only, 2 all edge parts, 10 + k the edge part k only.
"""
function inner_gen(V, tj::ComplexF64, R::Breaks, s1::ComplexF64, s2::ComplexF64, M0::ComplexF64, kind::Int, part::Int)
    kern = kernof(V)
    q = R.q; c0 = R.c0; ne = R.ne; b = R.b
    sg = imag(tj) >= 0 ? 1.0 : -1.0
    Is = zero(ComplexF64); Ie = zeros(ComplexF64, ne)
    W = kind == 1 ? (v -> 2v * v) : (v -> q + 2c0 * v)
    wants = part == 0 || part == 1
    want(k) = (part == 0 || part == 2 || part == 10 + k) && !R.negl[k]
    kind == 2 && wants && (Is = psi_half(V, s1, s2, M0))
    lX = kern == 1 ? 1 / (abs(s1) + 2) : 1 / (abs((s2 - s1) / 2) + 2)   # scale of the algebraic factor in X
    ed = [0.0; b; Inf]
    for j in 1:ne+1
        a = ed[j]; bb = ed[j+1]; sgn = R.sgn[j]
        kern == 1 && sgn < 0 && continue
        sd = (kern == 1 || sgn < 0) ? s1 : s2
        if isinf(bb)
            # the ray from a (a = 0: smooth; a = b_ne: edge ne)
            if j == 1; wants || continue; else; want(j - 1) || continue; end
            alpha = tj + c0 * sd; beta = 2alpha * a + sd * q
            cst = exp(-tj * a * a - sd * xat(R, 0.0, a))
            A = rho -> W(a + rho) * phialg(V, xat(R, rho, a), sgn, s1, s2)
            if V isa NCK
                lv = lX / max(abs(dXv(R, a)), 1e-300)
                c0 != 0 && (lv = min(lv, sqrt(lX / abs(c0))))
                lv = min(lv, R.ell0 / 16)
                first = lv; p = 1
                if j > 1 && R.kindb[j-1] == 0; first = R.ell[j-1]; elseif j > 1; lv = min(lv, R.ell[j-1] / 16); first = lv; p = 4; end
                val = exp(-tj * a * a - sd * xat(R, 0.0, a)) * rayexp_nc(V, R, a, sgn, s1, s2, W, tj + c0 * sd, 2(tj + c0 * sd) * a + sd * q, first, p)
                j == 1 ? (Is += val) : (Ie[j-1] += val)
                continue
            end
            lv = lX / max(abs(dXv(R, a)), 1e-300)
            c0 != 0 && (lv = min(lv, sqrt(lX / abs(c0))))
            lv = min(lv, R.ell0 / 16)                            # (rayint's first piece is 8 lv)
            p = 1
            if j > 1 && R.kindb[j-1] == 0
                # from a pseudo-edge: the straight ray in the direction e of the Gaussian's decay (within 45 degrees of
                # the real axis, so at least 0.7 ell from the branch points v0 -+ i ell, whose cuts the level-curve path
                # of gaussexp would cross), the oscillation of e^{-beta rho e} along it resolved by pieces_nodes
                e = exp(-im * angle(alpha) / 2); aa = abs(alpha); be = beta * e
                Rr = (-real(be) + sqrt(real(be)^2 + 4aa * EXPO)) / (2aa)       # fallen by EXPO
                s, w = pieces_nodes(r -> -aa * r * r - be * r, Rr, min(Rr / 8, R.ell[j-1]), 1, 48)
                val = zero(ComplexF64)
                for k in eachindex(s); val += A(s[k] * e) * exp(-aa * s[k] * s[k] - be * s[k]) * w[k]; end
                val *= cst * e
            else
                if j > 1; lv = min(lv, R.ell[j-1] / 16); p = 4; end
                val = cst * rayexp(A, alpha, beta, lv, p)
            end
            j == 1 ? (Is += val) : (Ie[j-1] += val)
        else
            # a finite segment [a, bb]: segint
            L = bb - a; h = L / 2; kap = KAPPA / h; m = a + h
            dep = L / 8
            if j == 1 && R.vsad > 0 && R.vsad < bb                # (R.vsad is NaN when the saddle is irrelevant)
                # the saddle of the exponent inside the first segment (c0 < 0, large q): the partition is placed midway
                # between the saddle and bb, and sharpened, so that the saddle's oscillation, e^{i psi} with
                # psi ~ q^2 Im t, which the smooth part's rule follows, leaks into the edge part by at most erf(-6), and
                # the edge into the smooth part likewise; the depth of the paths is reduced with it, keeping |chi| <= e^2.25
                m = max(m, (R.vsad + bb) / 2); kap = max(kap, KAPPA / (bb - m))
                kap = min(kap, 4KAPPA / L); m = min(m, bb - KAPPA / kap)     # at most twice as sharp
                dep = min(dep, KAPPA / 4 / kap)
            end
            chi = v -> (1 + erf(kap * (v - m))) / 2               # 1 near bb, 0 near a
            pa = j > 1 ? 4 : 1
            ws_ = j == 1 && wants; wl_ = j > 1 && want(j - 1); wr_ = want(j)
            (ws_ || wl_ || wr_) || continue
            ella = j > 1 ? R.ell[j-1] : R.ell0
            ka = j > 1 ? R.kindb[j-1] : 1                         # the kinds of the two ends (0: a pseudo-edge)
            vs_, vl_, vr_ = segint(V, tj, sd, sgn, s1, s2, W, R, a, bb, chi, pa, 1 + 3R.kindb[j], ella, R.ell[j], ws_, wl_, wr_, sg, ka, R.kindb[j], dep)
            ws_ && (Is += vs_)
            wl_ && (Ie[j-1] += vl_)
            wr_ && (Ie[j] += vr_)
        end
    end
    Is, Ie
end

"""the location Re v* of the saddle of the exponent -t v^2 - s_d X(v) on the first segment, v* = -s_d q/(2 (t + c0 s_d)),
at the reference point t = tau + i (it moves little with t while the saddle matters); NaN when there is no segment or
the saddle is irrelevant (it must lie inside the first segment and close to the real axis; with c0 > 0 it can have a
positive real part but sits far below the axis)."""
function saddle_loc(model::Int, c::Float64, q::Float64, c0::Float64, tau::Float64, stat::Bool, kern::Int, R::Breaks)
    R.ne == 0 && return NaN
    _, s1, s2, _ = tparts(model, c, [complex(tau), complex(tau, 1.0)]; stat)
    sd = (kern == 1 || R.sgn[1] < 0) ? s1[2] : s2[2]
    vs = -sd * q / (2 * (complex(tau, 1.0) + c0 * sd))
    (real(vs) > 0 && real(vs) < R.b[1] && abs(imag(vs)) < 0.2 * (R.b[1] - real(vs))) ? real(vs) : NaN
end

"""the phase of the saddle factor of the smooth part: the exponent -t v^2 - s_d X(v) = -alpha v^2 - beta v - s_d/2,
alpha = t + c0 s_d, beta = s_d q, on the first segment (s_d the root that governs the decay of phi_t there), has its
saddle at v* = -beta/(2 alpha), where it exceeds its value at v = 0 by beta^2/(4 alpha).  The saddle is present when it
lies ahead on the ray from 0 (no breakpoint: Re z < 0, z = beta/(2 sqrt(alpha))) or inside the first segment
(0 < Re v* < b_1), and is not negligible (Re beta^2/(4 alpha) > -30).  With c0 < 0 the saddle sits inside the segment at
v* = -q s_d'/(2 (1 + c0 s_d')) (s_d' = ds_d/dt) for large |q|, with the phase growing like q^2 Im t, which the trapezoidal
rule cannot follow."""
function saddle_phase_gen(kern::Int, q::Float64, c0::Float64, ts::Vector{ComplexF64}, s1::Vector{ComplexF64}, s2::Vector{ComplexF64}, R::Breaks)
    psi = fill(NaN, length(ts))
    for j in eachindex(ts)
        sd = (kern == 1 || R.sgn[1] < 0) ? s1[j] : s2[j]
        alpha = ts[j] + c0 * sd; beta = sd * q
        z2 = beta * beta / (4alpha)
        if R.ne == 0
            z = beta / (2sqrt(alpha))
            (real(z) < 0 && real(z2) > -30) && (psi[j] = imag(z2))
        elseif !isnan(R.vsad)
            vs = -beta / (2alpha)
            (real(vs) > 0 && real(vs) < R.b[1] && real(z2) > -30) && (psi[j] = imag(z2))
        end
    end
    psi
end

"""the outer integral for the general kernel: the smooth part as in `outer` (the trapezoidal rule in u, w = tau sinh u, or
Gauss-Legendre panels following the phase of the saddle factor), and one Filon part per breakpoint, with its frequency
y_k = b_k^2."""
function outer_gen(model::Int, c::Float64, q::Float64, c0::Float64, R::Breaks, tau::Float64, inner::Function, weight::Function;
                   hu::Float64 = 0.075, N::Int = 20, pfac::Int = 1, edge_hw = nothing, stat::Bool = false, parts::Bool = false)
    cde = tau
    kern = kernel(model, stat)
    umax = asinh(WMAX / cde)
    us = collect(0.0:hu:umax)
    ts = [tau + im * cde * sinh(u) for u in us]
    P, s1, s2, M0 = tparts(model, c, ts; stat)
    psi = saddle_phase_gen(kern, q, c0, ts, s1, s2, R)
    if maximum(phase_steps(psi)) > 0.25
        w0 = min(0.5, tau / 2)
        edges0 = [0.0; w0 * 2.0 .^ (0:ceil(Int, log2(WMAX / w0)))]
        edges0[end] = WMAX
        K = 8
        wk = Float64[]
        for p in 1:length(edges0)-1
            append!(wk, range(edges0[p], edges0[p+1]; length = K + 1)[1:end-1])
        end
        push!(wk, WMAX)
        te0 = [tau + im * w for w in wk]
        _, s1e, s2e, _ = tparts(model, c, te0; stat)
        steps = phase_steps(saddle_phase_gen(kern, q, c0, te0, s1e, s2e, R))
        wn = Float64[]; wt = Float64[]
        for p in 1:length(edges0)-1
            a, b = edges0[p], edges0[p+1]
            dp = sum(steps[(p-1)*K+k] for k in 1:K)
            m = max(1, ceil(Int, dp / 50))
            n = max(24, 16 * cld(ceil(Int, 3dp / m), 16)) * NODEFAC[]
            u, gw = rule(n)
            for k in 1:m
                aa = a + (b - a) * (k - 1) / m; bb = a + (b - a) * k / m
                append!(wn, aa .+ (bb - aa) .* u); append!(wt, (bb - aa) .* gw)
            end
        end
        ts = [tau + im * w for w in [0.0; wn]]
        P, s1, s2, M0 = tparts(model, c, ts; stat)
        gs = Vector{Float64}(undef, length(wn))
        tmap!(j -> real(P[j+1] * inner(ts[j+1], s1[j+1], s2[j+1], M0[j+1], 1)[1] * weight(ts[j+1])) * wt[j], gs, length(wn))
        val = sum(gs) / pi
    else
        gs = Vector{Float64}(undef, length(ts))
        tmap!(j -> real(P[j] * inner(ts[j], s1[j], s2[j], M0[j], 1)[1] * weight(ts[j])) * cde * cosh(us[j]), gs, length(ts))
        gs[1] *= 0.5
        val = sum(gs) * hu / pi
    end
    ev = 0.0
    if edge_hw !== nothing && any(.!R.negl)                      # the trapezoidal rule in w, for checking
        ws = collect(0.0:edge_hw:WMAX); te = [tau + im * w for w in ws]
        P, s1, s2, M0 = tparts(model, c, te; stat)
        ge = Vector{Float64}(undef, length(te))
        tmap!(j -> real(P[j] * sum(inner(te[j], s1[j], s2[j], M0[j], 2)[2]) * weight(te[j])), ge, length(te))
        ge[1] *= 0.5
        ev = sum(ge) * edge_hw / pi
        return parts ? (val, ev) : val + ev
    end
    u, _ = rule(N)
    for k in 1:R.ne
        R.negl[k] && continue
        y = R.y[k]
        w0 = min(0.5, 1 / y, tau / 2)
        edges0 = [0.0; w0 * 2.0 .^ (0:ceil(Int, log2(WMAX / w0)))]
        edges0[end] = WMAX
        edges = Float64[0.0]
        for p in 1:length(edges0)-1
            a, b = edges0[p], edges0[p+1]
            m = b <= FILONW[] / y ? ceil(Int, (b - a) * y / 25) : 1
            append!(edges, range(a, b; length = m + 1)[2:end])
        end
        if pfac > 1
            e2 = Float64[]
            for p in 1:length(edges)-1
                append!(e2, range(edges[p], edges[p+1]; length = pfac + 1)[1:end-1])
            end
            edges = [e2; edges[end]]
        end
        wn = Float64[]
        for p in 1:length(edges)-1
            append!(wn, edges[p] .+ (edges[p+1] - edges[p]) .* u)
        end
        te = [tau + im * w for w in [0.0; wn]]
        P, s1, s2, M0 = tparts(model, c, te; stat)
        G = Vector{ComplexF64}(undef, length(wn))
        tmap!(j -> (t = te[j+1]; P[j+1] * inner(t, s1[j+1], s2[j+1], M0[j+1], 10 + k)[2][k] * weight(t) * exp(t * y)), G, length(wn))
        acc = zero(ComplexF64)
        for p in 1:length(edges)-1
            acc += filon_panel(view(G, (p-1)*N+1:p*N), edges[p], edges[p+1], y, N)
        end
        ev += real(exp(-tau * y) * acc) / pi
    end
    parts ? (val, ev) : val + ev
end

"""The value of the density (kind 1) or the distribution function (kind 2) at q through the general kernel (c0 != 0, a fixed
start g0 != 0, or gen = true for checking at c0 = 0).  In case DK with c0 > 0 the density is infinite at
x = -sqrt(2 c0 (1 + g0^2)).  With a fixed start the kernel is the noncentral one (nckernel) at each point of the line."""
function value_gen(kind::Int, q::Float64, model::Int, c::Float64, c0::Float64, tau::Float64, stat::Bool; g0::Float64 = 0.0, hw::Float64 = 0.0, kw...)
    kern = kernel(model, stat); V = Val(kern)
    stat && g0 != 0 && error("a fixed start g0 applies to the zero-start cases only")
    x0 = (g0 != 0 && model == 1) ? (1 + g0 * g0) / 2 : 0.5
    R = breakpts(q, c0, tau; x0)
    R.dbl && kern == 1 && kind == 1 && return Inf
    R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, R)
    inner = g0 == 0 ? ((t, s1, s2, M0, part) -> inner_gen(V, t, R, s1, s2, M0, kind, part)) :
                      ((t, s1, s2, M0, part) -> (K = nckernel(model, c, t, g0, s1, s2); inner_gen(K, t, R, s1, s2, K.pref * M0, kind, part)))
    weight = kind == 1 ? (t -> one(t)) : (t -> 1 / t)
    if hw > 0                                                      # the trapezoidal rule in w for the whole integrand, for checking
        w = collect(0.0:hw:WMAX); t = tau .+ im .* w
        P, s1, s2, M0 = tparts(model, c, t; stat)
        g = Vector{Float64}(undef, length(t))
        tmap!(j -> begin; Is, Ie = inner(t[j], s1[j], s2[j], M0[j], 0); real(P[j] * (Is + sum(Ie)) * weight(t[j])); end, g, length(t))
        g[1] *= 0.5
        return sum(g) * hw / pi
    end
    outer_gen(model, c, q, c0, R, tau, inner, weight; stat, kw...)
end

# =====================================================================================
# The double integral (Theorem S1 of the Supplementary Material): the inner inversion in the argument of V along a path through a saddle point, the
# outer integral over V = b.  A port of Raymond Kan's TR_LIMDIST.
#
# With A' = U + cV... in the notation of the paper, R = U + 1/2 and a(b) = 1/2 + x sqrt(b) + c0 b,
#   F(x) = int_0^inf H(b) db,  H(b) = E[delta(V - b) 1{R <= a(b)}].
# (i) Case DK (zero or fixed start): H(b) = (1/2 pi i) int M_V(s) P e^{-s b} ds with P = erf(sqrt(a/(2 kap_1))) (a > 0),
#     the analytic continuation of P[kap_1 Z^2 <= a]; the density uses dP/da.  With a zero start the product is in
#     closed form (Girsanov): M_V P = e^{-c/2} (th/sinh th)^{1/2} sqrt(a) E(a eta), eta = th coth th - c,
#     E(t) = erf(sqrt t)/sqrt t entire, analytic for s < (c^2 + pi^2)/2; the inversion runs through the saddle point of
#     the whole integrand on the best of three paths (a parabola opening into a wedge, a vertical line in the th-plane,
#     and f_V minus the same integral with erfc in place of erf), the logarithm of the integrand formed free of
#     cancellation in th - c and th coth th - th.
# (ii) Cases DU, DT (zero start) and SK, SU, ST (stationary start): with g_s(r) the density of R under the tilt (the
#     K_0 form) and h(r,b) = (1/2 pi i) int M_V(s) g_s(r) e^{-s b} ds the joint density of (R, V),
#       f(x) = int sqrt(b) h(a(b), b) db,   F(x) = P[R <= 1/2] + int a'(b) G(a(b), b) db,
#       G(r,b) = (1/2 pi i) int M_V(s) g_s(r) e^{-s b}/s ds (Re s > 0),
#     a path crossing the real axis left of 0 adds the residue g_0(r).  Cases DU, DT with a zero start use the Girsanov
#     forms: M_V g_s(r) = e^{-c/2} (pi^2 (-A0) cosh th det Sig)^{-1/2} exp(-A1 r/(2 A0)) K0(-S|r|/(2 A0)),
#     S = (A1^2 - 4 A0 A2)^{1/2}, analytic through t_*.
# Cases DU, DT with a fixed start g0 != 0: R is a noncentral form, its density has no closed form, and form (i) is used
#     with the noncentral probability (a one-dimensional integral; a triple integral overall).
# With an initial observation the path is the parabola s* + mu th^2 - 2 i mu th through the saddle point s* of
#     log M_V(s) - s b, mu >= 4.5, the branch of log D continued along it and the sign conditions checked at every node.
# The outer integral over b is a cosine-mapped Gauss-Legendre rule in log b over [E V e^-8, E V e^6] with breakpoints
#     at the zeros of a(b), bisected where an error estimate from the Legendre coefficients of the integrand exceeds
#     1e-12 (adaptb), and graded toward the sharp step of the integrand in the far upper tail for c > c0.

const DBG = Ref(false)
"""the Faddeeva function w(z) = e^{-z^2} erfc(-i z)"""
faddw(z) = erfcx(-im * z)
"""D(s, t) e^{-mu}, mu = sqrt(c^2 - 2t) (principal), and mu: the determinant with its exponential growth taken out, from
the closed forms with cosh mu, sinh mu/mu and (1 - cosh mu)/z scaled by e^{-mu}, so that nothing overflows at the large
tilts of the saddle-point searches; the series of Dfun below |z| = Z0."""
function Dscaled(model::Int, c::Float64, s::ComplexF64, t::ComplexF64)
    z = c * c - 2t; mu = sqrt(z)
    abs(z) < Z0 && return Dfun(model, c, s, t) * exp(-mu), mu
    e2 = exp(-2mu); Ct = (1 + e2) / 2; St = (1 - e2) / (2mu); Qt = (exp(-mu) - Ct) / z
    lam = -s; lx = t; e = exp(c)
    if model == 1
        return Ct - (c + s) * St, mu
    elseif model == 2
        A = 2 * c * c * lam - 4 * c * lx - 2 * lam * lam
        B = -c^3 + c * c * lam - 2 * lx - lam * lam
        return e * (c * c * Ct + A * Qt + B * St) / z, mu
    else
        c2, c3, c4, c5, c6, c7 = c^2, c^3, c^4, c^5, c^6, c^7
        P1 = c6 - 12 * c4 * lam + 2 * lx * (-c4 + 12 * c3 - 12 * c2) + 48 * lam * lam * (c2 - 3 * c + 3)
        P2 = (4 * lam * (-c6 + 6 * c4) + 16 * lam * lam * (c4 - 3 * c3 - 3 * c2 + 18 * c - 18)
              + 8 * lx * (c5 - 3 * c4 + c4 * lam - 4 * lam * lam * (c2 - 3 * c + 3)) + 16 * lx * lx * (-c3 + 3 * c2 - 6 * c + 6))
        P3 = (-c7 + lam * (c6 + 24 * c4) + 4 * lam * lam * (-c4 + 3 * c3 - 27 * c2 + 72 * c - 72)
              + 2 * lx * (c5 - 4 * c4 - 12 * c3 + 12 * c2 - c4 * lam + 4 * lam * lam * (c2 - 3 * c + 3)) + 16 * lx * lx * (c2 - 3 * c + 3))
        return e * (Ct * P1 + Qt * P2 + St * P3) / z^3, mu
    end
end
"""The initial-value term Q(u, v) of cases DU, DT in the determinant form, Q = c^{2k} e^c (v + u^2/2 + c u) g(z)/D(u, v),
with g and D scaled by e^{-mu} (valid on the whole domain of the moment generating function, unlike the Girsanov form
of qstart, which is singular where cosh mu = 0)."""
function qstart_d(model::Int, c::Float64, u::ComplexF64, v::ComplexF64)
    Dt, _ = Dscaled(model, c, u, v)
    (model == 2 ? c^2 : c^4) * exp(c) * (v + u * u / 2 + c * u) * gstat_scaled(model, c * c - 2v) / Dt
end
"""e^z K_0(z) for any complex z (k0x where Re z >= 0, the library routine otherwise, the leading asymptotic term beyond
its range)"""
k0scaled(z::ComplexF64) = real(z) >= 0 ? k0x(z) : abs(z) > 1e7 ? sqrt(pi / (2z)) : besselkx(0, z)

# ---------- the adaptive outer rule in log b ----------
"""the rules of the outer integral for M nodes: the Gauss-Legendre rule on [0, 1] (u, gw), the cosine-mapped rule
(z1, w1) and the twice-mapped rule (z2, w2), and the matrix LM of the Legendre coefficients of the values at the nodes."""
function outer_setup(M::Int)
    u, gw = rule(M)
    z1 = (1 .- cos.(pi .* u)) ./ 2; w1 = gw .* pi ./ 2 .* sin.(pi .* u)
    z2 = (1 .- cos.(pi .* z1)) ./ 2; w2 = gw .* (pi / 2)^2 .* sin.(pi .* z1) .* sin.(pi .* u)
    xg = 2 .* u .- 1
    Pn = zeros(M, M); Pn[:, 1] .= 1; M > 1 && (Pn[:, 2] .= xg)
    for n in 1:M-2; Pn[:, n+2] .= ((2n + 1) .* xg .* Pn[:, n+1] .- n .* Pn[:, n]) ./ (n + 1); end
    LM = ((2 .* (0:M-1) .+ 1) ./ 2) .* (Pn' .* (2 .* gw)')
    (u = u, gw = gw, z1 = z1, w1 = w1, z2 = z2, w2 = w2, LM = LM)
end
const OUTERRULES = Dict{Int,Any}()
outer_rules(M) = lock(RULELOCK) do; get!(() -> outer_setup(M), OUTERRULES, M); end

"""
    adaptb(PA, PL, PEL, PER, PX, nx, zs, ws, ze, we, LM, gw, evalH, otol, maxlev) -> F, err, nun, npairs, nref

Adaptive outer rule in log b.  Each piece [PA, PA + PL] of point PX carries the M-point rule (zs, ws) on [0, 1], or
(ze, we) when it ends at a zero of a(b) (PEL, PER).  The Legendre coefficients C_n of the integrand in the rule's
variable estimate the error: the largest of the last two, extrapolated to n = 3M/2 by their decay from the middle
(est = C_hi^3/C_mid^2 when C_hi < C_mid, otherwise C_hi).  Pieces with est > max(otol, 10 times the noise floor of
the inner error estimates) are bisected, up to maxlev times; the estimates of the accepted pieces, plus the rule
applied to the error estimates of the inner integrals, make up err.  evalH(B, IX) returns the outer integrand and
its error at the pairs (b, x[IX]).
"""
function adaptb(PA::Vector{Float64}, PL::Vector{Float64}, PEL::Vector{Bool}, PER::Vector{Bool}, PX::Vector{Int}, nx::Int,
                zs, ws, ze, we, LM, gw, evalH, otol::Float64, maxlev::Int)
    M = length(zs); mh = M ÷ 2
    F = zeros(nx); err = zeros(nx); nref = zeros(Int, nx); npairs = 0
    Ha = Float64[]; Ea = Float64[]; Wa = Float64[]; Xa = Int[]
    for lev in 0:maxlev
        np = length(PA); np == 0 && break
        B = Matrix{Float64}(undef, M, np); W = similar(B); IXm = Matrix{Int}(undef, M, np)
        for p in 1:np
            z, w = (PEL[p] || PER[p]) ? (ze, we) : (zs, ws)
            for k in 1:M
                B[k, p] = exp(PA[p] + PL[p] * z[k]); W[k, p] = PL[p] * w[k] * B[k, p]; IXm[k, p] = PX[p]
            end
        end
        Hv, Ev = evalH(vec(B), vec(IXm))
        H = reshape(Hv, M, np); E = reshape(Ev, M, np)
        npairs += M * np
        G = W .* H; G[W .== 0] .= 0
        I = vec(sum(G, dims = 1))
        C = abs.(LM * (G ./ gw))
        est = zeros(np)
        if M >= 4
            for p in 1:np
                chi = max(C[M, p], C[M-1, p]); cmid = max(C[mh+1, p], C[mh, p])
                est[p] = chi < cmid ? chi^3 / cmid^2 : chi
            end
        end
        est[.!isfinite.(est)] .= 0
        WE = abs.(W) .* E; WE[W .== 0] .= 0
        nse = vec(sum(WE, dims = 1)) .+ 1e3 * eps() .* vec(sum(abs.(G), dims = 1))
        rf = (est .> max.(otol, 10 .* nse)) .& (lev < maxlev)
        for p in findall(.!rf)
            F[PX[p]] += I[p]; err[PX[p]] += est[p] + sum(view(WE, :, p))
            append!(Ha, view(H, :, p)); append!(Ea, view(E, :, p)); append!(Wa, view(W, :, p)); append!(Xa, view(IXm, :, p))
        end
        any(rf) || break
        ir = findall(rf)
        for p in ir; nref[PX[p]] += 1; end
        h = PL[ir] ./ 2
        PA = [PA[ir]; PA[ir] .+ h]; PL = [h; h]; PX = [PX[ir]; PX[ir]]
        PEL = [PEL[ir]; falses(length(ir))]; PER = [falses(length(ir)); PER[ir]]
    end
    nun = zeros(Int, nx)
    if !isempty(Xa)
        sc = zeros(nx)
        for k in eachindex(Xa); sc[Xa[k]] = max(sc[Xa[k]], abs(Ha[k])); end
        for k in eachindex(Xa); (Ea[k] > 1e-9 * sc[Xa[k]] && abs(Wa[k]) * Ea[k] > 1e-14) && (nun[Xa[k]] += 1); end
    end
    F, err, nun, npairs, nref
end

"""the trapezoidal sum of the values G along a path (both halves folded into the real part), with the nested rule of
twice the step as error estimate, plus rounding and the truncation at the last node."""
function fin(G::Vector{ComplexF64})
    any(!isfinite, G) && return 0.0, Inf
    I = real(sum(G)); Ih = real(sum(2 * G[k] for k in 1:2:length(G)))
    I, abs(I - Ih) + 1e-14 * sum(abs, G) + 1e3 * abs(G[end])
end

# ---------- the saddle point of a real convex function of s (one pair at a time) ----------
"""width: the step d = sc 10^k at which (f(s+d) + f(s-d))/2 - f(s) is closest to 1/2; returns (a1, f(s))."""
function wscan1(Lf, s::Float64, sc::Float64, smx::Float64)
    L0 = Lf(s); best = Inf; dd = sc; Db = NaN
    for k in -6:12
        d = min(sc * 10.0^k, max((smx - s) / 2, eps() * abs(s)))
        Dl = (Lf(s - d) + Lf(s + d)) / 2 - L0
        m = abs(log(max(Dl, floatmin()) / 0.5))
        if m < best; best = m; dd = d; Db = Dl; end
    end
    g2 = 2Db / dd^2
    (!(g2 > 0) || !isfinite(g2)) && (g2 = 1 / sc^2)
    1 / sqrt(g2), L0
end
"""minimum of the real convex function Lf(s) (NaN or -Inf outside its domain), s < smx: bracketing, then the Illinois
variant of regula falsi on the derivative, then the curvature at the minimum; returns (s, a1, g2, L0) with a1 the
width 1/sqrt(f'')."""
function rsad1(Lf, b::Float64, smx::Float64)
    sc = 1 / b
    s0 = min(-sc, smx - sc / 2); t0 = -(sc / 2)^2 / 2 - abs(smx); (b < 0.05 && t0 < s0) && (s0 = t0)
    w0, _ = wscan1(Lf, s0, sc, smx)
    hd(s, w) = max(1e-4w, 1e-12 * abs(s))
    g(s, h) = (Lp = Lf(s + h); Lm = Lf(s - h); isfinite(Lp) ? (Lp - Lm) / (2h) : Inf)
    sL = s0; gL = g(sL, hd(sL, w0))
    sR = smx - 1e-12 * max(1.0, abs(smx))
    for k in 1:200                                                   # move left until f' < 0
        gL < 0 && break
        sR = min(sR, sL); sL -= max(sc, w0) * 4.0^k; gL = g(sL, hd(sL, w0))
    end
    gR = g(sR, hd(sR, w0)); (!isfinite(gR) || gR <= 0) && (gR = Inf)
    side = 0
    for _ in 1:300
        w = sR - sL; w > 1e-9 * w0 || break
        sn = isfinite(gR) ? sL - gL * w / (gR - gL) : (sL + sR) / 2
        (sn > sL && sn < sR) || (sn = (sL + sR) / 2)
        gn = g(sn, hd(sn, w0))
        if gn < 0
            sL = sn; gL = gn; side == -1 && (gR /= 2); side = -1
        else
            sR = sn; gR = gn; side == 1 && (gL /= 2); side = 1
        end
        isfinite(gR) || (gR = Inf)
    end
    s = (sL + sR) / 2
    a1, L0 = wscan1(Lf, s, sc, smx)
    s, a1, 1 / a1^2, L0
end

# ---------- case DK, zero start: the closed form of E[e^{sV}; R <= a] ----------
"""the saddle point of Lf(s) - s b... minimum of the convex Lf, warm-started at (s0, a10) from a neighbouring pair: Newton on
the derivative with central differences at the scale of the width, damped to four widths per step, stopped when the
step is below a thousandth of the width (the path only needs an approximate saddle: any real point of the domain gives
the same integral); falls back to rsad1 when the curvature is not positive or the domain end is hit."""
function rsad_warm(Lf, b::Float64, smx::Float64, s0::Float64, a10::Float64)
    s = s0; a1 = a10
    isfinite(s) && isfinite(a1) && a1 > 0 || return rsad1(Lf, b, smx)
    s = min(s, smx - a1 / 4)
    for _ in 1:30
        h = max(1e-4 * a1, 1e-12 * abs(s)); h = min(h, (smx - s) / 2)
        L0 = Lf(s); Lp = Lf(s + h); Lm = Lf(s - h)
        (isfinite(L0) && isfinite(Lp) && isfinite(Lm)) || return rsad1(Lf, b, smx)
        g1 = (Lp - Lm) / (2h); g2 = (Lp - 2L0 + Lm) / (h * h)
        g2 > 0 || return rsad1(Lf, b, smx)
        a1 = 1 / sqrt(g2)
        ds = clamp(-g1 / g2, -4a1, 4a1)
        sn = s + ds
        sn < smx - 1e-12 * max(1.0, abs(smx)) || (sn = (s + smx) / 2)
        if abs(ds) < 1e-3 * a1
            return sn, a1, g2, Lf(sn)
        end
        s = sn
    end
    rsad1(Lf, b, smx)
end

"""eta = th coth th - c at real s (cancellation-free)"""
function dketa(s::Float64, c::Float64)
    th = sqrt(complex(c * c - 2s))
    del = c > 0 ? -2s / (th + c) : th - c
    abs(th) < 1e-4 && return real(del + 1 - th + th * th / 3)
    e2 = exp(-2th)
    real(del + 2th * e2 / (1 - e2))
end
"""end of the real domain of M_V (case DK): eta(s) = 0, eta decreasing in s"""
function dksmax(c::Float64)
    c == 1 && return 0.5
    lo = min(0.0, c * c / 2) - 1; hi = (c * c + pi^2) / 2
    while dketa(lo, c) <= 0; lo = 2lo - 1; end
    for _ in 1:2000
        m = (lo + hi) / 2; (m == lo || m == hi) && break
        dketa(m, c) > 0 ? (lo = m) : (hi = m)
    end
    lo
end
"""log of the integrand times e^{-s b} along a path S (continuity of sqrt(eta) along it), case DK, cancellation-free in
del = th - c and ep = th coth th - th: kind 0: e^{-c/2} (th/sinh th)^{1/2} sqrt(a) E(a eta) (cdf); kind 1: M_V;
kind 2: M_V erfc(sqrt(a eta)); kind 3: e^{-c/2} (th/sinh th)^{1/2} e^{-a eta}/sqrt(pi a) (pdf); with
-s b = del (th + c) b/2 and -a eta - s b = -v eta - c b ep + del^2 b/2, v = a - c b."""
function dklam(S::Vector{ComplexF64}, b::Float64, c::Float64, a::Float64, v::Float64, kind::Int)
    n = length(S); L = Vector{ComplexF64}(undef, n); rprev = zero(ComplexF64)
    for k in 1:n
        s = S[k]; q = c * c - 2s; th = sqrt(q)
        del = c > 0 ? -2s / (th + c) : th - c
        if abs(th) < 1
            r = one(q); tm = one(q); ch = one(q)
            for j in 1:14; tm *= q / ((2j) * (2j + 1)); r += tm; end
            tm = one(q); for j in 1:14; tm *= q / ((2j - 1) * (2j)); ch += tm; end
            ls = -log(r) / 2; ep = ch / r - th
        else
            e2 = exp(-2th)
            ep = 2th * e2 / (1 - e2)
            ls = (log(th) - th - log((1 - e2) / 2)) / 2
        end
        eta = del + ep
        if kind == 3
            L[k] = -c / 2 + ls - log(pi * a) / 2 - v * eta - c * b * ep + del * del * b / 2
        elseif kind > 0
            r = sqrt(eta)
            k > 1 && real(r * conj(rprev)) < 0 && (r = -r)
            rprev = r
            if kind == 1
                L[k] = -c / 2 + ls + del * (th + c) * b / 2 - log(r)
            else
                L[k] = -c / 2 + ls - log(r) - v * eta - c * b * ep + del * del * b / 2 + log(faddw(im * sqrt(a) * r))
            end
        else
            t = a * eta; zt = sqrt(t)
            if real(t) > 2
                L[k] = -c / 2 + ls + del * (th + c) * b / 2 + log(1 - exp(-t) * faddw(im * zt)) - log(zt) + log(a) / 2
            else
                if abs(zt) < 0.2
                    ser = zero(t); term = one(t)
                    for j in 0:20; ser += term / (2j + 1); term = -term * t / (j + 1); end
                    lg = t + log(2 / sqrt(pi) * ser)
                else
                    lg = log((exp(t) - faddw(im * zt)) / zt)
                end
                L[k] = -c / 2 + ls + log(a) / 2 - v * eta - c * b * ep + del * del * b / 2 + lg
            end
        end
    end
    L
end

"""the parabola s^ + mu t^2 - 2 i mu t near the saddle, bending into the wedge Re s - s^ = kap |Im s|; trapezoidal rule
in u, t = sinh(u); lk maps the path to the log of the integrand."""
function pathW(lk, b::Float64, sq::Float64, mu::Float64, NN::Int; kap::Float64 = 2.0, tl::Float64 = 45.0, divS::Bool = false)
    tmax = max(10.0, (tl + 5) / (2kap * b * mu))
    ht = asinh(tmax) / NN
    S = Vector{ComplexF64}(undef, NN + 1); dSW = similar(S)
    for j in 0:NN
        tau = j * ht; TH = sinh(tau); Wt = cosh(tau) * ht; j == 0 && (Wt /= 2)
        rq = sqrt(1 + (TH / (2kap))^2)
        S[j+1] = sq + mu * (TH^2 / rq) - 2im * mu * TH
        dSW[j+1] = (mu * (2TH / rq - TH^3 / (4kap^2 * rq^3)) - 2im * mu) * (im / pi) * Wt
    end
    G = exp.(lk(S)) .* dSW
    divS && (G ./= S)
    fin(G)
end
"""the vertical line th = thA + i y in the th-plane (Gaussian decay e^{-y^2 b/2}), thA >= th^ and large enough that the
oscillation of a coth(th) cannot build up against the Gaussian."""
function pathT(lk, b::Float64, a::Float64, sq::Float64, a1q::Float64, NN::Int, c::Float64; tl::Float64 = 45.0, divS::Bool = false, polegap::Bool = false)
    qh = c * c - 2sq; th0 = qh > 0 ? sqrt(qh) : 0.0
    tA = max(th0, asinh(abs(a) / sqrt(6b)) / 2, 1.0)
    if polegap                                                       # the crossing point at least a1 from the pole at 0
        sA = (c * c - tA^2) / 2
        abs(sA) < a1q && (tA = sqrt(max(c * c + 2a1q, 1.0)))
    end
    sA = (c * c - tA^2) / 2
    hq = 1e-4 * a1q
    L3 = real.(lk(ComplexF64[sA - hq, sA, sA + hq]))
    Ls = (L3[3] - L3[1]) / (2hq); Lss = (L3[3] - 2L3[2] + L3[1]) / hq^2
    sg = 1 / sqrt(max(Lss * tA^2 - Ls, b)); isfinite(sg) || (sg = 1 / sqrt(b))
    ym = max(12sg, sqrt(2 * (tl + 10) / b))
    um = asinh(ym / sg)
    S = Vector{ComplexF64}(undef, NN + 1); TW = similar(S)
    for j in 0:NN
        U = j * um / NN; Y = sg * sinh(U); Wt = sg * cosh(U) * um / NN; j == 0 && (Wt /= 2)
        T = tA + im * Y; S[j+1] = (c * c - T * T) / 2; TW[j+1] = T * Wt / pi
    end
    G = exp.(lk(S)) .* TW
    divS && (G ./= S)
    I, E = fin(G)
    I, E, sA
end

"""
    agreepaths(evalpath, np, NN, tol) -> I, E

An inner inversion by `np` paths (`evalpath(p, NN)` returns the trapezoidal value and the nested-rule estimate of path
p with NN nodes).  Each path is first refined on its own, the node count doubled up to three times until its estimate is
below tol max(1, |I|); a value is accepted as soon as two paths agree to within that tolerance, the one with the smaller
estimate being kept and the discrepancy entering the error.  The nested rule alone can be fooled: an oscillating tail
that is not resolved by the rule and its half aliases into the same wrong value on both (case DK, c = 5, c0 = -5,
v = 0.086, a = 35: the parabola with 96 and 192 nodes agrees with itself to 1e-15 and is off by 1e-7, which the
vertical line exposes).  When no two paths agree, the value with the smallest estimate is kept if every other path is
consistent with it, the discrepancy within ten times the sum of the two estimates (a path that converges slowly, or not
at all, is no evidence against a resolved one); otherwise the inconsistent paths and the best one are recomputed with the node
count doubled, up to three more times, until two agree or all are consistent, and if they still disagree the best
value is returned with the smallest inconsistent discrepancy as its error, which the outer rule then reports."""
function agreepaths(evalpath, np::Int, NN::Int, tol::Float64)
    I = fill(NaN, np); E = fill(Inf, np); N = fill(NN, np)
    scale(p, q) = max(1.0, abs(I[p]), abs(I[q]))
    agree(p, q) = isfinite(I[p]) && isfinite(I[q]) && abs(I[p] - I[q]) <= 2tol * scale(p, q)
    accept(p, q) = (b = E[p] <= E[q] ? p : q; (I[b], max(E[b], abs(I[p] - I[q]))))
    inconsistent(b, q) = q != b && isfinite(I[q]) && abs(I[b] - I[q]) > 10(E[b] + E[q])
    for p in 1:np
        NNq = NN
        for _ in 0:3
            Ip, Ep = evalpath(p, NNq)
            if Ep < E[p]; I[p] = Ip; E[p] = Ep; N[p] = NNq; end
            Ep <= tol * max(1.0, abs(Ip)) && break
            NNq *= 2
        end
        for q in 1:p-1; agree(p, q) && return accept(p, q); end
    end
    b = argmin(E)
    isfinite(E[b]) || return 0.0, Inf
    bad = [q for q in 1:np if inconsistent(b, q)]
    isempty(bad) && return I[b], E[b]
    for _ in 1:3
        for p in [b; bad]
            N[p] *= 2
            I[p], E[p] = evalpath(p, N[p])                            # the finer rule replaces the value, whatever its estimate
        end
        for p in 1:np, q in 1:p-1; agree(p, q) && return accept(p, q); end
        b = argmin(E)
        isfinite(E[b]) || return 0.0, Inf
        bad = [q for q in 1:np if inconsistent(b, q)]
        isempty(bad) && return I[b], E[b]
    end
    I[b], max(E[b], minimum(abs(I[b] - I[q]) for q in bad))
end

"""the saddle data of the pair (b, a) in case DK: (s, a1, g2, L0) of the whole integrand and, for the cdf, of M_V and of
M_V erfc (the third route); warm-started from `prev` (the data of a neighbouring b) when given"""
function dksaddles(b::Float64, a::Float64, c::Float64, pdfm::Bool, prev)
    v = a - c * b; k0 = pdfm ? 3 : 0
    s1 = (c * c + pi^2) / 2; smx = dksmax(c)
    lk(S, k) = dklam(S, b, c, a, v, k)
    f0 = S -> real(lk([complex(S)], k0)[1])
    sad = prev === nothing ? rsad1(f0, b, s1) : rsad_warm(f0, b, s1, prev[1][1], prev[1][2])
    pdfm && return (sad, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    fB = S -> real(lk([complex(S)], 1)[1]); fC = S -> real(lk([complex(S)], 2)[1])
    sB = prev === nothing ? rsad1(fB, b, smx) : rsad_warm(fB, b, smx, prev[2][1], prev[2][2])
    sC = prev === nothing ? rsad1(fC, b, smx) : rsad_warm(fC, b, smx, prev[3][1], prev[3][2])
    (sad, sB, sC)
end
"""H = E[delta(V - b) 1{R <= a}] (pdf: dH/da) for the pair (b, a), case DK with a zero start: three paths, each with a
nested-rule error estimate, refined by doubling; the best is kept."""
function dkpair(b::Float64, a::Float64, c::Float64, pdfm::Bool, NN::Int; sad = nothing)
    a > 0 || return 0.0, 0.0
    v = a - c * b; tol = 1e-13
    k0 = pdfm ? 3 : 0
    lk(S, k) = dklam(S, b, c, a, v, k)
    sad === nothing && (sad = dksaddles(b, a, c, pdfm, nothing))
    s, a1, g2, L0 = sad[1]
    Lh = L0 - log(2pi * max(g2, 1e-6 * b * b)) / 2                    # log of the saddle-point approximation
    lw = pdfm ? 1.5log(b) : log(b)
    Lh + lw < -35 && return 0.0, 0.0                                  # negligible (in d log b)
    sB, aB = sad[2][1], sad[2][2]; sC, aC = sad[3][1], sad[3][2]
    function evalpath(pth::Int, NNq::Int)
        if pth == 1
            I, E = pathW(S -> lk(S, k0), b, s, a1 / 2, NNq)
        elseif pth == 2
            I, E, _ = pathT(S -> lk(S, k0), b, a, s, a1, NNq, c)
        else
            IB, EB = pathW(S -> lk(S, 1), b, sB, aB / 2, NNq); IC, EC, _ = pathT(S -> lk(S, 2), b, a, sC, aC, NNq, c)
            I = IB - IC; E = EB + EC
        end
        I, E
    end
    agreepaths(evalpath, pdfm ? 2 : 3, NN, tol)
end

# ---------- cases DU, DT, zero start: the closed form of E[e^{sV} delta(R - r)] ----------
"""The Girsanov pieces (th, del, lch, dS, D, A2, A1, A0) of models 2 and 3 without the matrices of `girsanov` (no
allocation: the 2x2 and 3x3 determinants and adjugates written out), for the inner loops of the double integral."""
@inline function girs_fast(model::Int, c::Float64, t::ComplexF64)
    q = c * c - 2t; th = sqrt(q)
    del = c > 0 ? -2t / (th + c) : th - c
    e2 = exp(-2th); em = exp(-th)
    if abs(th) < 1
        g1 = polyq(GT, 1, q); g2 = -polyq(GE, 2, q); g3 = -polyq(GT, 2, q); g4 = polyq(GE, 3, q); g5 = polyq(GT, 3, q)
        lch = log(cosh(th)); N11 = 1 - c * g1
    else
        tn = (1 - e2) / (1 + e2); sc = 2em / (1 + e2)
        g1 = tn / th; g2 = (1 - sc) / th^2; g3 = (th - tn) / th^3; g4 = (th^2 - 2 + 2sc) / (2th^4); g5 = (th^3 / 3 - th + tn) / th^5
        t1 = 2 * e2 / (1 + e2); lch = th + log((1 + e2) / 2)
        N11 = (del + c * t1) / th
    end
    if model == 2
        dS = g1 * g3 - g2 * g2
        D = N11 * (1 + 2t * g3) + 2c * t * g2 * g2
        return (th = th, del = del, lch = lch, dS = dS, D = D, A2 = D / dS, A1 = (g1 - 2g2) / dS + 2t, A0 = -one(q))
    end
    # Sg = [g1 g2 g3; g2 g3 g4; g3 g4 g5], Qd = [c 0 0; 0 -8t 12t; 0 12t -24t], N = I - Qd Sg with N11 as above
    S11, S12, S13, S22, S23, S33 = g1, g2, g3, g3, g4, g5
    N12 = -c * S12; N13 = -c * S13
    N21 = -(-8t * S12 + 12t * S13); N22 = 1 - (-8t * S22 + 12t * S23); N23 = -(-8t * S23 + 12t * S33)
    N31 = -(12t * S12 - 24t * S13); N32 = -(12t * S22 - 24t * S23); N33 = 1 - (12t * S23 - 24t * S33)
    D = N11 * (N22 * N33 - N23 * N32) - N12 * (N21 * N33 - N23 * N31) + N13 * (N21 * N32 - N22 * N31)
    dS = S11 * (S22 * S33 - S23 * S23) - S12 * (S12 * S33 - S23 * S13) + S13 * (S12 * S23 - S22 * S13)
    # adj(Sg)
    Si11 = S22 * S33 - S23 * S23; Si12 = -(S12 * S33 - S13 * S23); Si13 = S12 * S23 - S13 * S22
    Si21 = -(S12 * S33 - S23 * S13); Si22 = S11 * S33 - S13 * S13; Si23 = -(S11 * S23 - S13 * S12)
    Si31 = S12 * S23 - S22 * S13; Si32 = -(S11 * S23 - S12 * S13); Si33 = S11 * S22 - S12 * S12
    # P' = Sg^{-1} - Qd, with P'11 from row 1 of N (cancellation-free)
    P11 = (N11 * Si11 + N12 * Si21 + N13 * Si31) / dS
    P12 = Si12 / dS; P13 = Si13 / dS; P21 = Si21 / dS; P22 = Si22 / dS + 8t; P23 = Si23 / dS - 12t
    P31 = Si31 / dS; P32 = Si32 / dS - 12t; P33 = Si33 / dS + 24t
    # adj(P')
    aP11 = P22 * P33 - P23 * P32; aP12 = -(P12 * P33 - P13 * P32); aP13 = P12 * P23 - P13 * P22
    aP21 = -(P21 * P33 - P23 * P31); aP22 = P11 * P33 - P13 * P31; aP23 = -(P11 * P23 - P13 * P21)
    aP31 = P21 * P32 - P22 * P31; aP32 = -(P11 * P32 - P12 * P31); aP33 = P11 * P22 - P12 * P21
    # A1 = tr(K adj P'), A0 = tr(adj(K) P') with K = [1 2 -6; 2 -12 12; -6 12 0], adj K = [-144 -72 -48; -72 -36 -24; -48 -24 -16]
    A1 = (1aP11 + 2aP21 - 6aP31) + (2aP12 - 12aP22 + 12aP32) + (-6aP13 + 12aP23 + 0aP33)
    A0 = (-144P11 - 72P21 - 48P31) + (-72P12 - 36P22 - 24P32) + (-48P13 - 24P23 - 16P33)
    (th = th, del = del, lch = lch, dS = dS, D = D, A2 = D / dS, A1 = A1, A0 = A0)
end
const IICOUNT = Ref(0)
"""log J(s;r) - s b along a path S, continuous along it, J(s;r) = E[e^{sV} delta(R - r)] = M_V(s) g_s(r):
log J = -c/2 - lcd/2 - log pi - log(-A0)/2 - 2 A2 r/(A1+S) + log K0s(S r/(-2 A0)), r > 0 (r < 0: (A1+S)|r|/(2 A0)
in place of the exponential term), S = sqrt(A1^2 - 4 A0 A2), lcd = log(cosh th det Sig); -s b = del (th + c) b/2."""
function iilog(S::Vector{ComplexF64}, b::Float64, c::Float64, model::Int, r::Float64)
    n = length(S); L = Vector{ComplexF64}(undef, n)
    lcd = Vector{Float64}(undef, n); la0 = similar(lcd)
    IICOUNT[] += n
    for k in 1:n
        G = girs_fast(model, c, S[k])
        lcdk = G.lch + log(G.dS); la0k = log(-G.A0)
        lcd[k] = imag(lcdk); la0[k] = imag(la0k)
        Sq = sqrt(G.A1 * G.A1 - 4 * G.A0 * G.A2)                     # principal: Re S >= 0 (g decays in |r|)
        z = Sq * abs(r) / (-2 * G.A0)
        ex = r < 0 ? (G.A1 + Sq) * abs(r) / (2 * G.A0) : -2 * G.A2 * r / (G.A1 + Sq)
        L[k] = -c / 2 - real(lcdk) / 2 - log(pi) - real(la0k) / 2 + ex + log(k0scaled(z)) + G.del * (G.th + c) * b / 2
    end
    unwrap!(lcd); unwrap!(la0)
    for k in 1:n; L[k] -= im * (lcd[k] / 2 + la0[k] / 2); end
    L
end
"""abscissa on the real axis: the first s > 0 where cosh th det Sig, A1^2 - 4 A0 A2 or -A0 leaves (0, Inf)"""
const IIABS = Dict{Tuple{Int,Float64},Float64}()
function iiabs(c::Float64, model::Int)
    lock(RULELOCK) do; get!(IIABS, (model, c)) do
        okf(s) = (G = girs_fast(model, c, complex(s)); v = exp(G.lch + log(G.dS));
                  real(v) > 0 && abs(imag(v)) < 1e-9 * abs(v) && real(G.A1 * G.A1 - 4 * G.A0 * G.A2) > 0 && real(-G.A0) > 0)
        sp = (c * c + pi^2 / 4) / 2
        lo = 0.0; hi = sp * (1 - 1e-12)
        okf(hi) && return hi
        for _ in 1:200
            m = (lo + hi) / 2; (m == lo || m == hi) && break
            okf(m) ? (lo = m) : (hi = m)
        end
        lo
    end; end
end
"""the saddle data (s, a1, g2, L0) of log J(s;r) - s b, warm-started from `prev` when given"""
function iisaddle(b::Float64, r::Float64, c::Float64, model::Int, prev)
    abs(r) < 1e-200 && (r = 1e-200)
    f = S -> real(iilog([complex(S)], b, c, model, r)[1])
    sab = iiabs(c, model)
    s, a1, g2, L0 = prev === nothing ? rsad1(f, b, sab) : rsad_warm(f, b, sab, prev[1], prev[2])
    if s > sab - 1 / b                                                # a saddle pinned at the abscissa: cross 1/b to its left
        s = sab - 1 / b; a1 = 1 / b; g2 = b * b; L0 = f(s)
    end
    (s, a1, g2, L0)
end
"""cdf: G(r,b) = (1/2 pi i) int J(s;r) e^{-s b}/s ds (Re s > 0), pdf: h(r,b) = (1/2 pi i) int J(s;r) e^{-s b} ds, for the
pair (b, r), cases DU and DT with a zero start"""
function iipair(b::Float64, r::Float64, c::Float64, model::Int, pdfm::Bool, NN::Int; sad = nothing)
    tol = 1e-13
    abs(r) < 1e-200 && (r = 1e-200)                                 # (log singularity of g at r = 0)
    lk(S) = iilog(S, b, c, model, r)
    sab = iiabs(c, model)
    sad === nothing && (sad = iisaddle(b, r, c, model, nothing))
    s, a1, g2, L0 = sad
    Lh = L0 - log(2pi * max(g2, 1e-6 * b * b)) / 2
    lw = pdfm ? 1.5log(b) : log(b) + log(1 + abs(c) + abs(r) / b) - log(max(abs(s), a1))
    if Lh + lw < -35
        (!pdfm && s < 0) && return real(exp(lk([complex(0.0)])[1])), 0.0   # the residue at the pole
        return 0.0, 0.0
    end
    function evalpath(pth::Int, NNq::Int)
        if pth == 1
            sq = s
            if !pdfm
                abs(sq) < a1 && (sq = sq >= 0 ? a1 : -a1)
                sq = min(sq, sab - 0.5a1)
            end
            I, E = pathW(lk, b, sq, a1 / 2, NNq; divS = !pdfm)
            (!pdfm && sq < 0) && (I += real(exp(lk([complex(0.0)])[1])))
        else
            I, E, sA = pathT(lk, b, r, s, a1, NNq, c; divS = !pdfm, polegap = !pdfm)
            (!pdfm && sA < 0) && (I += real(exp(lk([complex(0.0)])[1])))
        end
        I, E
    end
    agreepaths(evalpath, 2, NN, tol)
end
"""P[k1 Z1^2 + k2 Z2^2 <= a], k1 > 0 > k2 real: 2 int_{l0}^inf phi(l; s2) erf(sqrt((2a + l^2)/(2 s1))) dl"""
function Pk0(k1::Float64, k2::Float64, a::Float64)
    s1 = 2k1; s2 = -2k2; l0 = sqrt(max(0.0, -2a)); sd = sqrt(s2)
    f(l) = 2 * exp(-l * l / (2s2)) / sqrt(2pi * s2) * erf(sqrt(max(2a + l * l, 0.0) / (2s1)))
    quadgk(f, l0, l0 + 40sd; atol = 1e-15, rtol = 1e-13)[1]
end

# ---------- the sharp step of the outer integrand in the far upper tail (c > c0) ----------
"""the conditional mean m and standard deviation sd of R given V = b from the tilt at the saddle point s* of
log M_V(s) - s b (double saddle point: Var[R | V] = K_uu - K_us^2/K_ss)"""
function condmom(model::Int, c::Float64, smx::Float64, lb::Float64)
    b = exp(lb)
    if model == 1
        Lf = S -> real(dklam([complex(S)], b, c, 1.0, 1.0, 1)[1])
        mf = S -> (k1 = 1 / (2 * dketa(S, c)); (k1, 2k1 * k1))
    else
        Lf = S -> (G = girs_fast(model, c, complex(S)); real(-c / 2 - G.lch / 2 - log(G.D) / 2 + G.del * (G.th + c) * b / 2))
        mf = S -> (G = girs_fast(model, c, complex(S)); A2 = real(G.A2); A1 = real(G.A1); A0 = real(G.A0); (A1 / (2A2), ((A1 / A2)^2 - 2A0 / A2) / 2))
    end
    s, a1, _, _ = rsad1(Lf, b, smx)
    h = min(1e-3 * a1, (smx - s) / 4)
    m, vt = mf(s); mp = mf(s + h)[1]; mm = mf(s - h)[1]
    dm = (mp - mm) / (2h)
    kss = (Lf(s - h) - 2Lf(s) + Lf(s + h)) / h^2
    vc = max(vt - dm * dm / kss, 1e-8vt)
    m, sqrt(vc)
end
const STEPGRID = Dict{Tuple{Int,Float64,Float64,Float64},Any}()
"""the conditional mean and standard deviation of R given V = b on a coarse grid in log b over [lo, hi]"""
function stepgrid(model::Int, c::Float64, lo::Float64, hi::Float64)
    lock(RULELOCK) do; get!(STEPGRID, (model, c, lo, hi)) do
        smx = model == 1 ? dksmax(c) : (t = tstar(model, c); isfinite(t) ? t : (c * c + pi^2 / 4) / 2)
        lb = collect(range(lo, hi; length = 61))
        r = Vector{Tuple{Float64,Float64}}(undef, 61)
        tmap!(j -> condmom(model, c, smx, lb[j]), r, 61)
        (smx = smx, lb = lb, m = first.(r), sd = last.(r))
    end; end
end
"""the points where a(b) = 1/2 + x sqrt(b) + c0 b crosses the conditional mean of R (sign changes on the grid, refined
by the Illinois variant of regula falsi), and the widths in log b, sd/|d(a - m)/d log b|; only those below wmax"""
function steproots(model::Int, c::Float64, Gr, x::Float64, c0::Float64, wmax::Float64)
    phf(l, m) = 0.5 + x * exp(l / 2) + c0 * exp(l) - m
    ph = [phf(Gr.lb[j], Gr.m[j]) for j in eachindex(Gr.lb)]
    ok = isfinite.(ph) .& isfinite.(Gr.sd)
    ls = Float64[]; ws = Float64[]
    for j in 1:length(ph)-1
        (ok[j] && ok[j+1] && sign(ph[j]) != sign(ph[j+1])) || continue
        la = Gr.lb[j]; lc = Gr.lb[j+1]; fa = ph[j]; fc = ph[j+1]; sa = Gr.sd[j]; sdc = Gr.sd[j+1]; side = 0
        for _ in 1:6
            l = la - fa * (lc - la) / (fc - fa)
            mn, sn = condmom(model, c, Gr.smx, l); fn = phf(l, mn)
            isfinite(fn) || break
            if sign(fn) == sign(fa)
                la = l; fa = fn; sa = sn; side == -1 && (fc /= 2); side = -1
            else
                lc = l; fc = fn; sdc = sn; side == 1 && (fa /= 2); side = 1
            end
        end
        t = fa / (fa - fc); l = la + t * (lc - la); sd = sa + t * (sdc - sa)
        m1 = condmom(model, c, Gr.smx, l - 1e-3)[1]; m2 = condmom(model, c, Gr.smx, l + 1e-3)[1]
        sl = (phf(l + 1e-3, m2) - phf(l - 1e-3, m1)) / 2e-3
        wd = sd / abs(sl)
        wd < wmax && (push!(ls, l); push!(ws, max(wd, 1e-6)))
    end
    ls, ws
end

# ---------- the outer pieces in log b (zero start) ----------
"""the pieces [PA, PA + PL] in log b of the points x: breakpoints at the zeros of a(b) (the support edge or the kink),
pieces of at most plen, grading toward the near-double zeros for c0 > 0, grading toward the step of the integrand
(steproots), and (dk) the pieces between the two zeros dropped, where a(b) < 0"""
function pieces_logb(x::Vector{Float64}, model::Int, c::Float64, c0::Float64, lo::Float64, hi::Float64, plen::Float64, stepw::Float64, dk::Bool)
    Gr = stepgrid(model, c, lo, hi)
    PA = Float64[]; PL = Float64[]; PX = Int[]; PEL = Bool[]; PER = Bool[]; brk = Vector{Vector{Float64}}(undef, length(x))
    for k in eachindex(x)
        xk = x[k]
        if c0 == 0
            rts = [-1 / (2xk)]
        else
            d = xk * xk - 2c0
            rts = d >= 0 ? [(-xk - sqrt(d)) / (2c0), (-xk + sqrt(d)) / (2c0)] : Float64[]
        end
        rts = [r for r in rts if r > 0]
        lr = sort([2log(r) for r in rts]); lr = [l for l in lr if l > lo && l < hi]
        sp = Float64[]; sg = Inf
        if c0 > 0 && xk < 0
            r0 = -xk / (2c0); amin = 0.5 - xk * xk / (4c0); l0 = 2log(r0)
            if length(lr) == 2 && lr[2] - lr[1] < 1e-10
                lr = [(lr[1] + lr[2]) / 2]                               # a double zero
            elseif length(lr) == 2
                sg = lr[2] - lr[1]
            elseif l0 > lo && l0 < hi
                if amin <= 1e-13; lr = [l0]; else; sg = 2sqrt(amin / c0) / r0; sp = [l0]; end
            end
        end
        brk[k] = exp.(lr)
        pts = [lo; hi; lr]
        if sg < plen
            for g in [lr; sp]
                for d in sg .* 2.0 .^ (0:ceil(Int, log2(plen / sg)))
                    push!(pts, g - d); push!(pts, g + d)
                end
            end
        end
        ls, ws = steproots(model, c, Gr, xk, c0, stepw)
        for g in eachindex(ls)
            push!(pts, ls[g])
            for d in 2ws[g] .* 2.0 .^ (0:ceil(Int, log2(plen / (2ws[g]))))
                push!(pts, ls[g] - d); push!(pts, ls[g] + d)
            end
        end
        pts = [p for p in pts if lo <= p <= hi]
        for g in lr
            pts = [p for p in pts if !(abs(p - g) < 1e-9 * min(max(sg, 1e-3), plen) && p != g)]   # no slivers at a zero
        end
        pts = sort(unique(pts))
        dk && length(lr) == 2 && (pts = [p for p in pts if p <= lr[1] || p >= lr[2]])
        for p in 1:length(pts)-1
            dk && length(lr) == 2 && pts[p] == lr[1] && continue           # a(b) < 0 between the zeros
            np = max(1, ceil(Int, (pts[p+1] - pts[p]) / plen)); ed = range(pts[p], pts[p+1]; length = np + 1)
            for q in 1:np
                push!(PA, ed[q]); push!(PL, ed[q+1] - ed[q]); push!(PX, k)
                push!(PEL, q == 1 && any(==(pts[p]), lr)); push!(PER, q == np && any(==(pts[p+1]), lr))
            end
        end
    end
    PA, PL, PX, PEL, PER, brk
end

"""the mean of int_0^1 J^2 (case DK), an upper bound of E V in all three cases"""
EVdk(c) = abs(c) < 1e-8 ? 0.5 : (exp(2c) - 1 - 2c) / (4c * c)

"""cases DK, DU, DT with a zero start by the double integral; returns (F, info)"""
function double_zero(x::Vector{Float64}, model::Int, c::Float64, c0::Float64, pdfm::Bool; M::Int = model == 1 ? 32 : 64, plen::Float64 = 2.0,
                     Nv::Int = 96, otol::Float64 = 1e-12, maxlev::Int = 12, stepw::Float64 = 0.1, errtol::Float64 = 1e-10, blo::Float64 = 8.0, bhi::Float64 = 6.0, bmin::Float64 = 3e-3)
    t0 = time()
    EB = EVdk(c)
    lo = model == 1 ? min(log(bmin), log(EB) - blo) : log(bmin); hi = log(EB) + bhi
    OR = outer_rules(M)
    zs, ws = OR.u, OR.gw
    ze, we = model == 1 ? (OR.z1, OR.w1) : (OR.z2, OR.w2)             # the cosine map once (the square-root edge) or twice (the K_0 edge)
    PA, PL, PX, PEL, PER, brk = pieces_logb(x, model, c, c0, lo, hi, plen, stepw, model == 1)
    nx = length(x)
    npairs_all = Ref(0)
    evalH = (B, IX) -> begin
        n = length(B); H = Vector{Float64}(undef, n); E = similar(H)
        # the pairs come in blocks of M consecutive b's of one piece: the saddle points are found along each block with
        # warm starts (a few evaluations per pair instead of a hundred), blocks in parallel; then the path integrals of
        # all pairs in parallel
        nblk = n ÷ M
        sads = Vector{Any}(undef, n)
        tmap!(p -> begin
            prev = nothing
            for j in (p-1)*M+1:p*M
                b = B[j]; xx = x[IX[j]]; a = 0.5 + xx * sqrt(b) + c0 * b
                if model == 1
                    sads[j] = a > 0 ? dksaddles(b, a, c, pdfm, prev) : nothing
                else
                    sads[j] = iisaddle(b, a, c, model, prev)
                end
                sads[j] === nothing || (prev = sads[j])
            end
            nothing
        end, Vector{Nothing}(undef, nblk), nblk)
        r = Vector{Tuple{Float64,Float64}}(undef, n)
        tmap!(j -> begin
            b = B[j]; xx = x[IX[j]]; a = 0.5 + xx * sqrt(b) + c0 * b
            if model == 1
                h, e = dkpair(b, a, c, pdfm, Nv; sad = sads[j])
                pdfm && (h *= sqrt(b); e *= sqrt(b))
            else
                h, e = iipair(b, a, c, model, pdfm, Nv; sad = sads[j])
                fac = pdfm ? sqrt(b) : xx / (2sqrt(b)) + c0
                h *= fac; e *= abs(fac)
            end
            (h, e)
        end, r, n)
        for j in 1:n; H[j], E[j] = r[j]; end
        H, E
    end
    F, err, nun, npairs, nref = adaptb(PA, PL, PEL, PER, PX, nx, zs, ws, ze, we, OR.LM, OR.gw, evalH, otol, maxlev)
    if model != 1 && !pdfm
        # the boundary term of the integration by parts: P[R <= a(b_lo)] without tilt, with the weights from the Girsanov
        # forms at s = 0 (the roots nu = 2 kap of A2 nu^2 - A1 nu + A0)
        G0 = girsanov(model, c, complex(0.0)); S0 = sqrt(G0.A1 * G0.A1 - 4 * G0.A0 * G0.A2)
        k10 = real((G0.A1 + S0) / (4 * G0.A2)); k20 = real(G0.A0 / (G0.A1 + S0)); blo_ = exp(lo)
        for k in 1:nx; F[k] += Pk0(k10, k20, 0.5 + x[k] * sqrt(blo_) + c0 * blo_); end
    end
    if pdfm && c0 > 0                                                 # the zeros of a(b) coincide: the density is infinite
        for k in 1:nx; abs(x[k] + sqrt(2c0)) <= 8eps() * max(1.0, sqrt(2c0)) && (F[k] = Inf); end
    end
    status = [2 * (err[k] > errtol) for k in 1:nx]
    F, (time = time() - t0, breaks = brk, EB = EB, nbad = zeros(Int, nx), nunres = nun, err = err, status = status, npairs = npairs, nrefine = nref)
end

# ---------- with an initial observation: fixed start g0 (DK, DU, DT) or stationary start (SK, SU, ST) ----------
"""the continuous logarithm of the values Z along a path (a vector in path order), starting from the real point Z[1]
> 0; the phase is unwrapped, and where consecutive phases differ by more than pi/4 the path is refined by the midpoint
values f(midpoint) up to 40 times"""
function pathlog(f, S::Vector{ComplexF64}, Z::Vector{ComplexF64}; round0::Bool = true)
    n = length(S); t = collect(0.0:n-1); tt = copy(t); lv = log.(Z)
    for _ in 1:40
        bad = [k for k in 1:length(tt)-1 if abs(angle(exp(im * (imag(lv[k+1]) - imag(lv[k]))))) > pi / 4]
        isempty(bad) && break
        tm = [(tt[k] + tt[k+1]) / 2 for k in bad]
        sm = [(fl = min(floor(Int, x), n - 2); fr = x - fl; S[fl+1] * (1 - fr) + S[fl+2] * fr) for x in tm]
        lm = log.(f(sm))
        tt = [tt; tm]; lv = [lv; lm]
        ix = sortperm(tt); tt = tt[ix]; lv = lv[ix]
    end
    ph = Vector{Float64}(undef, length(tt)); ph[1] = imag(lv[1])
    for k in 2:length(tt); ph[k] = ph[k-1] + angle(exp(im * (imag(lv[k]) - imag(lv[k-1])))); end
    round0 && (ph .-= 2pi * round(ph[1] / (2pi)))                   # (Z[1] > 0: the logarithm is real at the start)
    loc = indexin(t, tt)
    [real(lv[loc[k]]) + im * ph[loc[k]] for k in 1:n]
end

"""the pieces of the initial-value term of cases DU, DT: Q0 = gg^2 Q(0, s), and the squared tilted means m1^2, m2^2 of
(X1, X2) from Q(lam) - Q0 = (lam/2) [m1^2/(1 - lam s1) - m2^2/(1 + lam s2)] at lam = 1/(4 s1) and -1/(4 s2)"""
function qpieces(model::Int, c::Float64, S::ComplexF64, gg::Float64, s1::ComplexF64, s2::ComplexF64, qf = qstart_d)
    la = 0.25 / s1; lb = -0.25 / s2
    Q0 = gg^2 * qf(model, c, complex(0.0), S)
    Qa = gg^2 * qf(model, c, la, S) - Q0
    Qb = gg^2 * qf(model, c, lb, S) - Q0
    A11 = la / 2 / (1 - la * s1); A12 = -la / 2 / (1 + la * s2)
    A21 = lb / 2 / (1 - lb * s1); A22 = -lb / 2 / (1 + lb * s2)
    dt = A11 * A22 - A12 * A21
    Q0, (Qa * A22 - A12 * Qb) / dt, (A11 * Qb - A21 * Qa) / dt
end

"""density of R at aa (real) under the tilt with weights k1, k2 (K_0 form), or case DK with a fixed start"""
function gden(model::Int, dkfixed::Bool, g0::Float64, aa::Float64, k1::ComplexF64, k2::ComplexF64, mq::ComplexF64)
    if dkfixed
        if g0 != 0
            v = 2k1; m = g0 * mq; r2 = 2aa + g0^2
            r2 > 0 || return zero(ComplexF64)
            r = sqrt(r2)
            return (exp(-(r - m)^2 / (2v)) + exp(-(r + m)^2 / (2v))) / (sqrt(2pi * v) * r)
        end
        aa > 0 || return zero(ComplexF64)
        return exp(-aa / (2k1)) / sqrt(2pi * k1 * aa)
    end
    s2 = 1 / (2k1); s1 = 1 / (2k2); al = (s1 + s2) / 2; be = (s2 - s1) / 2
    z = be * abs(aa)
    exp(-al * aa - z) * k0scaled(z) / (2pi * sqrt(k1) * sqrt(-k2))
end

"""P(kap_1, kap_2; a) (cdf) or its derivative in a (pdf): central (Nl-point Gauss-Legendre in l after l = alpha sinh tau
for a >= 0, u = sqrt(2a + l^2) for a < 0), or noncentral with the squared means m1s, m2s (nc = true)"""
function Pkap(k1::ComplexF64, k2::ComplexF64, a::Float64, pdfm::Bool, Nl::Int; nc::Bool = false, m1s::ComplexF64 = zero(ComplexF64), m2s::ComplexF64 = zero(ComplexF64))
    zl, wl = rule(Nl)
    s1 = 2k1; s2 = -2k2
    m1 = sqrt(m1s); m2 = sqrt(m2s)
    lm = sqrt(2 * 41.4 / max(real(1 / s2), 1e-12)) + (nc ? abs(m2) : 0.0)   # range of l: exp(-l^2 Re(1/(2 s2))) < 1e-18
    F1(u) = nc ? (erf((u - m1) / sqrt(2s1)) + erf((u + m1) / sqrt(2s1))) / 2 : erf(u / sqrt(2s1))
    f1(u) = nc ? (exp(-(u - m1)^2 / (2s1)) + exp(-(u + m1)^2 / (2s1))) / sqrt(2pi * s1) : (2 / sqrt(pi)) * exp(-u * u / (2s1)) / sqrt(2s1)
    f2(l) = nc ? (exp(-(l - m2)^2 / (2s2)) + exp(-(l + m2)^2 / (2s2))) / (2sqrt(2pi * s2)) : exp(-l * l / (2s2)) / sqrt(2pi * s2)
    P = zero(ComplexF64)
    if a >= 0
        al = max(sqrt(2a), 1e-8lm); tm = asinh(lm / al)
        for k in 1:Nl
            T = zl[k] * tm; Lg = al * sinh(T); dL = al * cosh(T) * wl[k] * tm
            u = sqrt(2a + Lg * Lg)
            P += (pdfm ? f2(Lg) * f1(u) / u : f2(Lg) * F1(u)) * dL
        end
    else
        l0 = sqrt(-2a); um = sqrt(max(lm * lm - l0 * l0, 0.0) + 1e-30)
        for k in 1:Nl
            U = zl[k] * um; WU = wl[k] * um
            Lg = sqrt(U * U + l0 * l0)
            P += (pdfm ? f2(Lg) * f1(U) / Lg : f2(Lg) * F1(U) * U / Lg) * WU
        end
    end
    2P
end

"""cases with an initial observation (fixed start g0 != 0 in DK, DU, DT; stationary start in SK, SU, ST) by the
double integral; returns (F, info)"""
function double_start(x::Vector{Float64}, model::Int, c::Float64, c0::Float64, g0::Float64, stat::Bool, pdfm::Bool;
                      M::Int = 64, Nv::Int = 96, Nl::Int = 96, plen::Float64 = 2.0, otol::Float64 = 1e-12, maxlev::Int = 12, errtol::Float64 = 1e-10,
                      blo::Float64 = 8.0, bhi::Float64 = 6.0, mumin::Float64 = 4.5, tdec::Float64 = 41.5)
    t0 = time()
    s02 = stat ? -1 / (2c) : 0.0
    stat && (g0 = 0.0)
    st = stat && model == 1                                           # case SK (closed form)
    nz = st || (g0 != 0 && model == 1)                                # DK: initial-value terms in closed form
    nzR = model != 1 && (stat || g0 != 0)                             # DU, DT: initial-value terms (qstart)
    smax0 = tstar(model, c)                                           # real domain end of the zero-start M_V
    # log M_V(s) at real s with the initial-value terms; ok: s in the domain
    function LB(s::Float64)
        s < smax0 || return (-Inf, false)
        if model == 1
            G = girsanov(model, c, complex(s))
            v = real(G.eta); L = -c / 2 - real(G.lsh) / 2 - log(abs(v)) / 2      # (D = e^{lsh} eta with e^{lsh} = sinh th/th > 0)
        else
            # the determinant itself (the Girsanov forms are singular where cosh th = 0, inside the domain): D = e^{mu} Dt
            Dt, mu = Dscaled(model, c, complex(0.0), complex(s))
            v = real(mu) > 1 ? real(Dt) : real(Dt * exp(mu)); L = -(real(log(Dt)) + real(mu)) / 2
        end
        ok = isfinite(v) && v > 0 && isfinite(L)
        if nzR
            Q0 = real(qstart_d(model, c, complex(0.0), complex(s))); ok &= isfinite(Q0)
            if stat
                d1 = 1 - 2s02 * Q0; ok &= d1 > 0; L -= log(abs(d1)) / 2
            else
                L += g0^2 * Q0
            end
        end
        if nz
            aq = real(complex(s) / G.eta); ok &= isfinite(aq)
            if st
                d1 = 1 - 2aq * s02; ok &= d1 > 0; L -= log(abs(d1)) / 2
            else
                L += g0^2 * aq
            end
        end
        L, ok
    end
    hh = 1e-5; EB = (LB(hh)[1] - LB(-hh)[1]) / (2hh)
    # the initial-value terms move the right end of the real domain of M_V to the left; the crossing points of all paths
    # must lie to its left: located by bisection on the domain flag
    smax = smax0
    if !LB(smax * (1 - 1e-9))[2]
        lo = 0.0; hi = smax
        for _ in 1:80
            md = (lo + hi) / 2
            LB(md)[2] ? (lo = md) : (hi = md)
        end
        smax = lo
    end
    useg = !(nzR && !stat) && !(model == 1 && !st)                   # the density form (ii): the stationary starts
    dkfixed = model == 1 && !st
    K = model == 2 ? KDU : (model == 3 ? KDT : zeros(0, 0))
    # log M_V and the weights (and means) of R under the tilt at the nodes S of a path (S[1] real, in the domain)
    function nodepar(S::Vector{ComplexF64})
        n = length(S)
        Gs = model == 1 ? [girsanov(model, c, s) for s in S] : nothing
        if model == 1
            Lz = [-c / 2 - G.lsh / 2 for G in Gs]
            lz = pathlog(sm -> [girsanov(1, c, s).eta for s in sm], S, [G.eta for G in Gs])
            L = Lz .- lz ./ 2
            k1 = [1 / (2G.eta) for G in Gs]; k2 = zeros(ComplexF64, n)
        else
            # log M_V = -(log Dt + mu)/2 with D = e^{mu} Dt, the slowly varying Dt continued along the path and mu = sqrt(c^2 - 2s)
            # principal, continuous since Im s <= 0 on the path; the weights from D(u, s) at u = 0, -1, 1 (lam = -u)
            D0 = [Dscaled(model, c, complex(0.0), s) for s in S]
            lz = pathlog(sm -> [Dscaled(model, c, complex(0.0), s)[1] for s in sm], S, [d[1] for d in D0]; round0 = false)
            lt = [lz[k] + D0[k][2] for k in 1:n]                        # log D, real at the start node (D > 0 there): fix the multiple of 2 pi i
            lt .-= 2pi * im * round(imag(lt[1]) / (2pi))
            L = -lt ./ 2
            k1 = Vector{ComplexF64}(undef, n); k2 = similar(k1)
            for k in 1:n
                a0 = D0[k][1]; dp = Dscaled(model, c, complex(-1.0), S[k])[1]; dm = Dscaled(model, c, complex(1.0), S[k])[1]
                a1 = (dp - dm) / 2; a2 = (dp + dm) / 2 - a0
                p1 = a1 / (2a0); p2 = a2 / (4a0)                           # kap_1 + kap_2, kap_1 kap_2
                dsc = sqrt(p1 * p1 - 4p2)
                k1[k] = (p1 + dsc) / 2; k2[k] = (p1 - dsc) / 2
                real(k2[k]) > real(k1[k]) && ((k1[k], k2[k]) = (k2[k], k1[k]))
            end
        end
        mq = zeros(ComplexF64, n); m2s = zeros(ComplexF64, n)
        if nzR
            s1 = 2 .* k1; s2 = -2 .* k2
            if stat
                Q0 = similar(k1); mu1 = similar(k1); mu2 = similar(k1)
                for k in 1:n; Q0[k], mu1[k], mu2[k] = qpieces(model, c, S[k], 1.0, s1[k], s2[k]); end
                d1 = 1 .- 2s02 .* Q0
                ld = log.(d1); ph = unwrap!(imag.(ld)); ld = real.(ld) .+ im .* ph
                L .-= ld ./ 2
                st2 = s02 ./ d1
                for k in 1:n
                    p1 = (k1[k] + k2[k]) + st2[k] * (mu1[k] - mu2[k]) / 2
                    p2 = k1[k] * k2[k] + st2[k] * (mu1[k] * k2[k] - mu2[k] * k1[k]) / 2
                    dsc = sqrt(p1 * p1 - 4p2)
                    k1[k] = (p1 + dsc) / 2; k2[k] = (p1 - dsc) / 2
                    real(k2[k]) > real(k1[k]) && ((k1[k], k2[k]) = (k2[k], k1[k]))
                end
            else
                for k in 1:n
                    Q0, mq[k], m2s[k] = qpieces(model, c, S[k], g0, s1[k], s2[k])
                    L[k] += Q0
                end
            end
        end
        if nz
            aq = [S[k] / Gs[k].eta for k in 1:n]; mqk = [exp(-Gs[k].lsh) / Gs[k].eta for k in 1:n]
            if st
                d1 = 1 .- 2s02 .* aq
                ld = log.(d1); ph = unwrap!(imag.(ld)); ld = real.(ld) .+ im .* ph
                L .-= ld ./ 2
                for k in 1:n
                    t2 = s02 / d1[k]; v = 2k1[k]
                    p1 = (mqk[k]^2 * t2 + v - t2) / 2; p2 = -t2 * v / 4
                    dsc = sqrt(p1 * p1 - 4p2)
                    k1[k] = (p1 + dsc) / 2; k2[k] = (p1 - dsc) / 2
                    real(k2[k]) > real(k1[k]) && ((k1[k], k2[k]) = (k2[k], k1[k]))
                end
            else
                L .+= g0^2 .* aq; mq .= mqk
            end
        end
        L, k1, k2, mq, m2s
    end
    Pk(k1, k2, a, mq, m2) = begin
        (nzR && !stat) && return Pkap(k1, k2, a, pdfm, Nl; nc = true, m1s = mq, m2s = m2)
        if dkfixed
            m = g0 * mq; r = sqrt(2a + g0^2); sg = sqrt(4k1)             # J(1) ~ N(m, 2 kap_1), {J(1)^2 <= r^2}
            pdfm && return (exp(-((r - m) / sg)^2) + exp(-((r + m) / sg)^2)) / (sqrt(pi) * sg / 2) / r / 2
            return (erf((r - m) / sg) + erf((r + m) / sg)) / 2
        end
        Pkap(k1, k2, a, pdfm, Nl)
    end
    k10 = k20 = mq0 = zero(ComplexF64)
    if useg && !pdfm
        _, k1u, k2u, mqu, _ = nodepar([complex(0.0)]); k10 = k1u[1]; k20 = k2u[1]; mq0 = mqu[1]
    end
    OR = outer_rules(M)
    nx = length(x); F = zeros(nx); err = zeros(nx); nun = zeros(Int, nx); nref = zeros(Int, nx); nbad = zeros(Int, nx); brk = Vector{Vector{Float64}}(undef, nx)
    lo = log(EB) - blo; hi = log(EB) + bhi
    blo_ = exp(lo); brng = exp(hi) - blo_
    q0 = dkfixed ? (1 + g0^2) / 2 : 0.5                                # edge (DK) or kink: a(b) + g0^2/2 = 0
    for i in 1:nx
        xx = x[i]; nbadi = Ref(0); nunres = Ref(0)
        if c0 == 0
            sr = [-q0 / xx]
        else
            d = xx * xx - 4c0 * q0
            sr = d >= 0 ? [(-xx - sqrt(d)) / (2c0), (-xx + sqrt(d)) / (2c0)] : Float64[]
        end
        sr = [r for r in sr if r > 0]; bk = sort([log(r * r) for r in sr]); bk = [l for l in bk if lo < l < hi]
        brk[i] = exp.(bk)
        br = [lo; bk; hi]; bb = [br[1]]
        for jj in 2:length(br)
            np0 = max(1, ceil(Int, (br[jj] - br[jj-1]) / plen))
            append!(bb, br[jj-1] .+ (1:np0) .* (br[jj] - br[jj-1]) ./ np0)
        end
        PA = bb[1:end-1]; PL = bb[2:end] .- PA
        PEL = [useg && any(l -> abs(a - l) < 1e-12, bk) for a in PA]; PER = [useg && any(l -> abs(b - l) < 1e-12, bk) for b in PA .+ PL]
        # H(b) = E[delta(V - b) 1{R <= a(b)}] (cdf) or sqrt(b) d/da of it (pdf), for the b's of one piece
        function Hb(bv::Vector{Float64})
            a = 0.5 .+ xx .* sqrt.(bv) .+ c0 .* bv
            H = zeros(length(bv))
            act = dkfixed ? findall(a .+ g0^2 / 2 .> 0) : collect(1:length(bv))
            isempty(act) && return H
            bs = bv[act]; as = a[act]; nb = length(bs)
            s = zeros(nb); a1 = zeros(nb); sbd = fill(smax, nb)
            for j in 1:nb
                # saddle point s* of log M_V(s) - s b: log M_V is convex on its real domain, so g1(s) = L'(s) - b increases;
                # a safeguarded Newton iteration keeps a bracket [slo, shi] (points outside the domain count as g1 = +Inf)
                sj = j > 1 ? s[j-1] : 0.0; slo = -Inf; shi = smax; sb = smax; g2 = 1.0   # (warm start: the b's of a block increase)
                for _ in 1:200
                    L0, ok0 = LB(sj)
                    bad = !ok0 || sj >= smax
                    bad && (shi = sj; sb = min(sb, sj))
                    e = min(1e-4 * max(1.0, abs(sj)), 0.1 * (sb - sj)); okp = false; Lp = L0; Lm = L0
                    for _ in 1:30
                        (okp || bad) && break
                        Lpt, okt = LB(sj + e); Lmt = LB(sj - e)[1]; okt &= sj + e < smax
                        Lp = Lpt; Lm = Lmt; okp = okt
                        okt || (e /= 10; sb = min(sb, sj + 10e))
                    end
                    g1 = (Lp - Lm) / (2e) - bs[j]; g2a = (Lp - 2L0 + Lm) / e^2
                    gd = !bad && okp
                    gd && g1 < 0 && (slo = max(slo, sj))
                    gd && g1 >= 0 && (shi = min(shi, sj))
                    gd && g2a > 0 && (g2 = g2a)
                    sn = (gd && g2a > 0) ? sj - g1 / g2a : NaN
                    A0 = slo; B0 = shi
                    if !(sn > A0 && sn < B0)
                        if isfinite(A0) && isfinite(B0); sn = (A0 + B0) / 2
                        elseif !isfinite(A0); sn = B0 - max(1.0, 2abs(B0))
                        else; sn = A0 + max(1.0, abs(A0)); end
                    end
                    conv = gd && (abs(g1) < 1e-9 * bs[j] || (B0 - A0) < 1e-12 * (1 + abs(sj)))
                    conv && break
                    sj = sn
                end
                # a fixed start in cases DU, DT: the initial-value term has an essential singularity at the end of the
                # domain; the point is kept at a relative distance of 2% from it
                (nzR && !stat && isfinite(smax0)) && (sj = min(sj, smax0 - 0.02 * max(1.0, abs(smax0))))
                for _ in 1:2                                             # L'' at the scale of the saddle-point width
                    a1j = 1 / sqrt(g2)
                    e = min(0.05a1j, 0.3 * (sb - sj))
                    L0 = LB(sj)[1]; Lp, okp = LB(sj + e); Lm, okm = LB(sj - e); okp &= sj + e < smax
                    g2n = (Lp - 2L0 + Lm) / e^2
                    (okp && okm && g2n > 0 && isfinite(g2n)) && (g2 = g2n)
                end
                a1j = 1 / sqrt(g2)
                if useg && !pdfm                                         # keep the crossing point one width from the pole at 0
                    if abs(sj) < a1j
                        sj = (sj >= 0 && a1j < 0.5sb) ? a1j : -a1j
                    end
                end
                s[j] = sj; a1[j] = a1j; sbd[j] = sb
                DBG[] && println("  b = ", bs[j], " a = ", as[j], " s* = ", sj, " a1 = ", a1j, " sbd = ", sb, " smax = ", smax, " LB(s*) = ", LB(sj))
            end
            mu = max.(a1 ./ 2, mumin); sth = a1 ./ (2mu); thm = sqrt.(tdec ./ (bs .* mu))
            # the Talbot-type parabola s* + mu th^2 - 2 i mu th, th >= 0, through the saddle point; trapezoidal rule in
            # tau, th = sth sinh(tau), with the nested rule as error estimate; unresolved columns recomputed with 2 Nv, 4 Nv
            function pathint(j::Int, NN::Int)
                ht = asinh(thm[j] / sth[j]) / NN
                S = Vector{ComplexF64}(undef, NN + 1); W = similar(S)
                for k in 0:NN
                    tau = k * ht; TH = sth[j] * sinh(tau); w = sth[j] * cosh(tau) * ht; k == 0 && (w /= 2)
                    S[k+1] = s[j] + mu[j] * TH * TH - 2im * mu[j] * TH
                    W[k+1] = 2mu[j] / pi * (1 + im * TH) * w
                end
                L, k1, k2, mq, m2 = nodepar(S)
                nbadi[] += count(k -> real(k1[k]) <= 0, 1:NN+1) + ((model != 1 || st) ? count(k -> real(k2[k]) >= 0, 1:NN+1) : 0)
                G = Vector{ComplexF64}(undef, NN + 1)
                for k in 1:NN+1
                    if useg
                        g = gden(model, dkfixed, g0, as[j], k1[k], k2[k], mq[k])
                        G[k] = pdfm ? exp(L[k] - S[k] * bs[j]) * g * sqrt(bs[j]) : exp(L[k] - S[k] * bs[j]) * g / S[k]
                    else
                        P = Pk(k1[k], k2[k], as[j], mq[k], m2[k])
                        pdfm && (P *= sqrt(bs[j]))
                        G[k] = exp(L[k] - S[k] * bs[j]) * P
                    end
                end
                if DBG[] && any(!isfinite, G)
                    k = findfirst(!isfinite, G); println("    NaN at node ", k, ": S = ", S[k], " L = ", L[k], " k1 = ", k1[k], " k2 = ", k2[k], " mq = ", mq[k], " m2 = ", m2[k], " a = ", as[j], " b = ", bs[j])
                end
                Ij = real(sum(G[k] * W[k] for k in 1:NN+1)); Ihj = real(sum(G[k] * 2W[k] for k in 1:2:NN+1))   # (names distinct from Hb's I, Ih: a closure assigns the enclosing variable)
                Ij, Ihj
            end
            I = zeros(nb); Ih = zeros(nb)
            for j in 1:nb; I[j], Ih[j] = pathint(j, Nv); DBG[] && println("  I = ", I[j], " Ih = ", Ih[j]); end
            sc = maximum(abs, I) + floatmin(); tabs = 1e-11 / brng
            for rf in 1:2
                bad = [j for j in 1:nb if abs(I[j] - Ih[j]) > 1e-9sc && abs(I[j] - Ih[j]) > tabs]
                isempty(bad) && break
                for j in bad; I[j], Ih[j] = pathint(j, Nv * 2^rf); end
            end
            nunres[] += count(j -> abs(I[j] - Ih[j]) > 1e-9sc && abs(I[j] - Ih[j]) > tabs, 1:nb)
            if useg && !pdfm
                for j in 1:nb
                    s[j] < 0 && (I[j] += real(gden(model, dkfixed, g0, as[j], k10, k20, mq0)))
                    I[j] *= xx / (2sqrt(bs[j])) + c0
                end
            end
            H[act] .= I
            H
        end
        evalH = (B, IX) -> begin
            # the pieces' b's come in blocks of M: one Hb call per block, blocks in parallel
            nblk = length(B) ÷ M
            Hs = Vector{Vector{Float64}}(undef, nblk)
            tmap!(p -> Hb(B[(p-1)*M+1:p*M]), Hs, nblk)
            reduce(vcat, Hs), zeros(length(B))
        end
        zs, ws = OR.z1, OR.w1; ze, we = OR.z2, OR.w2
        Fi, erri, _, _, nrefi = adaptb(PA, PL, PEL, PER, ones(Int, length(PA)), 1, zs, ws, ze, we, OR.LM, OR.gw, evalH, otol, maxlev)
        F[i] = Fi[1]; err[i] = erri[1]; nref[i] = nrefi[1]; nbad[i] = nbadi[]; nun[i] = nunres[]
        if useg && !pdfm                                              # the boundary term of the integration by parts
            alo = 0.5 + xx * sqrt(blo_) + c0 * blo_
            F[i] += real(Pk(k10, k20, alo, mq0, zero(ComplexF64)))
        end
    end
    status = [1 * (nbad[k] > 0) + 2 * (nun[k] > 0 || err[k] > errtol) for k in 1:nx]
    F, (time = time() - t0, breaks = brk, EB = EB, smax = smax, nbad = nbad, nunres = nun, err = err, status = status, nrefine = nref)
end

"""
    tr_limdist(x, model, c; c0 = 0.0, g0 = 0.0, start = :zero, pdf = false, opts...) -> (F, info)

The limiting distribution function (or density, pdf = true) by the double integral of Theorem S1 of the Supplementary Material at the points x (a
vector), with a fixed start g0 or a stationary start.  info.status is 0 where the computation is clean, otherwise 1
(sign violations on the path) plus 2 (unresolved inner integrals, or an error estimate above errtol); a nonzero
status gives a warning.  Options: M (outer nodes per piece), Nv (nodes on the contour), Nl (nodes of the integral over l
for the noncentral probability), plen, otol, maxlev, stepw, errtol, blo, bhi, bmin, mumin, tdec.
"""
function tr_limdist(x::Vector{Float64}, model::Int, c::Float64; c0::Float64 = 0.0, g0::Float64 = 0.0, start::Symbol = :zero, pdf::Bool = false, warn::Bool = true, kw...)
    stat = startflag(start, c)
    stat && g0 != 0 && error("a fixed start g0 applies to the zero-start cases only")
    if !stat && g0 == 0
        F, info = double_zero(x, model, c, c0, pdf; kw...)
    else
        F, info = double_start(x, model, c, c0, g0, stat, pdf; kw...)
    end
    if warn && any(!=(0), info.status)
        iw = findfirst(!=(0), info.status)
        @warn "tr_limdist: $(count(!=(0), info.status)) of $(length(x)) points with nonzero status (first: x = $(x[iw]), status $(info.status[iw]), error estimate $(info.err[iw])); the result may be inaccurate there"
    end
    F, info
end

end # module
