# The conditional integral representation (Theorem 3(i)) of the limit in case DK (zero start, c0 = 0) with the
# t-inversion done by the FFT, Carr-Madan style, on a uniform grid in v = V, for all v at once:
#   H(v;x) = (1/2 pi i) int P(t) Psi_t(a(v)) e^{-tv} dt,   a(v) = 1/2 + x sqrt(v),
#   Psi_t(a) = erf(sqrt(s1 a))/sqrt(s1) = sqrt(a) E_t(a),  E_t entire in a,
# one FFT per Chebyshev node a_j of a in [0, amax] gives E(v_k, a_j) = H~(v_k, a_j)/sqrt(a_j) on the tensor grid;
# then F(x) = int_0^inf H(v;x) dv per x by Gauss-Legendre in a variable that removes the sqrt onset at the support
# edge, with E interpolated (barycentric Chebyshev in a, local polynomial in v).  The line is Re t = sigma; the
# integrand is analytic for Re t < (c^2 + pi^2)/2 (the first zero of sinh(mu)/mu), so the trapezoidal rule in omega
# converges like exp(-2 pi d/dw) with d the distance of the line to that point.
using FFTW, SpecialFunctions, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "DFExact.jl")); using .DFExact

struct Table
    c::Float64; sigma::Float64; dv::Float64; nv::Int; amax::Float64
    anodes::Vector{Float64}; bw::Vector{Float64}        # Chebyshev-Lobatto nodes in a and barycentric weights
    E::Matrix{Float64}                                  # nv x (na+1): E(v_k, a_j)
    P0::Float64; s10::Float64                           # P(0), s1(0) for checks
end

function build(c::Float64; sigma::Float64 = 0.0, N::Int = 2^16, dw::Float64 = 0.25, amax::Float64 = 21.0, na::Int = 96)
    dv = 2pi / (N * dw)
    w = ((0:N-1) .- N ÷ 2) .* dw
    t = ComplexF64[sigma + im * wi for wi in w]
    P, s1, _, _ = DFExact.tparts(1, c, t)
    minimum(real, s1) > 0 || @warn "Re s1 <= 0 on the line" minimum(real, s1)
    anodes = [amax * (1 - cos(j * pi / na)) / 2 for j in 0:na]
    bw = [(-1.0)^j * (j == 0 || j == na ? 0.5 : 1.0) for j in 0:na]
    E = Matrix{Float64}(undef, N, na + 1)
    plan = plan_fft(Vector{ComplexF64}(undef, N); flags = FFTW.MEASURE)
    gs = [Vector{ComplexF64}(undef, N) for _ in 1:Threads.nthreads()]
    Gs = [Vector{ComplexF64}(undef, N) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for j in 0:na
        a = anodes[j+1]; g = gs[Threads.threadid()]; G = Gs[Threads.threadid()]
        for m in 1:N
            if a == 0
                g[m] = P[m] * 2 / sqrt(pi)
            else
                z = sqrt(s1[m] * a)                     # erf(z)/z is even in z: the branch is immaterial
                g[m] = P[m] * erf(z) / z
            end
        end
        mul!(G, plan, g)
        fac = dw / (2pi)
        for k in 0:N-1
            E[k+1, j+1] = real(G[k+1]) * (iseven(k) ? fac : -fac) * exp(-sigma * k * dv)
        end
    end
    P0, s10, _, _ = DFExact.tparts(1, c, [complex(0.0, 0.0)])
    Table(c, sigma, dv, N, amax, anodes, bw, E, real(P0[1]), real(s10[1]))
end

# E(v, a) from the table: 6-point Lagrange in v on the uniform grid, barycentric Chebyshev in a
function interp(T::Table, v::Float64, a::Float64)
    k = clamp(floor(Int, v / T.dv) - 2, 0, T.nv - 6)       # rows k+1 .. k+6
    u = v / T.dv - k                                       # in [2, 3] away from the ends
    # Lagrange weights for nodes 0..5 at u
    lw = ntuple(i -> begin
            p = 1.0
            for m in 0:5
                m == i - 1 && continue
                p *= (u - m) / (i - 1 - m)
            end
            p
        end, 6)
    na = length(T.anodes) - 1
    num = 0.0; den = 0.0
    for j in 0:na
        d = a - T.anodes[j+1]
        ev = 0.0
        for i in 1:6
            ev += lw[i] * T.E[k+i, j+1]
        end
        if abs(d) < 1e-14
            return ev
        end
        q = T.bw[j+1] / d
        num += q * ev; den += q
    end
    num / den
end

const GL = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
function gl(n::Int)
    get!(GL, n) do
        b = [k / sqrt(4k^2 - 1) for k in 1:n-1]
        J = SymTridiagonal(zeros(n), b); ev = eigen(J)
        ev.values, 2 * ev.vectors[1, :] .^ 2
    end
end

# F(x) = int_0^inf sqrt(a(v)) E(v, a(v)) dv, a(v) = 1/2 + x sqrt(v)
function cdf_fft(T::Table, x::Float64; npan::Int = 6, n::Int = 48)
    xn, wn = gl(n)
    F = 0.0
    if x >= 0
        wmax = sqrt((T.nv - 6) * T.dv)                       # v up to the end of the table
        wmax = min(wmax, sqrt((T.amax - 0.5)^2 / max(x, 1e-300)^2))   # a(v) <= amax
        edges = range(0.0, wmax, length = npan + 1)
        for p in 1:npan
            lo, hi = edges[p], edges[p+1]; h = (hi - lo) / 2; mid = (hi + lo) / 2
            for i in 1:n
                w = mid + h * xn[i]; v = w * w; a = 0.5 + x * w
                F += wn[i] * h * 2 * w * sqrt(a) * interp(T, v, a)
            end
        end
    else
        # a = p^2, p in [0, sqrt(1/2)]; w = (p^2 - 1/2)/x = (1/2 - p^2)/|x|; dv = 2 w dw, |dw/dp| = 2p/|x|
        pmax = sqrt(0.5)
        wmax = sqrt((T.nv - 6) * T.dv)                       # the table ends at v = wmax^2; H is negligible there
        pmin = sqrt(max(0.0, 0.5 - abs(x) * wmax))           # w <= wmax  <=>  p^2 >= 1/2 - |x| wmax
        edges = range(pmin, pmax, length = npan + 1)
        for pn in 1:npan
            lo, hi = edges[pn], edges[pn+1]; h = (hi - lo) / 2; mid = (hi + lo) / 2
            for i in 1:n
                p = mid + h * xn[i]; w = (0.5 - p * p) / abs(x); v = w * w
                F += wn[i] * h * 4 * p * p * w / abs(x) * interp(T, v, p * p)
            end
        end
    end
    F
end

# checks of the table: int H~(v, a) dv = P(R <= a) = P(0) erf(sqrt(s1(0) a)) / sqrt(s1(0))
function check_table(T::Table)
    worst = 0.0
    for (j, a) in enumerate(T.anodes)
        a == 0 && continue
        num = sum(@view T.E[:, j]) * T.dv * sqrt(a)
        ex = T.P0 * erf(sqrt(T.s10 * a)) / sqrt(T.s10)
        worst = max(worst, abs(num - ex))
    end
    worst
end
