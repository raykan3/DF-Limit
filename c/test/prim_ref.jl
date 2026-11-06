# reference values of the primitives for the C engine test
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
using SpecialFunctions
out = open(ARGS[1], "w")
w(tag, vals...) = (print(out, tag); for v in vals; if v isa Complex; @printf(out, " %.17g %.17g", real(v), imag(v)); else; @printf(out, " %.17g", v); end; end; println(out))
for z in (complex(0.3, 0.2), complex(1.5, -0.7), complex(7.2, 3.1), complex(0.4, 22.0), complex(30.0, -5.0), complex(-2.0, 1.0), complex(-20.0, 3.0))
    w("k0", z, D.k0scaled(z))
    w("erf", z, erf(z)); w("faddw", z, erfcx(-im * z))
end
for m in 1:3, c in (-5.0, 0.0, 2.5, 7.0), (s, t) in ((complex(0.3), complex(0.5)), (complex(-1.0, 0.4), complex(2.0, -3.0)), (complex(1.0), complex(-50.0, 10.0)), (complex(0.0), complex(0.1, 0.1)))
    w("Dfun", m, c, s, t, D.Dfun(m, c, s, t))
    m > 1 && c < 0 && w("Dstat", m, c, s, t, D.Dfun(m, c, s, t; stat = true))
    m == 1 && c < 0 && w("Dstat", m, c, s, t, D.Dfun(m, c, s, t; stat = true))
    Dt, mu = D.Dscaled(m, c, s, t); w("Dscaled", m, c, s, t, Dt, mu)
    G = D.girsanov(m, c, t)
    w("girs", m, c, t, G.th, G.del, G.eta, G.lsh, G.lch, G.dS, G.D, G.A2, G.A1, G.A0)
    m > 1 && w("qstart", m, c, s, t, D.qstart(m, c, s, t), D.qstart_d(m, c, s, t))
    m == 1 && w("qstart", m, c, s, t, D.qstart(m, c, s, t), D.qstart(m, c, s, t))
end
for m in 1:3, c in (-20.0, -9.5, -5.0, 0.0, 2.5, 10.0), st in (false, true)
    (st && c >= 0) && continue
    w("tstar", m, c, st ? 1 : 0, D.tstar(m, c; stat = st))      # (c = -20, -9.5: the first zero of D just below the pole of eta in case DK)
end
for m in 1:3, c in (-5.0, 0.0, 3.0, 10.0), st in (false, true)
    (st && c >= 0) && continue
    tau = D.default_tau(m, c; stat = st)
    ts = [tau + im * tau * sinh(u) for u in 0:0.5:6]
    P, s1, s2, M0 = D.tparts(m, c, ts; stat = st)
    for j in eachindex(ts); w("tparts", m, c, st ? 1 : 0, ts[j], P[j], s1[j], m == 1 && !st ? complex(0.0) : s2[j], m == 1 && !st ? complex(0.0) : M0[j]); end
end
close(out)
