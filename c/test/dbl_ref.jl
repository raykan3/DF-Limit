# reference values of the double integral (cdf and density) for the C engine test: c/test/dbl_ref.txt
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
out = open(ARGS[1], "w")
P = ((1, 0.0, 0.0, 0.0, false, [-2.5, -0.5, 1.5]), (2, 0.0, 0.0, 0.0, false, [-3.4, -1.6, 0.2]), (3, -5.0, 0.0, 0.0, false, [-3.5, -2.5]),
     (1, 2.0, 2.0, 0.0, false, [-1.5, 0.0, 1.5]), (2, 2.0, 1.0, 0.0, false, [-3.0, -0.5, 1.3]), (3, -2.5, -5.0, 0.0, false, [-3.6, -1.1]), (2, 5.0, -5.0, 0.0, false, [57.11]),
     (1, -5.0, 0.0, 0.0, true, [-2.5, -1.0]), (2, -5.0, 0.0, 0.0, true, [-3.0, -1.2]), (3, -10.0, 0.0, 0.0, true, [-3.5, -2.2]),
     (1, -5.0, 0.0, 1.5, false, [-3.0, -2.0, 0.0]), (2, -5.0, 0.0, 1.5, false, [-3.0, -2.0]), (3, -5.0, 0.0, 1.5, false, [-3.5, -2.5]), (2, 2.0, 0.0, 1.0, false, [-2.0, 0.0]), (3, -10.0, -3.0, 0.8, false, [-3.0, -1.0]), (2, -2.5, 1.0, 0.0, true, [-2.0, 0.0]))
for (model, c, c0, g0, st, xs) in P
    F, _ = D.tr_limdist(xs, model, c; c0, g0, start = st ? :stationary : :zero); f, _ = D.tr_limdist(xs, model, c; c0, g0, start = st ? :stationary : :zero, pdf = true)
    for (k, x) in enumerate(xs); @printf(out, "%d %.17g %.17g %.17g %d %.17g %.17g %.17g\n", model, c, c0, g0, st ? 1 : 0, x, F[k], f[k]); end
end
close(out)
