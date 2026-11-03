# reference values of the single integral with a local null value c0 != 0 (and the general kernel at c0 = 0): cdf and
# density, for the C engine test c/test/gen_ref.txt
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
out = open(ARGS[1], "w")
P = ((1, 2.0, 2.0, false, [-1.5, 0.0, 1.5, 3.0]), (2, 2.0, 1.0, false, [-3.0, -0.5, 1.3]), (3, -2.5, -5.0, false, [-3.6, -1.1, 0.3]),
     (2, 5.0, -5.0, false, [57.11]), (1, 0.0, -5.0, false, [-1.1, 2.7, 6.5]), (2, 0.0, 3.0, false, [-1.0, -2.5]), (3, -5.0, 2.0, false, [-3.0, -1.0]),
     (2, 10.0, 5.0, false, [0.0, 10.0]), (3, 10.0, 5.0, false, [0.0]), (2, -5.0, 1.0, true, [-2.0, 0.0]), (1, -5.0, -1.0, true, [-2.5, 0.5]),
     (1, 2.0, 2.0, false, [-2.1, -1.9]), (2, 0.0, 0.0, false, [-2.86, 0.5]), (1, 0.0, 0.0, false, [-1.94, 0.5]), (3, -5.0, 0.0, false, [-3.5]),
     # the far roots of X(v) with c0 != 0 (Raymond Kan, November 2026): kept now, where dropping them failed or was wrong
     (2, -1.2, -1.2, true, [9.0, 3.0]), (2, -1.2, 1.2, true, [-9.0]), (2, -1.2, -0.05, true, [3.0, -2.0]), (3, 0.0, -2.0, false, [-9.0, 5.0]))
for (model, c, c0, st, xs) in P
    for x in xs
        gen = c0 == 0
        F = D.cdf(x, model; c, c0, start = st ? :stationary : :zero, gen); f = D.density(x, model; c, c0, start = st ? :stationary : :zero, gen)
        @printf(out, "%d %.17g %.17g %d %.17g %.17g %.17g\n", model, c, c0, st ? 1 : 0, x, F, f)
    end
end
close(out)
