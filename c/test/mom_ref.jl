# reference moments for the C engine test: c/test/mom_ref.txt (model c c0 g0 stat m1..m4)
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
out = open(ARGS[1], "w")
for (model, c, c0, g0, st) in ((1, 0.0, 0.0, 0.0, false), (2, 0.0, 0.0, 0.0, false), (3, 0.0, 0.0, 0.0, false), (2, -5.0, 0.0, 0.0, false), (3, -10.0, 0.0, 0.0, false),
                                (1, 5.0, 0.0, 0.0, false), (2, 5.0, 0.0, 0.0, false), (2, 10.0, 10.0, 0.0, false), (3, -5.0, -5.0, 0.0, false), (1, 2.0, 2.0, 0.0, false),
                                (2, -5.0, 0.0, 0.0, true), (1, -2.5, 0.0, 0.0, true), (3, -10.0, 1.0, 0.0, true),
                                (1, -5.0, 0.0, 1.5, false), (2, -5.0, 0.0, 1.5, false), (3, -5.0, 0.0, 0.8, false), (2, 2.0, 1.0, 1.0, false))
    m = D.moments(model; c, c0, g0, start = st ? :stationary : :zero, pmax = 4)
    @printf(out, "%d %.17g %.17g %.17g %d %.17g %.17g %.17g %.17g\n", model, c, c0, g0, st ? 1 : 0, m...)
end
close(out)
