# reference values of the single integral with a fixed start g0 != 0 (the noncentral kernel): cdf and density, for the C
# engine test c/test/fixed_ref.txt (columns: model c c0 g0 stat x F f)
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
out = open(ARGS[1], "w")
P = ((1, -5.0, 0.0, 1.0, [-2.0, 0.5]), (1, 0.0, 0.0, 1.5, [-1.0]), (1, 2.0, 0.0, 1.0, [1.0, -1.5]), (1, -5.0, -5.0, 1.0, [-2.0]), (1, 2.0, 2.0, 1.0, [-1.0, -2.5]),
     (2, -5.0, 0.0, 1.0, [-2.86, -1.0, 0.5]), (2, -2.0, 0.0, 1.5, [-2.0]), (2, 2.0, 0.0, 1.0, [0.0, -3.0]), (2, -5.0, -5.0, 1.0, [-2.0]), (2, 1.0, 1.0, 1.0, [0.0]),
     (3, -5.0, 0.0, 2.0, [-3.0, 0.0]), (3, 2.0, 0.0, 1.0, [-2.0]), (3, -2.5, -5.0, 1.0, [-1.1]), (3, 0.0, 2.0, 0.5, [-2.3, -1.9]),
     (2, -20.0, 0.0, 1.0, [-4.0, -1.0]), (2, -5.0, 0.0, 3.0, [-3.0]), (3, -10.0, 0.0, 3.0, [-3.0]), (2, 5.0, 0.0, 1.0, [-3.0, 2.0]), (1, 5.0, 0.0, 1.0, [0.0]),
     (2, 3.0, 0.0, 1.0, [0.0]), (2, -20.0, 0.0, 1.5, [-4.0]))
for (model, c, c0, g0, xs) in P
    for x in xs
        F = D.cdf(x, model; c, c0, g0); f = D.density(x, model; c, c0, g0)
        @printf(out, "%d %.17g %.17g %.17g %d %.17g %.17g %.17g\n", model, c, c0, g0, 0, x, F, f)
    end
end
close(out)
