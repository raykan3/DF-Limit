# reference values of the single integral at c0 = 0 (cdf and density) for the C engine test: c/test/single_ref.txt
using Printf
include(joinpath(@__DIR__, "..", "..", "julia", "DFExact.jl")); using .DFExact; const D = DFExact
out = open(ARGS[1], "w")
for model in 1:4, (c, st) in ((0.0, false), (-5.0, false), (-20.0, false), (0.5, false), (3.0, false), (10.0, false), (-5.0, true), (-10.0, true))
    model == 4 && (st || c > 2) && continue                 # (model 4, the GLS-detrended trend case: zero start; c <= 2 here)
    xs = model == 1 ? [-3.0, -1.94, -0.72, 0.0, 0.5, 2.0, 8.0] : [-4.0, -2.86, -1.5, -0.5, 0.3, 2.0, 8.0]
    for x in xs
        c > 2 && x < 0 && continue
        F = D.cdf(x, model; c, start = st ? :stationary : :zero); f = D.density(x, model; c, start = st ? :stationary : :zero)
        @printf(out, "%d %.17g %d %.17g %.17g %.17g\n", model, c, st ? 1 : 0, x, F, f)
    end
end
close(out)
