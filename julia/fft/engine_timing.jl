include(joinpath(@__DIR__, "..", "DFExact.jl")); using .DFExact
for x in (-2.0, 0.0, 1.0); DFExact.cdf(x, 1; c = 0.0); DFExact.cdf(x, 1; c = -5.0); DFExact.cdf(x, 2; c = -5.0); ltratio_cdf(x, "DK", 0.0; method = :double); end
xs = collect(-4.0:0.05:4.0)
for (m, c) in ((1, 0.0), (1, -5.0), (2, -5.0))
    t = @elapsed F = [DFExact.cdf(x, m; c = c) for x in xs]
    println("model $m c=$c single: ", length(xs), " values, ", round(1000t / length(xs), digits = 2), " ms each")
end
t = @elapsed F = [ltratio_cdf(x, "DK", 0.0; method = :double) for x in xs[1:40]]
println("DK c=0 double: ", round(1000t / 40, digits = 2), " ms each")
