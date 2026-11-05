include(joinpath(@__DIR__, "fft_cond.jl"))
using Statistics
c = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.0
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2^16
na = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 96
dw = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.25
sigma = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
println("threads ", Threads.nthreads(), "  c=$c N=$N na=$na dw=$dw sigma=$sigma  vmax=", round(2pi / dw, digits = 2))
T = build(c; N = 2^12, na = 8)                       # warm-up
tb = @elapsed T = build(c; sigma, N, dw, na)
println("build: ", round(tb, digits = 3), " s;  table check (int H dv vs P(R<=a)): ", check_table(T))
xs = collect(-4.0:0.25:4.0)
cdf_fft(T, -1.0); cdf_fft(T, 1.0)
tx = @elapsed Ff = [cdf_fft(T, x) for x in xs]
DFExact.cdf(-1.0, 1; c); 
te = @elapsed Fe = [DFExact.cdf(x, 1; c) for x in xs]
err = Ff .- Fe
println("per x: fft ", round(1e6 * tx / length(xs), digits = 1), " us,  engine ", round(1e3 * te / length(xs), digits = 2), " ms")
println("max |err| ", maximum(abs, err), "  at x = ", xs[argmax(abs.(err))])
for (x, fe, e) in zip(xs, Fe, err)
    @printf("%6.2f  %.12f  %+.2e\n", x, fe, e)
end
# node-count and panel checks at three points
for x in (-2.0, 0.5, 3.0)
    @printf("x=%5.2f  n48/6pan %.3e   n64/10pan %.3e\n", x, cdf_fft(T, x) - DFExact.cdf(x, 1; c), cdf_fft(T, x; npan = 10, n = 64) - DFExact.cdf(x, 1; c))
end
