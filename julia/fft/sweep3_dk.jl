include(joinpath(@__DIR__, "fft_cond.jl"))
xs = [-3.5, -2.0, -1.0, 0.0, 1.0, 2.5, 4.0]
function report(label, c; npan = 10, n = 64, kw...)
    T = build(c; kw...)
    e = [cdf_fft(T, x; npan, n) - DFExact.cdf(x, 1; c = c) for x in xs]
    @printf("%-40s check %.1e  max|err| %.2e\n", label, check_table(T), maximum(abs, e))
end
build(0.0; N = 2^12, na = 8)
report("c=-20 amax=3 N=2^17 dw=0.25 (wmax 16384)", -20.0; amax = 3.0, N = 2^17)
report("c=0 na=64", 0.0; na = 64)
report("c=0 na=96 npan=6 n=48", 0.0; npan = 6, n = 48)
T = build(0.0); xs2 = collect(range(-4.0, 4.0, length = 1001))
cdf_fft(T, 1.0; npan = 6, n = 48)
t = @elapsed [cdf_fft(T, x; npan = 6, n = 48) for x in xs2]; println("per x with 6x48 nodes: ", round(1e6 * t / 1001, digits = 1), " us")
T = build(0.0; na = 64); cdf_fft(T, 1.0; npan = 6, n = 48)
t = @elapsed [cdf_fft(T, x; npan = 6, n = 48) for x in xs2]; println("per x with na=64, 6x48 nodes: ", round(1e6 * t / 1001, digits = 1), " us")
