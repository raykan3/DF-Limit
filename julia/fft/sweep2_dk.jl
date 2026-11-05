include(joinpath(@__DIR__, "fft_cond.jl"))
xs = [-3.5, -2.0, -1.0, -0.004, 0.0, 0.004, 1.0, 2.5, 4.0]
function report(label, c; kw...)
    T = build(c; kw...)
    e = [cdf_fft(T, x; npan = 10, n = 64) - DFExact.cdf(x, 1; c = c) for x in xs]
    @printf("%-34s check %.1e  max|err| %.2e  (x: %s)\n", label, check_table(T), maximum(abs, e), join([@sprintf("%.0e", abs(v)) for v in e], " "))
end
build(0.0; N = 2^12, na = 8)
report("c=0 default", 0.0)
report("c=-5 na=160", -5.0; na = 160)
report("c=-5 amax=12 na=96", -5.0; amax = 12.0)
report("c=-20 amax=3 na=96", -20.0; amax = 3.0)
report("c=-20 amax=3 na=160", -20.0; amax = 3.0, na = 160)
report("c=1 default", 1.0)
