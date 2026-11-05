include(joinpath(@__DIR__, "fft_cond.jl"))
c = 0.0
T = build(c; N = 2^12, na = 8); T = build(c)
xs = collect(range(-4.0, 4.0, length = 1001))
for pass in 1:2
    tx = @elapsed Ff = [cdf_fft(T, x; npan = 10, n = 64) for x in xs]
    te = @elapsed Fe = [DFExact.cdf(x, 1; c = c) for x in xs]
    println("pass $pass: 1001 points  fft ", round(1e6 * tx / 1001, digits = 1), " us/x (total ", round(tx, digits = 3), " s)   engine ", round(1e3 * te / 1001, digits = 3), " ms/x (total ", round(te, digits = 2), " s)   max|diff| ", maximum(abs, Ff .- Fe))
end
tb = @elapsed build(c); println("build (10 threads): ", round(tb, digits = 3), " s")
