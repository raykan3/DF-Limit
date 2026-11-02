using JSON, Printf
include("DFExact.jl")
using .DFExact

println("Threads: ", Threads.nthreads())
ref = JSON.parsefile(joinpath(@__DIR__, "python_exact_reference.json"))
DFExact.cdf(-1.0, 1); DFExact.density(-1.0, 1); DFExact.cdf(-1.0, 2); DFExact.density(-1.0, 3)   # compile
println("Against the stored reference values (python_exact_reference.json, computed by this code on 2026-10-31):")
worstF = worstf = 0.0
for r in ref
    model = r["model"]; c = Float64(r["c"]); q = Float64(r["q"])
    tF = @elapsed F = DFExact.cdf(q, model; c = c)
    tf = @elapsed f = DFExact.density(q, model; c = c)
    global worstF = max(worstF, abs(F - r["cdf"])); global worstf = max(worstf, abs(f / r["density"] - 1))
    @printf("M%d c=%g q=%6.3f: F = %.10f (reference %.10f, %5.1f ms, reference run %5.1f ms)  f = %.10f (reference %.10f, %5.1f ms, reference run %5.1f ms)\n",
            model, c, q, F, r["cdf"], 1e3tF, 1e3r["t_cdf"], f, r["density"], 1e3tf, 1e3r["t_density"])
end
@printf("worst |cdf diff| = %.1e, worst density rel diff = %.1e\n", worstF, worstf)

isfile(joinpath(@__DIR__, "python_reference.json")) && begin
q64 = JSON.parsefile(joinpath(@__DIR__, "python_reference.json"))
println("\nAt the 64 exact quantiles of df_quantiles.json: cdf(q) - prob")
worst = 0.0; ttot = 0.0
for r in q64
    t0 = time(); F = DFExact.cdf(Float64(r["q"]), r["model"]; c = Float64(r["c"])); global ttot += time() - t0
    global worst = max(worst, abs(F - r["prob"]))
end
@printf("worst |cdf(q) - prob| = %.1e over %d points, %.0f ms per point on average\n", worst, length(q64), 1e3ttot / length(q64))
end

tq = @elapsed qq = DFExact.quantile(0.05, 2)
@printf("\nM2 5%% quantile by Newton: %.9f in %.2f s\n", qq, tq)
