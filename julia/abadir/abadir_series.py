# Abadir (1995), eq. (2.11): the limiting distribution function of the Dickey-Fuller t-ratio (case DK, c = 0) for
# z < 0 as a triple series of incomplete gamma functions, in 40-digit arithmetic (mpmath); the density by central
# differences.  Writes abadir_series.json for abadir_check.jl.  Run: python3 abadir_series.py
import json
from mpmath import mp, mpf, sqrt, exp, pi, gamma, gammainc, binomial, factorial
mp.dps = 40

def F211(z):
    x = -sqrt(2) * z
    tot = mpf(0)
    for j in range(0, 60):
        cj = gamma(j + mpf(1) / 2) / (factorial(j) * gamma(mpf(1) / 2))
        y = (2 * j + 1) ** 2 * x ** 2
        sj = mpf(0)
        for l in range(0, j + 1):
            sk = mpf(0)
            for k in range(0, 200):
                term = x ** k / factorial(k) * gammainc(mpf(1) / 4 - mpf(l) / 2 + mpf(k) / 2, y)
                sk += term
                if k > 10 and abs(term) < mpf(10) ** (-45) * abs(sk):
                    break
            sj += binomial(j, l) * (-x) ** l * sk
        tot += cj * sj
        if j > 2 and abs(cj * sj) < mpf(10) ** (-45) * abs(tot):
            break
    return sqrt(x / pi) * exp(-x ** 2 / 4) * tot

out = {}
h = mpf(10) ** (-6)
for z in [-4, -3, -2.5, -2, -1.5, -1, -0.5, -0.2, -0.05]:
    zz = mpf(z)
    F = F211(zz)
    f = (F211(zz + h) - F211(zz - h)) / (2 * h)
    out[str(z)] = {"F": mp.nstr(F, 25), "f": mp.nstr(f, 25)}
    print(z, mp.nstr(F, 18), mp.nstr(f, 16), flush=True)
json.dump(out, open("abadir_series.json", "w"), indent=1)
