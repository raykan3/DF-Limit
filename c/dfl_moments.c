/* dfl_moments.c: the moments E tau^p for any c, c0 and start (Proposition 3): Cauchy integrals in u of the moment
 * generating function at real tilts, the Girsanov forms with a zero start, g0^2 Q(u, v) for a fixed start, D_S for
 * the stationary start; the integral over s by the trapezoidal rule in log s.  A port of moments_err of DFExact.jl. */
#include "dflimit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const cdouble ZERO = {0}, ONE = {1};
cdouble dfl_girs_detu(int model, double c, cdouble u, cdouble v, dfl_girs_t *G);

/* the logarithm of Z continued along a closed circle starting at a point where Z > 0; returns whether it closes */
static bool circlog(int n, const cdouble *Z, cdouble *lz)
{
    bool ok = true;
    double ph = cimag(clog(Z[0])); if (fabs(ph) >= 1e-8) ok = false;
    lz[0] = CPX(log(cabs(Z[0])), ph);
    for (int k = 1; k < n; k++) {
        double d = carg(cexp(CPX(0, cimag(clog(Z[k])) - cimag(clog(Z[k - 1])))));
        if (fabs(d) >= M_PI / 3) ok = false;
        ph += d; lz[k] = CPX(log(cabs(Z[k])), ph);
    }
    if (fabs(ph + carg(cdiv(Z[0], Z[n - 1])) - cimag(lz[0])) >= 1e-6) ok = false;
    return ok;
}
/* log E exp(u R - u/2 + v V), zero start, by the Girsanov forms on a circle U with arguments V */
static bool zmgf(int model, double c, int n, const cdouble *U, const cdouble *V, cdouble *L)
{
    cdouble *arg = malloc(n * sizeof(cdouble)), *base = malloc(n * sizeof(cdouble)), *lz = malloc(n * sizeof(cdouble));
    for (int k = 0; k < n; k++) {
        dfl_girs_t G;
        if (model == 1) { dfl_girsanov(1, c, V[k], &G); arg[k] = csub(G.eta, U[k]); base[k] = csub(CPX(-c / 2, 0), cadd(cscale(0.5, U[k]), cscale(0.5, G.lsh))); }
        else { arg[k] = dfl_girs_detu(model, c, U[k], V[k], &G); base[k] = csub(CPX(-c / 2, 0), cadd(cscale(0.5, U[k]), cscale(0.5, G.lch))); }
    }
    bool ok = circlog(n, arg, lz);
    for (int k = 0; k < n; k++) L[k] = csub(base[k], cscale(0.5, lz[k]));
    free(arg); free(base); free(lz);
    return ok;
}
static double zarg(int model, double c, double u, double v)
{
    dfl_girs_t G;
    if (model == 1) { dfl_girsanov(1, c, CPX(v, 0), &G); return creal(csub(G.eta, CPX(u, 0))); }
    cdouble d = dfl_girs_detu(model, c, CPX(u, 0), CPX(v, 0), &G);
    return creal(d) * (creal(cexp(G.lch)) < 0 ? -1 : 1);
}
static double zrad(int model, double c, double c0, double s, double r0)
{
    double R = INFINITY;
    for (int sgi = 0; sgi < 2; sgi++) {
        double sg = sgi == 0 ? 1 : -1, lo = 0, hi = INFINITY, u = sg * r0;
        for (int k = 0; k <= 80; k++) {
            double d = zarg(model, c, u, -s - c0 * u);
            if (!(d > 0) || !isfinite(d)) { hi = fabs(u); break; }
            lo = fabs(u); u *= 2;
        }
        if (isfinite(hi)) for (int it = 0; it < 30; it++) { double m = (lo + hi) / 2, d = zarg(model, c, sg * m, -s - c0 * sg * m); if (d > 0 && isfinite(d)) lo = m; else hi = m; }
        R = fmin(R, lo);
    }
    return R > 0 ? R : r0 / 0.4;
}

int dfl_moments(int model, double c, double c0, double g0, bool stat, int pmax, double *m, double *errest, char *err, size_t errlen)
{
    if (stat && g0 != 0) { snprintf(err, errlen, "a fixed start g0 applies to the zero-start cases only"); return 1; }
    if (model == 4) { snprintf(err, errlen, "the moments are not available for the GLS-detrended case (model 4)"); return 1; }
    int kern = (model == 1 && !stat) ? 1 : 2;
    double smax = dfl_tstar(model, c, stat);
    double h = 0.05, tlo = fmin(-60.0, -2 * c - 40), thi = 8.0, rfac = 0.4; int Nu = 64;
    int ns = (int) floor((thi - tlo) / h + 1e-9) + 1;
    bool zs = !stat;
    double extra = (model == 1 && !stat) ? -c / 2 : 0.0;
    double *acc = calloc((size_t) pmax * ns, sizeof(double));
    cdouble *w = malloc(Nu * sizeof(cdouble)), *U = malloc(Nu * sizeof(cdouble)), *V = malloc(Nu * sizeof(cdouble)), *L = malloc(Nu * sizeof(cdouble)), *dv = malloc(Nu * sizeof(cdouble)), *lz = malloc(Nu * sizeof(cdouble));
    for (int k = 0; k < Nu; k++) w[k] = cexp(CPX(0, 2 * M_PI * k / Nu));
    for (int i = 0; i < ns; i++) {
        double s = exp(tlo + i * h);
        cdouble t = CPX(-s, 0), P, s1, s2, M0;
        dfl_tparts(model, c, 1, &t, stat, &P, &s1, &s2, &M0);
        double r = rfac * (kern == 1 ? cabs(s1) : fmin(cabs(s1), cabs(s2)));
        if (c0 != 0) r = fmin(r, 0.4 * (s + smax) / fabs(c0));
        if (zs && c > 0 && g0 == 0) r = fmax(r, rfac * zrad(model, c, c0, s, r));
        if (stat || g0 != 0) r = fmin(r, 0.25);
        bool ok = false;
        for (int tries = 0; tries < 8 && !ok; tries++) {
            for (int k = 0; k < Nu; k++) { U[k] = cscale(r, w[k]); V[k] = csub(CPX(-s, 0), cscale(c0, U[k])); }
            if (zs) {
                ok = zmgf(model, c, Nu, U, V, L);
                if (g0 != 0) for (int k = 0; k < Nu; k++) L[k] = cadd(L[k], cscale(g0 * g0, dfl_qstart(model, c, U[k], V[k])));
            } else {
                for (int k = 0; k < Nu; k++) dv[k] = dfl_Dfun(model, c, U[k], V[k], stat);
                ok = circlog(Nu, dv, lz);
                for (int k = 0; k < Nu; k++) L[k] = csub(cadd(cscale(-0.5, U[k]), CPX(extra, 0)), cscale(0.5, lz[k]));
            }
            if (!ok) r /= 2;
        }
        if (!ok) { snprintf(err, errlen, "moments: no circle in u free of zeros of D was found at s = %g", s); free(acc); free(w); free(U); free(V); free(L); free(dv); free(lz); return 1; }
        for (int j = 1; j <= pmax; j++) {
            double fac = 1; for (int q = 2; q <= j; q++) fac *= q;
            cdouble sum = ZERO;
            for (int k = 0; k < Nu; k++) sum = cadd(sum, cmul(cexp(L[k]), cpow(w[k], CPX(-j, 0))));
            double Mj = fac / (Nu * pow(r, j)) * creal(sum);
            acc[(j - 1) * ns + i] = pow(s, j / 2.0) * Mj;
        }
    }
    for (int j = 1; j <= pmax; j++) {
        double Isum = 0, Isum2 = 0;
        for (int i = 0; i < ns; i++) { Isum += acc[(j - 1) * ns + i]; if (i % 2 == 0) Isum2 += acc[(j - 1) * ns + i]; }
        double g = tgamma(j / 2.0);
        m[j - 1] = h * Isum / g;
        if (errest) errest[j - 1] = fabs(h * Isum - 2 * h * Isum2) / g;
    }
    free(acc); free(w); free(U); free(V); free(L); free(dv); free(lz);
    return 0;
}
