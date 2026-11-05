/* dfl_api.c: the user-level entry points of the C engine, with the argument conventions of the MATLAB programs:
 * dfl_single (the single integral of Theorem 2 for a vector of x, the general kernel when c0 != 0), dfl_dist (the
 * automatic choice between the single and the double integral at each x, as ltratio_cdf/ltratio_pdf with method
 * 'auto'), dfl_icdf (quantiles by safeguarded Newton from the normal approximation), and the validation shared by all. */
#include "dflimit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static int check_args(int kind, int model, double c, double c0, double g0, bool stat, char *err, size_t errlen)
{
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (model < 1 || model > 3) { snprintf(err, errlen, "model must be 1, 2 or 3 (no deterministic terms, intercept, intercept and trend)"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    if (!isfinite(c0)) { snprintf(err, errlen, "c0 must be a finite real scalar"); return 1; }
    if (!isfinite(g0)) { snprintf(err, errlen, "g0 must be a finite real scalar"); return 1; }
    if (stat && !(c < 0)) { snprintf(err, errlen, "a stationary start (cases SK, SU, ST) requires c < 0"); return 1; }
    if (stat && g0 != 0) { snprintf(err, errlen, "g0 (a fixed start) applies to cases DK, DU, DT only"); return 1; }
    return 0;
}
/* the value at a non-finite x: 0 at -Inf, 1 (cdf) or 0 (density) at Inf, NaN at NaN */
static double nonfinite(int kind, double x) { return isnan(x) ? NAN : x > 0 ? (kind == 2 ? 1.0 : 0.0) : 0.0; }

int dfl_single(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o, double *out, char *err, size_t errlen)
{
    dfl_single_opts d; if (o == NULL) { dfl_single_defaults(&d); o = &d; }
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
        out[k] = (c0 != 0 || g0 != 0 || o->gen) ? dfl_single_value_gen(kind, x[k], model, c, c0, g0, stat, o) : dfl_single_value_c0(kind, x[k], model, c, stat, o);
    }
    return 0;
}

/* the method of method 0 (automatic) at the point x: the single integral at c0 = 0 and with a fixed start; with c0 != 0
 * and a zero start the double integral, except for c0 > 0 at x >= 0, where X(v) has no positive root, the single
 * integral has no breakpoint and takes 0.2-5 ms against 1-60 ms (ten threads).  Elsewhere with c0 != 0 the double
 * integral is the faster one at every c, by 2-10x, measured point by point over c in [-5, 5], c0 in {-5, -1, 1, 3} and
 * x in [-4, 32] in the three cases (with c0 < 0 the saddle part of the single integral costs up to 1 s at x ~ 8 with
 * c0 = -1).  The tails below 1e-3 are recomputed by the double integral in any case (dfl_dist). */
int dfl_automethod(int model, double c, double c0, double g0, double x) { (void) model; (void) c; return (c0 != 0 && g0 == 0 && !(c0 > 0 && x >= 0)) ? 2 : 1; }

int dfl_dist(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, int method,
             const dfl_single_opts *so, const dfl_double_opts *dopt, double *out, int *nstatus, char *err, size_t errlen)
{
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    if (method < 0 || method > 2) { snprintf(err, errlen, "method must be 0 (automatic), 1 (single integral) or 2 (double integral)"); return 1; }
    dfl_single_opts sd; if (so == NULL) { dfl_single_defaults(&sd); so = &sd; }
    dfl_double_opts dd; if (dopt == NULL) { dfl_double_defaults(&dd); dopt = &dd; }
    if (nstatus) *nstatus = 0;
    int nd = 0; double *xd = malloc((n + 1) * sizeof(double)); int *id = malloc((n + 1) * sizeof(int));
    for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
        int m = method == 0 ? dfl_automethod(model, c, c0, g0, x[k]) : method;
        if (m == 2) { xd[nd] = x[k]; id[nd] = k; nd++; }
        else out[k] = (c0 != 0 || g0 != 0 || so->gen) ? dfl_single_value_gen(kind, x[k], model, c, c0, g0, stat, so) : dfl_single_value_c0(kind, x[k], model, c, stat, so);
    }
    /* the tails: the single integral's error is absolute (1e-14 to 1e-21, not reduced by finer rules), so with method 0
     * the values of the single integral below 1e-3 (a density in absolute value, a distribution function in the lower
     * tail) are recomputed by the double integral, which keeps its relative accuracy there (as the MATLAB programs do) */
    int nchosen = nd;                                                   /* the points chosen for the double integral */
    if (method == 0) for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) continue;
        bool sgl = true; for (int j = 0; j < nd; j++) if (id[j] == k) { sgl = false; break; }
        if (sgl && (kind == 1 ? fabs(out[k]) : out[k]) < 1e-3) { xd[nd] = x[k]; id[nd] = k; nd++; }
    }
    int rc = 0;
    if (nd > 0) {
        double *Fd = malloc(nd * sizeof(double)); dfl_double_info *info = malloc(nd * sizeof(dfl_double_info));
        rc = dfl_double(kind, nd, xd, model, c, c0, g0, stat, dopt, Fd, info, err, errlen);
        if (rc == 0) for (int j = 0; j < nd; j++) {
            if (j < nchosen || info[j].status == 0) out[id[j]] = Fd[j];   /* a flagged tail point keeps the single integral's value */
            if (nstatus && info[j].status) (*nstatus)++;
        }
        free(Fd); free(info);
    }
    if (kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;   /* a density below the error of the method */
    free(xd); free(id);
    return rc;
}

/* the inverse error function: Giles' approximation refined by two Newton steps on erf */
static double erfinv(double y)
{
    if (!(y > -1 && y < 1)) return y <= -1 ? -INFINITY : y >= 1 ? INFINITY : NAN;
    double w = -log((1 - y) * (1 + y)), p;
    if (w < 5) { w -= 2.5; p = 2.81022636e-08; p = 3.43273939e-07 + p * w; p = -3.5233877e-06 + p * w; p = -4.39150654e-06 + p * w; p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w; p = -0.00417768164 + p * w; p = 0.246640727 + p * w; p = 1.50140941 + p * w; }
    else { w = sqrt(w) - 3; p = -0.000200214257; p = 0.000100950558 + p * w; p = 0.00134934322 + p * w; p = -0.00367342844 + p * w; p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w; p = 0.00943887047 + p * w; p = 1.00167406 + p * w; p = 2.83297682 + p * w; }
    double x = p * y;
    for (int it = 0; it < 2; it++) x -= (erf(x) - y) / (2 / sqrt(M_PI) * exp(-x * x));
    return x;
}

int dfl_icdf(int n, const double *p, int model, double c, double c0, double g0, bool stat, int method, double tol,
             const dfl_single_opts *so, const dfl_double_opts *dopt, double *q, double *Fq, int *iter, int *conv, char *err, size_t errlen)
{
    if (check_args(2, model, c, c0, g0, stat, err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); return 1; }
    double m[2], me[2];
    if (dfl_moments(model, c, c0, g0, stat, 2, m, me, err, errlen)) return 1;
    double mu = m[0], sd = sqrt(fmax(m[1] - m[0] * m[0], 1e-6));
    if (!(tol > 0)) tol = 1e-10;
#define CDF(xx, out) do { if (dfl_dist(2, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, err, errlen)) return 1; } while (0)
#define PDF(xx, out) do { if (dfl_dist(1, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, err, errlen)) return 1; } while (0)
    for (int k = 0; k < n; k++) {
        double pk = p[k], x = mu + sd * sqrt(2) * erfinv(2 * pk - 1), lo = x - sd, hi = x + sd, F, f;
        for (int it = 0; it < 100; it++) { CDF(lo, F); if (!(F > pk)) break; lo -= 2 * (x - lo); }
        for (int it = 0; it < 100; it++) { CDF(hi, F); if (!(F < pk)) break; hi += 2 * (hi - x); }
        int j, cv = 0;
        for (j = 1; j <= 60; j++) {
            CDF(x, F); PDF(x, f);
            if (F > pk) hi = x; else lo = x;
            double xn = x - (F - pk) / f;
            if (!(lo < xn && xn < hi) || !isfinite(xn)) xn = (lo + hi) / 2;
            if (fabs(xn - x) < tol) { x = xn; cv = 1; break; }
            x = xn;
        }
        q[k] = x; if (iter) iter[k] = j > 60 ? 60 : j; if (conv) conv[k] = cv;
        if (Fq) CDF(x, Fq[k]);
    }
#undef CDF
#undef PDF
    return 0;
}
