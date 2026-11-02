/* dfl_special.c: the special functions of the engine: the complex error function (Faddeeva package), e^z K_0(z) by
 * series, Taylor cells and the asymptotic series (the cells seeded by the Julia engine into tables.c), Gauss-Legendre
 * rules, spherical Bessel functions for Filon's rule. */
#include "dflimit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void dfl_faddeeva_w(double re, double im, double *ore, double *oim);      /* faddeeva/faddeeva_shim.cpp */
void dfl_faddeeva_erf(double re, double im, double *ore, double *oim);
cdouble dfl_faddw(cdouble z) { double a, b; dfl_faddeeva_w(creal(z), cimag(z), &a, &b); return CPX(a, b); }
cdouble dfl_erf(cdouble z) { double a, b; dfl_faddeeva_erf(creal(z), cimag(z), &a, &b); return CPX(a, b); }

#define EULERGAMMA 0.5772156649015329
#define K0N 12
#define K0NF 14

static cdouble k0_series(cdouble z)          /* |z| < 2: K_0(z) e^z by the power series, 14 terms */
{
    cdouble q4 = cscale(0.25, cmul(z, z)), term = CPX(1, 0), I0 = term, S = CPX(0, 0);
    double H = 0;
    for (int k = 1; k <= 14; k++) {
        term = cscale(1.0 / (k * k), cmul(term, q4)); H += 1.0 / k;
        I0 = cadd(I0, term); S = cadd(S, cscale(H, term));
    }
    cdouble lz = cadd(clog(cscale(0.5, z)), CPX(EULERGAMMA, 0));
    return cmul(cexp(z), csub(S, cmul(lz, I0)));
}
static cdouble k0_asym(cdouble z)            /* |z| >= 25: the asymptotic series, 21 terms */
{
    cdouble w = cdiv(CPX(1, 0), z), acc = CPX(K0A[20], 0);
    for (int k = 19; k >= 0; k--) acc = cadd(cmul(acc, w), CPX(K0A[k], 0));
    return cmul(csqrt(cdiv(CPX(M_PI, 0), cscale(2, z))), acc);
}
static cdouble k0_cells(cdouble z, const double *T, int nn, double h, int ni)   /* Taylor cell of side h; Im z >= 0 */
{
    int i = (int) floor(creal(z) / h), j = (int) floor(cimag(z) / h);
    int cell = i + ni * j;
    cdouble u = csub(z, CPX((i + 0.5) * h, (j + 0.5) * h));
    const double *c = T + 2 * nn * cell;
    cdouble acc = CPX(c[2 * (nn - 1)], c[2 * (nn - 1) + 1]);
    for (int n = nn - 2; n >= 0; n--) acc = cadd(cmul(acc, u), CPX(c[2 * n], c[2 * n + 1]));
    return cmul(cexp(z), acc);
}
static cdouble k0_left(cdouble z);
cdouble dfl_k0x(cdouble z)
{
    double a = cabs(z);
    if (creal(z) < 0) return k0_left(z);           /* (off the tables: a node of a path slightly left of the axis) */
    if (a >= DFL_K0R) return k0_asym(z);
    if (a < 1) return k0_series(z);
    bool lower = cimag(z) < 0;
    cdouble zu = lower ? conj(z) : z, r;
    if (a < 2) r = k0_cells(zu, K0F, K0F_N, DFL_K0HF, 16);
    else r = k0_cells(zu, K0C, K0C_N, DFL_K0H, DFL_K0NI);
    return lower ? conj(r) : r;
}
/* e^z K_0(z) for any z: the table for Re z >= 0; for Re z < 0 (rare, in regions of the double integral where the sign
 * conditions fail) the power series up to |z| = 12 and the asymptotic series beyond */
static cdouble k0_left(cdouble z)
{
    if (cabs(z) > 12) return k0_asym(z);
    cdouble q4 = cscale(0.25, cmul(z, z)), term = CPX(1, 0), I0 = term, S = CPX(0, 0);
    double H = 0;
    for (int k = 1; k <= 60; k++) {
        term = cscale(1.0 / (k * k), cmul(term, q4)); H += 1.0 / k;
        I0 = cadd(I0, term); S = cadd(S, cscale(H, term));
        if (cabs(term) < 1e-18 * cabs(I0)) break;
    }
    cdouble lz = cadd(clog(cscale(0.5, z)), CPX(EULERGAMMA, 0));
    return cmul(cexp(z), csub(S, cmul(lz, I0)));
}
cdouble dfl_k0scaled(cdouble z) { return dfl_k0x(z); }

/* Gauss-Legendre on [0, 1]: Newton on the Legendre polynomial from the Chebyshev initial guesses (16 digits) */
void dfl_gl_rule(int n, double *x, double *w)
{
    for (int i = 0; i < n; i++) {
        double t = cos(M_PI * (i + 0.75) / (n + 0.5)), p0, p1, dp;
        for (int it = 0; it < 100; it++) {
            p0 = 1; p1 = t;
            for (int k = 2; k <= n; k++) { double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = n * (t * p1 - p0) / (t * t - 1);
            double dt = p1 / dp; t -= dt;
            if (fabs(dt) < 1e-16) break;
        }
        p0 = 1; p1 = t;
        for (int k = 2; k <= n; k++) { double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        dp = n * (t * p1 - p0) / (t * t - 1);
        x[n - 1 - i] = (1 + t) / 2;                   /* ascending */
        w[n - 1 - i] = 1.0 / ((1 - t * t) * dp * dp);  /* 2/((1-t^2) P'^2) on [-1,1], halved for [0,1] */
    }
}
/* cached rules by n (n <= 4096); thread-safe through a lock */
#include "dfl_lock.h"
static double *RULEX[4097], *RULEW[4097];
static dfl_lock_t rulelock = DFL_LOCK_INIT;
const double *dfl_rule(int n, const double **w)
{
    if (n < 1 || n > 4096) { *w = NULL; return NULL; }
    dfl_lock(&rulelock);
    if (RULEX[n] == NULL) {
        double *x = malloc(n * sizeof(double)), *ww = malloc(n * sizeof(double));
        dfl_gl_rule(n, x, ww);
        RULEW[n] = ww; RULEX[n] = x;
    }
    dfl_unlock(&rulelock);
    *w = RULEW[n];
    return RULEX[n];
}

/* j_n(x), n = 0..N-1, x >= 0: upward recurrence for x > N, otherwise Miller's downward recurrence from n = N + 40 */
void dfl_sphbessel(double *j, int N, double x)
{
    if (x < 1e-12) { for (int n = 0; n < N; n++) j[n] = 0; j[0] = 1; return; }
    if (x > N) {
        j[0] = sin(x) / x; if (N > 1) j[1] = (sin(x) - x * cos(x)) / (x * x);
        for (int n = 1; n < N - 1; n++) j[n + 1] = (2 * n + 1) / x * j[n] - j[n - 1];
    } else {
        int M = N + 40; double jp = 0, jc = 1e-300;
        double *tmp = malloc((M + 1) * sizeof(double)); tmp[M] = jc;
        for (int n = M - 1; n >= 0; n--) {
            double jn = (2 * n + 3) / x * jc - jp; jp = jc; jc = jn; tmp[n] = jc;
            if (fabs(jc) > 1e200) { for (int k = n; k <= M; k++) tmp[k] /= jc; jp /= jc; jc = 1; }
        }
        double f = (sin(x) / x) / tmp[0];
        for (int n = 0; n < N; n++) j[n] = tmp[n] * f;
        free(tmp);
    }
}
