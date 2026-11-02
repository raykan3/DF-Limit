/* dfl_test_vmath.c: the vectorizable exp and sincos against libm, and the two versions of the l-rule G of the
 * noncentral kernel against each other, on random arguments; with timings */
#include "dflimit.h"
#include "dfl_vmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
cdouble dfl_ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm);
static double urand(double lo, double hi) { return lo + (hi - lo) * (rand() / (RAND_MAX + 1.0)); }
int main(void)
{
    srand(12345);
    enum { N = 1 << 16 };
    static double x[N], y[N], s[N], c[N];
    double we = 0, ws = 0, wc = 0;
    for (int i = 0; i < N; i++) x[i] = urand(-700, 700) * (i % 2 ? 1 : 0.01);
    dfl_vexp(N, x, y);
    for (int i = 0; i < N; i++) { double r = fabs(y[i] - exp(x[i])) / exp(x[i]); if (r > we) we = r; }
    for (int i = 0; i < N; i++) x[i] = urand(-2000, 2000) * (i % 2 ? 1 : 0.001);
    dfl_vsincos(N, x, s, c);
    for (int i = 0; i < N; i++) { double es = fabs(s[i] - sin(x[i])), ec = fabs(c[i] - cos(x[i])); if (es > ws) ws = es; if (ec > wc) wc = ec; }
    printf("vexp: worst relative error %.2e;  vsincos: worst absolute errors sin %.2e cos %.2e (N = %d)\n", we, ws, wc, N);
    /* G: arguments of the kind met on the line: a = s2 - s1 with Re a > 0, |b| up to 40, Y anywhere with Re Y > 0 mostly */
    int fails = 0; double wg = 0; cdouble worstY = {0};
    for (int i = 0; i < 2000; i++) {
        double aa = exp(urand(log(2.0), log(200.0))), pha = urand(-1.3, 1.3);
        cdouble a = CPX(aa * cos(pha), aa * sin(pha));
        /* |b1| + |b2| <= sqrt(600 Re a): the exponents of the integrand stay below about 600 (lam <= 300) */
        double bmax = sqrt(600 * creal(a));
        double bm = urand(0, bmax / 2), bp = urand(-0.5, 0.5); cdouble b1 = CPX(bm * cos(bp), bm * sin(bp));
        bm = urand(0, bmax / 2); bp = urand(-0.5, 0.5); cdouble b2 = CPX(bm * cos(bp), bm * sin(bp));
        double ym = exp(urand(log(1e-9), log(3.0))), yp = urand(-1.4, 1.4); cdouble Y = CPX(ym * cos(yp), ym * sin(yp));
        cdouble g1 = dfl_ncG(Y, a, b1, b2, 1), g2 = dfl_ncG(Y, a, b1, b2, 0);
        double r = cabs(csub(g1, g2)) / cabs(g1);
        if (r > wg) { wg = r; worstY = Y; }
        if (!(r < 1e-10)) { fails++; if (fails < 5) printf("  G mismatch %.2e at Y %g%+gi a %g%+gi b1 %g%+gi b2 %g%+gi\n", r, creal(Y), cimag(Y), creal(a), cimag(a), creal(b1), cimag(b1), creal(b2), cimag(b2)); }
    }
    printf("G: libm vs vectorized on 2000 random arguments: worst relative difference %.2e (at Y %g%+gi), %d above 1e-10 (the sums cancel where the terms oscillate; the engine values agree to 1e-15)\n", wg, creal(worstY), cimag(worstY), fails);
    cdouble Y = CPX(0.05, 0.02), a = CPX(40, 15), b1 = CPX(20, 0.3), b2 = CPX(2.5, 0.1), acc = {0};
    int reps = 20000;
    double t0 = dfl_wtime(); for (int i = 0; i < reps; i++) acc = cadd(acc, dfl_ncG(Y, a, b1, b2, 1)); double t1 = dfl_wtime();
    for (int i = 0; i < reps; i++) acc = cadd(acc, dfl_ncG(Y, a, b1, b2, 0)); double t2 = dfl_wtime();
    printf("G per call: libm %.2f us, vectorized %.2f us (Nl = %d; checksum %g)\n", 1e6 * (t1 - t0) / reps, 1e6 * (t2 - t1) / reps, DFL_NLNC, creal(acc));
    return fails ? 1 : 0;
}
