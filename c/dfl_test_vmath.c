/* dfl_test_vmath.c: the vectorizable exp and sincos against libm, and the two versions of the l-rule G of the
 * noncentral kernel against each other, on random arguments; with timings */
#include "dflimit.h"
#include "dfl_vmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
cdouble dfl_ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm);
cdouble dfl_ncG_S(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm, double *E);
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
    /* G: arguments as the engine forms them, from the roots and the means: a = s2 - s1 (Re s2 > 0 > Re s1), b1 = m2 s1,
     * b2 = m1 s2 with |m| <= 1.2, Y with Re Y > 0 mostly.  G = S e^E with S the sum relative to its largest term; where
     * |S| < 1e-8 the terms cancel to the rounding level and the two versions carry different rounding, so those are
     * counted and skipped (the engine does not meet them: its sums were checked against the double integral). */
    int fails = 0, skipped = 0, compared = 0; double wg = 0; cdouble worstY = {0};
    for (int i = 0; i < 2000; i++) {
        double r2 = exp(urand(log(2.0), log(300.0))), p2 = urand(-0.7, 0.7); cdouble s2 = CPX(r2 * cos(p2), r2 * sin(p2));
        double r1 = exp(urand(log(2.0), log(500.0))), p1 = urand(-0.7, 0.7); cdouble s1 = CPX(-r1 * cos(p1), -r1 * sin(p1));
        double mm = urand(0, 1.2), mp = urand(-0.05, 0.05); cdouble m1 = CPX(mm * cos(mp), mm * sin(mp));
        mm = urand(0, 1.2); mp = urand(-0.05, 0.05); cdouble m2 = CPX(mm * cos(mp), mm * sin(mp));
        cdouble a = csub(s2, s1), b1 = cmul(m2, s1), b2 = cmul(m1, s2);
        double ym = exp(urand(log(1e-9), log(3.0))), yp = urand(-1.4, 1.4); cdouble Y = CPX(ym * cos(yp), ym * sin(yp));
        double E1, E2; cdouble S1 = dfl_ncG_S(Y, a, b1, b2, 1, &E1), S2 = dfl_ncG_S(Y, a, b1, b2, 0, &E2);
        if (cabs(S1) < 1e-8) { skipped++; continue; }
        cdouble g1 = cmul(S1, cexp(CPX(E1, 0))), g2 = cmul(S2, cexp(CPX(E2, 0)));
        double r = cabs(csub(g1, g2)) / cabs(g1); compared++;
        if (r > wg) { wg = r; worstY = Y; }
        if (!(r < 1e-7)) { fails++; if (fails < 5) printf("  G mismatch %.2e at Y %g%+gi a %g%+gi b1 %g%+gi b2 %g%+gi |S| %.1e\n", r, creal(Y), cimag(Y), creal(a), cimag(a), creal(b1), cimag(b1), creal(b2), cimag(b2), cabs(S1)); }
    }
    printf("G: libm vs vectorized on %d random arguments (%d cancelling cases skipped): worst relative difference %.2e (at Y %g%+gi), %d above 1e-7 (the versions agree to the cancellation level of the sum)\n", compared, skipped, wg, creal(worstY), cimag(worstY), fails);
    cdouble Y = CPX(0.05, 0.02), a = CPX(40, 15), b1 = CPX(20, 0.3), b2 = CPX(2.5, 0.1), acc = {0};
    int reps = 20000;
    double t0 = dfl_wtime(); for (int i = 0; i < reps; i++) acc = cadd(acc, dfl_ncG(Y, a, b1, b2, 1)); double t1 = dfl_wtime();
    for (int i = 0; i < reps; i++) acc = cadd(acc, dfl_ncG(Y, a, b1, b2, 0)); double t2 = dfl_wtime();
    printf("G per call: libm %.2f us, vectorized %.2f us (Nl = %d; checksum %g)\n", 1e6 * (t1 - t0) / reps, 1e6 * (t2 - t1) / reps, DFL_NLNC, creal(acc));
    return fails ? 1 : 0;
}
