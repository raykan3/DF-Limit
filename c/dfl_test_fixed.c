/* dfl_test_fixed.c: the single integral with a fixed start g0 != 0 (the noncentral kernel) against the Julia values */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "test/fixed_ref.txt", "r"); if (!f) { printf("no reference file\n"); return 1; }
    dfl_single_opts o; dfl_single_defaults(&o);
    if (getenv("DFL_NLNC")) DFL_NLNC = atoi(getenv("DFL_NLNC"));          /* the l-rule's node count, for checking */
    int model, st, n = 0, fails = 0; double c, c0, g0, x, F, fd, wF = 0, wf = 0, tt = 0;
    while (fscanf(f, "%d %lf %lf %lf %d %lf %lf %lf", &model, &c, &c0, &g0, &st, &x, &F, &fd) == 8) {
        double t0 = dfl_wtime();
        double Fc = dfl_single_value_gen(2, x, model, c, c0, g0, st, &o), fc = dfl_single_value_gen(1, x, model, c, c0, g0, st, &o);
        tt += dfl_wtime() - t0;
        double eF = fabs(Fc - F), ef = fabs(fc - fd) / fmax(fabs(fd), 1e-300), ea = fabs(fc - fd); n++;
        bool inf_ok = isinf(fd) && isinf(fc);
        if (eF > wF) wF = eF; if (ef > wf && ea > 1e-11 && !inf_ok) wf = ef;
        if (!(eF < 1e-11) || !(ef < 1e-9 || ea < 1e-11 || inf_ok)) { fails++; printf("FAIL m=%d c=%g c0=%g g0=%g stat=%d x=%g: F %.15f vs %.15f (%.1e)  f %.15e vs %.15e (%.1e)\n", model, c, c0, g0, st, x, Fc, F, eF, fc, fd, ef); }
        else if (getenv("DFL_VERBOSE")) printf("ok   m=%d c=%g c0=%g g0=%g x=%g: F %.15f (%.1e) f %.15e (%.1e)\n", model, c, c0, g0, x, Fc, eF, fc, ef);
    }
    printf("%d points, %d failed, worst |dF| %.1e, worst |df/f| %.1e, %.1f ms per pair of values (%d threads)\n", n, fails, wF, wf, 1e3 * tt / n, dfl_nworkers());
    return fails ? 1 : 0;
}
