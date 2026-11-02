/* dfl_test_single.c: the single integral at c0 = 0 against the Julia engine's values (test/single_ref.txt) */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double dfl_single_value_c0(int kind, double x, int model, double c, bool stat, const dfl_single_opts *o);
int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "test/single_ref.txt", "r"); if (!f) { printf("no reference file\n"); return 1; }
    dfl_single_opts o; dfl_single_defaults(&o);
    int model, st, n = 0, fails = 0; double c, x, F, fd, wF = 0, wf = 0, tt = 0;
    while (fscanf(f, "%d %lf %d %lf %lf %lf", &model, &c, &st, &x, &F, &fd) == 6) {
        double t0 = dfl_wtime();
        double Fc = dfl_single_value_c0(2, x, model, c, st, &o), fc = dfl_single_value_c0(1, x, model, c, st, &o);
        tt += dfl_wtime() - t0;
        double eF = fabs(Fc - F), ef = fabs(fc - fd) / fmax(fabs(fd), 1e-300), ea = fabs(fc - fd); n++;
        if (eF > wF) wF = eF; if (ef > wf && ea > 1e-11) wf = ef;
        /* the density to 1e-9 relative, or to 1e-11 absolute where it is tiny (the quadrature's absolute noise) */
        if (!(eF < 1e-11) || !(ef < 1e-9 || ea < 1e-11)) { fails++; printf("FAIL m=%d c=%g stat=%d x=%g: F %.15f vs %.15f (%.1e)  f %.15e vs %.15e (%.1e)\n", model, c, st, x, Fc, F, eF, fc, fd, ef); }
    }
    printf("%d points, %d failed, worst |dF| %.1e, worst |df/f| %.1e, %.1f ms per pair of values (%d threads)\n", n, fails, wF, wf, 1e3 * tt / n, dfl_nworkers());
    return fails ? 1 : 0;
}
