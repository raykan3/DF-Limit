/* dfl_test_dbl.c: the double integral against the Julia values (test/dbl_ref.txt) */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "test/dbl_ref.txt", "r"); if (!f) { printf("no reference file\n"); return 1; }
    dfl_double_opts o; dfl_double_defaults(&o); char err[256];
    int model, st, n = 0, fails = 0; double c, c0, g0, x, F, fd, wF = 0, wf = 0, tt = 0;
    while (fscanf(f, "%d %lf %lf %lf %d %lf %lf %lf", &model, &c, &c0, &g0, &st, &x, &F, &fd) == 8) {
        double Fc, fc; dfl_double_info iF, ifd;
        double t0 = dfl_wtime();
        int rc = dfl_double(2, 1, &x, model, c, c0, g0, st, &o, &Fc, &iF, err, sizeof err);
        rc |= dfl_double(1, 1, &x, model, c, c0, g0, st, &o, &fc, &ifd, err, sizeof err);
        tt += dfl_wtime() - t0;
        n++;
        if (rc) { fails++; printf("FAIL m=%d c=%g c0=%g g0=%g stat=%d x=%g: %s\n", model, c, c0, g0, st, x, err); continue; }
        double eF = fabs(Fc - F), ef = fabs(fc - fd) / fmax(fabs(fd), 1e-300), ea = fabs(fc - fd);
        if (eF > wF) wF = eF; if (ea > wf) wf = ea;
        if (!(eF < 1e-10) || !(ef < 1e-8 || ea < 1e-10) || iF.status || ifd.status) { fails++; printf("FAIL m=%d c=%g c0=%g g0=%g stat=%d x=%g: F %.15f vs %.15f (%.1e)  f %.15e vs %.15e (%.1e)  status %d %d err %.1e %.1e\n", model, c, c0, g0, st, x, Fc, F, eF, fc, fd, ef, iF.status, ifd.status, iF.err, ifd.err); }
    }
    printf("%d points, %d failed, worst |dF| %.1e, worst |df| %.1e, %.0f ms per pair of values (%d threads)\n", n, fails, wF, wf, 1e3 * tt / n, dfl_nworkers());
    return fails ? 1 : 0;
}
