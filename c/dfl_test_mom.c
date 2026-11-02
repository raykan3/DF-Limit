/* dfl_test_mom.c: the moments against the Julia values (test/mom_ref.txt) */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "test/mom_ref.txt", "r"); if (!f) { printf("no reference file\n"); return 1; }
    int model, st, n = 0, fails = 0; double c, c0, g0, mr[4], m[4], e[4], worst = 0, tt = 0; char err[256];
    while (fscanf(f, "%d %lf %lf %lf %d %lf %lf %lf %lf", &model, &c, &c0, &g0, &st, &mr[0], &mr[1], &mr[2], &mr[3]) == 9) {
        double t0 = dfl_wtime();
        int rc = dfl_moments(model, c, c0, g0, st, 4, m, e, err, sizeof err);
        tt += dfl_wtime() - t0;
        n++;
        if (rc) { fails++; printf("FAIL m=%d c=%g c0=%g g0=%g stat=%d: %s\n", model, c, c0, g0, st, err); continue; }
        double wr = 0; for (int j = 0; j < 4; j++) wr = fmax(wr, fabs(m[j] - mr[j]) / fmax(fabs(mr[j]), 1e-300));
        if (wr > worst) worst = wr;
        if (!(wr < 1e-10)) { fails++; printf("FAIL m=%d c=%g c0=%g g0=%g stat=%d: %.12g %.12g %.12g %.12g vs %.12g %.12g %.12g %.12g (rel %.1e)\n", model, c, c0, g0, st, m[0], m[1], m[2], m[3], mr[0], mr[1], mr[2], mr[3], wr); }
    }
    printf("%d sets, %d failed, worst relative difference %.1e, %.0f ms per set\n", n, fails, worst, 1e3 * tt / n);
    return fails ? 1 : 0;
}
