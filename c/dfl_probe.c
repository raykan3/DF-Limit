/* dfl_probe.c: one value of the single integral with its options, for convergence studies (not part of the tests):
 * dfl_probe model c c0 g0 x [nodefac [nlnc [edge_hw [uniform_hw [tau [pfac [N]]]]]]] */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern long DFL_NCG_CALLS, DFL_NCG_SEG, DFL_NCG_RAY;
int main(int argc, char **argv)
{
    if (argc < 6) { printf("usage: dfl_probe model c c0 g0 x [nodefac [nlnc [edge_hw [uniform_hw [tau]]]]]\n"); return 1; }
    int model = atoi(argv[1]); double c = atof(argv[2]), c0 = atof(argv[3]), g0 = atof(argv[4]), x = atof(argv[5]);
    dfl_single_opts o; dfl_single_defaults(&o);
    if (argc > 6) o.nodefac = atoi(argv[6]);
    if (argc > 7) DFL_NLNC = atoi(argv[7]);
    if (argc > 8) o.edge_hw = atof(argv[8]);
    if (argc > 9) o.uniform_hw = atof(argv[9]);
    if (argc > 10) o.tau = atof(argv[10]);
    if (argc > 11) o.pfac = atoi(argv[11]);
    if (argc > 12) o.N = atoi(argv[12]);
    if (getenv("DFL_PROBE_DOUBLE")) {                                  /* the double integral at the same point, for comparison */
        dfl_double_opts d; dfl_double_defaults(&d); dfl_double_info info; char err[256]; double Fd, fd;
        double t0 = dfl_wtime(); dfl_double(2, 1, &x, model, c, c0, g0, false, &d, &Fd, &info, err, sizeof err); double t1 = dfl_wtime();
        dfl_double(1, 1, &x, model, c, c0, g0, false, &d, &fd, &info, err, sizeof err); double t2 = dfl_wtime();
        printf("double integral: F %.16f (%.1f ms)  f %.16e (%.1f ms)  threads %d\n", Fd, 1e3 * (t1 - t0), fd, 1e3 * (t2 - t1), dfl_nworkers());
    }
    DFL_NCG_CALLS = DFL_NCG_SEG = DFL_NCG_RAY = 0;
    double t0 = dfl_wtime(); double F = dfl_single_value_gen(2, x, model, c, c0, g0, false, &o); double t1 = dfl_wtime();
    long ncF = DFL_NCG_CALLS, nsF = DFL_NCG_SEG, nrF = DFL_NCG_RAY; DFL_NCG_CALLS = DFL_NCG_SEG = DFL_NCG_RAY = 0;
    double f = dfl_single_value_gen(1, x, model, c, c0, g0, false, &o); double t2 = dfl_wtime();
    printf("ncG calls: cdf %ld (segments %ld, rays %ld), density %ld (segments %ld, rays %ld)\n", ncF, nsF, nrF, DFL_NCG_CALLS, DFL_NCG_SEG, DFL_NCG_RAY);
    printf("F %.16f (%.1f ms)  f %.16e (%.1f ms)  nodefac %d nlnc %d edge_hw %g uniform_hw %g tau %g pfac %d N %d threads %d\n", F, 1e3 * (t1 - t0), f, 1e3 * (t2 - t1), o.nodefac, DFL_NLNC, o.edge_hw, o.uniform_hw, o.tau, o.pfac, o.N, dfl_nworkers());
    return 0;
}
