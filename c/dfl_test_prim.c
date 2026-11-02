/* dfl_test_prim.c: the primitives against the reference values written by the Julia engine (prim_ref.jl) */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int fails = 0, n = 0; static double worst = 0;
static void chk(const char *tag, cdouble got, cdouble want, double tol, const char *ctx)
{
    double e = cabs(csub(got, want)) / fmax(cabs(want), 1e-300); n++;
    if (!(e <= tol)) { fails++; printf("FAIL %s %s: got %.17g%+.17gi want %.17g%+.17gi rel %.1e\n", tag, ctx, creal(got), cimag(got), creal(want), cimag(want), e); }
    if (e > worst) worst = e;
}
static cdouble rd(FILE *f) { double a, b; if (fscanf(f, "%lf %lf", &a, &b) != 2) exit(2); return CPX(a, b); }
int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "prim_ref.txt", "r"); if (!f) { printf("no reference file\n"); return 1; }
    char tag[16];
    while (fscanf(f, "%15s", tag) == 1) {
        char ctx[128] = "";
        if (!strcmp(tag, "k0")) { cdouble z = rd(f), v = rd(f); snprintf(ctx, sizeof ctx, "z=%g%+gi", creal(z), cimag(z)); chk(tag, dfl_k0scaled(z), v, creal(z) >= 0 ? 2e-14 : 1e-7, ctx); }
        else if (!strcmp(tag, "erf")) { cdouble z = rd(f), v = rd(f); chk(tag, dfl_erf(z), v, 1e-13, ""); }
        else if (!strcmp(tag, "faddw")) { cdouble z = rd(f), v = rd(f); chk(tag, dfl_faddw(z), v, 1e-13, ""); }
        else if (!strcmp(tag, "Dfun") || !strcmp(tag, "Dstat")) { int m; double c; fscanf(f, "%d %lf", &m, &c); cdouble s = rd(f), t = rd(f), v = rd(f);
            snprintf(ctx, sizeof ctx, "m=%d c=%g s=%g%+gi t=%g%+gi", m, c, creal(s), cimag(s), creal(t), cimag(t)); chk(tag, dfl_Dfun(m, c, s, t, tag[1] == 's'), v, 1e-12, ctx); }
        else if (!strcmp(tag, "Dscaled")) { int m; double c; fscanf(f, "%d %lf", &m, &c); cdouble s = rd(f), t = rd(f), v = rd(f), mu = rd(f), mu2;
            snprintf(ctx, sizeof ctx, "m=%d c=%g t=%g%+gi", m, c, creal(t), cimag(t)); chk(tag, dfl_Dscaled(m, c, s, t, &mu2), v, 1e-12, ctx); chk("mu", mu2, mu, 1e-14, ctx); }
        else if (!strcmp(tag, "girs")) { int m; double c; fscanf(f, "%d %lf", &m, &c); cdouble t = rd(f); dfl_girs_t G; dfl_girsanov(m, c, t, &G);
            snprintf(ctx, sizeof ctx, "m=%d c=%g t=%g%+gi", m, c, creal(t), cimag(t));
            cdouble th = rd(f), del = rd(f), eta = rd(f), lsh = rd(f), lch = rd(f), dS = rd(f), Dd = rd(f), A2 = rd(f), A1 = rd(f), A0 = rd(f);
            chk("th", G.th, th, 1e-14, ctx); chk("del", G.del, del, 1e-13, ctx); chk("eta", G.eta, eta, 1e-12, ctx); chk("lsh", G.lsh, lsh, 1e-12, ctx); chk("lch", G.lch, lch, 1e-12, ctx);
            if (m > 1) { chk("dS", G.dS, dS, 1e-12, ctx); chk("D", G.D, Dd, 1e-11, ctx); chk("A2", G.A2, A2, 1e-11, ctx); chk("A0", G.A0, A0, 1e-11, ctx);
                double sc = fmax(cabs(A1), fmax(cabs(A2), cabs(A0)));          /* A1 vanishes at c = 0: relative to the scale of the A's */
                chk("A1", cadd(cscale(1 / sc, G.A1), CPX(1, 0)), cadd(cscale(1 / sc, A1), CPX(1, 0)), 1e-11, ctx); } }
        else if (!strcmp(tag, "qstart")) { int m; double c; fscanf(f, "%d %lf", &m, &c); cdouble s = rd(f), t = rd(f), q1 = rd(f), q2 = rd(f);
            snprintf(ctx, sizeof ctx, "m=%d c=%g s=%g%+gi t=%g%+gi", m, c, creal(s), cimag(s), creal(t), cimag(t)); chk(tag, dfl_qstart(m, c, s, t), q1, 1e-11, ctx); if (m > 1) chk("qstart_d", dfl_qstart_d(m, c, s, t), q2, 1e-11, ctx); }
        else if (!strcmp(tag, "tstar")) { int m, st; double c, v; fscanf(f, "%d %lf %d %lf", &m, &c, &st, &v); snprintf(ctx, sizeof ctx, "m=%d c=%g stat=%d", m, c, st); chk(tag, CPX(dfl_tstar(m, c, st), 0), CPX(v, 0), 1e-12, ctx); }
        else if (!strcmp(tag, "tparts")) { int m, st; double c; fscanf(f, "%d %lf %d", &m, &c, &st); cdouble t = rd(f), P = rd(f), s1 = rd(f), s2 = rd(f), M0 = rd(f);
            /* the reference lines of one grid come in order: collect them, then compare (the branches need the grid) */
            static cdouble tt[64], PP[64], S1[64], S2[64], MM[64]; static int k = 0, pm = -1, pst = -1; static double pc = NAN;
            if (k > 0 && (m != pm || c != pc || st != pst || creal(t) != creal(tt[0]) || cimag(t) <= cimag(tt[k - 1]) && k > 1 && cimag(t) == 0)) {
                cdouble Pc[64], s1c[64], s2c[64], M0c[64]; dfl_tparts(pm, pc, k, tt, pst, Pc, s1c, s2c, M0c);
                for (int j = 0; j < k; j++) { snprintf(ctx, sizeof ctx, "m=%d c=%g stat=%d t=%g%+gi", pm, pc, pst, creal(tt[j]), cimag(tt[j]));
                    chk("P", Pc[j], PP[j], 1e-11, ctx); chk("s1", s1c[j], S1[j], 1e-11, ctx); if (!(pm == 1 && !pst)) { chk("s2", s2c[j], S2[j], 1e-11, ctx); chk("M0", M0c[j], MM[j], 1e-11, ctx); } }
                k = 0;
            }
            pm = m; pc = c; pst = st; tt[k] = t; PP[k] = P; S1[k] = s1; S2[k] = s2; MM[k] = M0; k++;
            if (k == 13) { cdouble Pc[64], s1c[64], s2c[64], M0c[64]; dfl_tparts(pm, pc, k, tt, pst, Pc, s1c, s2c, M0c);
                for (int j = 0; j < k; j++) { snprintf(ctx, sizeof ctx, "m=%d c=%g stat=%d t=%g%+gi", pm, pc, pst, creal(tt[j]), cimag(tt[j]));
                    chk("P", Pc[j], PP[j], 1e-11, ctx); chk("s1", s1c[j], S1[j], 1e-11, ctx); if (!(pm == 1 && !pst)) { chk("s2", s2c[j], S2[j], 1e-11, ctx); chk("M0", M0c[j], MM[j], 1e-11, ctx); } }
                k = 0; }
        }
        else { printf("unknown tag %s\n", tag); return 1; }
    }
    printf("%d checks, %d failed, worst relative difference %.1e\n", n, fails, worst);
    return fails ? 1 : 0;
}
