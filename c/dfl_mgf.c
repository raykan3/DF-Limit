/* dfl_mgf.c: the joint moment generating function: the determinant D(s, t) (closed forms and series), the stationary
 * start, the Girsanov forms, the initial-value term, the roots and the prefactor along a set of nodes, t_*. */
#include "dflimit.h"
#include "dfl_lock.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define Z0 DFL_Z0
static const cdouble ZERO = {0}, ONE = {1};
static inline cdouble zof(double c, cdouble t) { return CPX(c * c - 2 * creal(t), -2 * cimag(t)); }

dfl_pieces_t dfl_pieces(cdouble z)
{
    dfl_pieces_t p;
    cdouble mu = csqrt(z);
    p.C = ccosh(mu);
    if (cabs(z) < 1e-3) {
        cdouble z2 = cmul(z, z), z3 = cmul(z2, z);
        p.S = cadd(cadd(ONE, cscale(1.0 / 6, z)), cadd(cscale(1.0 / 120, z2), cscale(1.0 / 5040, z3)));
        p.Q = csub(csub(CPX(-0.5, 0), cscale(1.0 / 24, z)), cadd(cscale(1.0 / 720, z2), cscale(1.0 / 40320, z3)));
    } else {
        p.S = cdiv(csinh(mu), mu); p.Q = cdiv(csub(ONE, p.C), z);
    }
    return p;
}
cdouble dfl_gstat(int model, cdouble z)
{
    if (cabs(z) < Z0) {
        const double *ser = model == 2 ? G2SER : G3SER;
        cdouble acc = ZERO;
        for (int j = 29; j >= 0; j--) acc = cadd(cmul(acc, z), CPX(ser[j], 0));
        return acc;
    }
    dfl_pieces_t p = dfl_pieces(z);
    if (model == 2) return cdiv(cadd(cscale(2, p.Q), p.S), z);
    cdouble t = cadd(cadd(cscale(-12, csub(ONE, cmul(z, p.Q))), cmul(p.Q, csub(CPX(24, 0), cscale(4, z)))), cmul(p.S, cadd(CPX(24, 0), z)));
    return cdiv(t, cmul(cmul(z, z), z));
}
cdouble dfl_gstat_scaled(int model, cdouble z)
{
    cdouble mu = csqrt(z);
    if (cabs(z) < Z0) return cmul(dfl_gstat(model, z), cexp(cneg(mu)));
    cdouble e2 = cexp(cscale(-2, mu));
    cdouble Ct = cscale(0.5, cadd(ONE, e2)), St = cdiv(cscale(0.5, csub(ONE, e2)), mu), Qt = cdiv(csub(cexp(cneg(mu)), Ct), z);
    if (model == 2) return cdiv(cadd(cscale(2, Qt), St), z);
    cdouble t = cadd(cadd(cscale(-12, Ct), cmul(Qt, csub(CPX(24, 0), cscale(4, z)))), cmul(St, cadd(CPX(24, 0), z)));
    return cdiv(t, cmul(cmul(z, z), z));
}
static cdouble dstat_add(int model, double c, cdouble s, cdouble t)
{
    double ck = model == 2 ? c : c * c * c;
    cdouble f = cadd(cadd(t, cscale(0.5, cmul(s, s))), cscale(c, s));
    return cscale(ck * exp(c), cmul(f, dfl_gstat(model, csub(CPX(c * c, 0), cscale(2, t)))));
}
/* the series branch: D/e^c = sum_j z^j sum_i lam^i sum_k c^k T[j][i][k] */
static cdouble dseries(int model, double c, cdouble lam, cdouble z)
{
    const double *T = model == 2 ? D1SER : D2SER;
    int nj = model == 2 ? D1SER_NJ : D2SER_NJ, ni = model == 2 ? D1SER_NI : D2SER_NI, nk = model == 2 ? D1SER_NK : D2SER_NK;
    double cp[8]; cp[0] = 1; for (int k = 1; k < nk; k++) cp[k] = cp[k - 1] * c;
    cdouble acc = ZERO;
    for (int j = nj - 1; j >= 0; j--) {
        cdouble cj = ZERO;
        for (int i = ni - 1; i >= 0; i--) {
            double a = 0; for (int k = 0; k < nk; k++) a += T[(j * ni + i) * nk + k] * cp[k];
            cj = cadd(cmul(cj, lam), CPX(a, 0));
        }
        acc = cadd(cmul(acc, z), cj);
    }
    return acc;
}
/* the closed forms of models 2 and 3 with the pieces (C or Ct, Q, S) given: common to Dfun and Dscaled */
static cdouble dclosed(int model, double c, cdouble lam, cdouble lx, cdouble z, cdouble Cc, cdouble Q, cdouble S)
{
    double e = exp(c);
    if (model == 2) {
        cdouble A = csub(csub(cscale(2 * c * c, lam), cscale(4 * c, lx)), cscale(2, cmul(lam, lam)));
        cdouble B = csub(csub(cadd(CPX(-c * c * c, 0), cscale(c * c, lam)), cscale(2, lx)), cmul(lam, lam));
        return cscale(e, cdiv(cadd(cadd(cscale(c * c, Cc), cmul(A, Q)), cmul(B, S)), z));
    }
    double c2 = c * c, c3 = c2 * c, c4 = c3 * c, c5 = c4 * c, c6 = c5 * c, c7 = c6 * c;
    cdouble l2 = cmul(lam, lam), x2 = cmul(lx, lx);
    cdouble P1 = cadd(cadd(CPX(c6, 0), cscale(-12 * c4, lam)), cadd(cscale(2 * (-c4 + 12 * c3 - 12 * c2), lx), cscale(48 * (c2 - 3 * c + 3), l2)));
    cdouble P2 = cadd(cadd(cscale(4 * (-c6 + 6 * c4), lam), cscale(16 * (c4 - 3 * c3 - 3 * c2 + 18 * c - 18), l2)),
                      cadd(cscale(8, cmul(lx, cadd(cadd(CPX(c5 - 3 * c4, 0), cscale(c4, lam)), cscale(-4 * (c2 - 3 * c + 3), l2)))),
                           cscale(16 * (-c3 + 3 * c2 - 6 * c + 6), x2)));
    cdouble P3 = cadd(cadd(CPX(-c7, 0), cscale(c6 + 24 * c4, lam)), cadd(cscale(4 * (-c4 + 3 * c3 - 27 * c2 + 72 * c - 72), l2),
                      cadd(cscale(2, cmul(lx, cadd(cadd(CPX(c5 - 4 * c4 - 12 * c3 + 12 * c2, 0), cscale(-c4, lam)), cscale(4 * (c2 - 3 * c + 3), l2)))),
                           cscale(16 * (c2 - 3 * c + 3), x2))));
    cdouble num = cadd(cadd(cmul(Cc, P1), cmul(Q, P2)), cmul(S, P3));
    return cscale(e, cdiv(num, cmul(cmul(z, z), z)));
}
cdouble dfl_Dfun(int model, double c, cdouble s, cdouble t, bool stat)
{
    cdouble lam = cneg(s), z = zof(c, t);
    dfl_pieces_t p = dfl_pieces(z);
    if (stat) {
        if (model == 1) {
            cdouble f = cdiv(cadd(cscale(2, t), cmul(s, s)), CPX(2 * c, 0));
            return cscale(exp(c), cadd(csub(p.C, cscale(c, p.S)), cmul(p.S, f)));
        }
        return cadd(dfl_Dfun(model, c, s, t, false), dstat_add(model, c, s, t));
    }
    if (model == 1) return csub(p.C, cmul(cadd(CPX(c, 0), s), p.S));
    if (cabs(z) < Z0) return cscale(exp(c), dseries(model, c, lam, z));
    return dclosed(model, c, lam, t, z, csub(ONE, cmul(z, p.Q)), p.Q, p.S);
}
cdouble dfl_Dscaled(int model, double c, cdouble s, cdouble t, cdouble *mu)
{
    cdouble z = zof(c, t), m = csqrt(z);
    *mu = m;
    if (cabs(z) < Z0) return cmul(dfl_Dfun(model, c, s, t, false), cexp(cneg(m)));
    cdouble e2 = cexp(cscale(-2, m));
    cdouble Ct = cscale(0.5, cadd(ONE, e2)), St = cdiv(cscale(0.5, csub(ONE, e2)), m), Qt = cdiv(csub(cexp(cneg(m)), Ct), z);
    if (model == 1) return csub(Ct, cmul(cadd(CPX(c, 0), s), St));
    return dclosed(model, c, cneg(s), t, z, Ct, Qt, St);
}

/* ---- the Girsanov forms ---- */
static cdouble polyq(const double *cf, int n, int from, cdouble q)   /* sum_{k=from}^{n-1} cf[k] q^{k-from} */
{
    cdouble acc = ZERO;
    for (int k = n - 1; k >= from; k--) acc = cadd(cmul(acc, q), CPX(cf[k], 0));
    return acc;
}
cdouble dfl_eta1(double c, cdouble t)
{
    cdouble mu = csqrt(zof(c, t));
    if (cabs(mu) < 1e-4) return cadd(csub(ONE, CPX(c, 0)), cscale(1.0 / 3, cmul(mu, mu)));
    cdouble e2 = cexp(cscale(-2, mu));
    return cadd(cdiv(cscale(-2, t), cadd(mu, CPX(c, 0))), cdiv(cscale(2, cmul(mu, e2)), csub(ONE, e2)));
}
static void det_adj3(const cdouble *M, cdouble *A, cdouble *det)
{
#define E(i, j) M[3 * (i) + (j)]
    A[0] = csub(cmul(E(1,1), E(2,2)), cmul(E(1,2), E(2,1))); A[1] = cneg(csub(cmul(E(0,1), E(2,2)), cmul(E(0,2), E(2,1)))); A[2] = csub(cmul(E(0,1), E(1,2)), cmul(E(0,2), E(1,1)));
    A[3] = cneg(csub(cmul(E(1,0), E(2,2)), cmul(E(1,2), E(2,0)))); A[4] = csub(cmul(E(0,0), E(2,2)), cmul(E(0,2), E(2,0))); A[5] = cneg(csub(cmul(E(0,0), E(1,2)), cmul(E(0,2), E(1,0))));
    A[6] = csub(cmul(E(1,0), E(2,1)), cmul(E(1,1), E(2,0))); A[7] = cneg(csub(cmul(E(0,0), E(2,1)), cmul(E(0,1), E(2,0)))); A[8] = csub(cmul(E(0,0), E(1,1)), cmul(E(0,1), E(1,0)));
    *det = cadd(cadd(cmul(E(0,0), A[0]), cmul(E(0,1), A[3])), cmul(E(0,2), A[6]));
#undef E
}
void dfl_girsanov(int model, double c, cdouble t, dfl_girs_t *G)
{
    cdouble q = zof(c, t), th = csqrt(q);
    cdouble del = c > 0 ? cdiv(cscale(-2, t), cadd(th, CPX(c, 0))) : csub(th, CPX(c, 0));
    cdouble e2 = cexp(cscale(-2, th)), em = cexp(cneg(th));
    cdouble g1, g2, g3, g4, g5, t1, lch, lsh, ep, N11;
    if (cabs(th) < 1) {
        g1 = polyq(GT, 45, 0, q); g2 = cneg(polyq(GE, 46, 1, q)); g3 = cneg(polyq(GT, 45, 1, q)); g4 = polyq(GE, 46, 2, q); g5 = polyq(GT, 45, 2, q);
        t1 = csub(ONE, cmul(th, g1)); lch = clog(ccosh(th));
        cdouble r = ONE, tm = ONE, ch = ONE;
        for (int k = 1; k <= 14; k++) { tm = cscale(1.0 / ((2 * k) * (2 * k + 1)), cmul(tm, q)); r = cadd(r, tm); }
        tm = ONE; for (int k = 1; k <= 14; k++) { tm = cscale(1.0 / ((2 * k - 1) * (2 * k)), cmul(tm, q)); ch = cadd(ch, tm); }
        lsh = clog(r); ep = csub(cdiv(ch, r), th);
        N11 = csub(ONE, cscale(c, g1));
    } else {
        cdouble tn = cdiv(csub(ONE, e2), cadd(ONE, e2)), sc = cdiv(cscale(2, em), cadd(ONE, e2));
        cdouble th2 = cmul(th, th), th3 = cmul(th2, th), th4 = cmul(th3, th), th5 = cmul(th4, th);
        g1 = cdiv(tn, th); g2 = cdiv(csub(ONE, sc), th2); g3 = cdiv(csub(th, tn), th3);
        g4 = cdiv(cadd(csub(th2, CPX(2, 0)), cscale(2, sc)), cscale(2, th4));
        g5 = cdiv(cadd(csub(cscale(1.0 / 3, th3), th), tn), th5);
        t1 = cdiv(cscale(2, e2), cadd(ONE, e2)); lch = cadd(th, clog(cscale(0.5, cadd(ONE, e2))));
        ep = cdiv(cscale(2, cmul(th, e2)), csub(ONE, e2)); lsh = csub(cadd(th, clog(cscale(0.5, csub(ONE, e2)))), clog(th));
        N11 = cdiv(cadd(del, cscale(c, t1)), th);
    }
    G->th = th; G->del = del; G->eta = cadd(del, ep); G->lsh = lsh; G->lch = lch;
    if (model == 1) { G->dim = 1; G->dS = G->D = G->A2 = G->A1 = G->A0 = ZERO; return; }
    if (model == 2) {
        G->dim = 2;
        cdouble *S = G->Sg, *N = G->N;
        S[0] = g1; S[1] = g2; S[2] = g2; S[3] = g3;
        /* N = I - Qd Sg, Qd = diag(c, -2t) */
        N[0] = N11; N[1] = cscale(-c, g2); N[2] = cscale(2, cmul(t, g2)); N[3] = cadd(ONE, cscale(2, cmul(t, g3)));
        G->dS = csub(cmul(g1, g3), cmul(g2, g2));
        G->D = cadd(cmul(N11, cadd(ONE, cscale(2, cmul(t, g3)))), cscale(2 * c, cmul(t, cmul(g2, g2))));
        G->A2 = cdiv(G->D, G->dS); G->A1 = cadd(cdiv(csub(g1, cscale(2, g2)), G->dS), cscale(2, t)); G->A0 = CPX(-1, 0);
        return;
    }
    G->dim = 3;
    cdouble *S = G->Sg, *N = G->N;
    S[0] = g1; S[1] = g2; S[2] = g3; S[3] = g2; S[4] = g3; S[5] = g4; S[6] = g3; S[7] = g4; S[8] = g5;
    cdouble Qd[9] = {CPX(c, 0), ZERO, ZERO, ZERO, cscale(-8, t), cscale(12, t), ZERO, cscale(12, t), cscale(-24, t)};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        cdouble acc = ZERO;
        for (int k = 0; k < 3; k++) acc = cadd(acc, cmul(Qd[3 * i + k], S[3 * k + j]));
        N[3 * i + j] = csub(i == j ? ONE : ZERO, acc);
    }
    N[0] = N11;
    cdouble aN[9], aS[9], D, dS;
    det_adj3(N, aN, &D); det_adj3(S, aS, &dS);
    G->D = D; G->dS = dS; G->A2 = cdiv(D, dS);
    cdouble Pp[9];
    for (int i = 0; i < 9; i++) Pp[i] = csub(cdiv(aS[i], dS), Qd[i]);
    Pp[0] = cdiv(cadd(cadd(cmul(N[0], aS[0]), cmul(N[1], aS[3])), cmul(N[2], aS[6])), dS);
    static const double K[9] = {1, 2, -6, 2, -12, 12, -6, 12, 0}, aK[9] = {-144, -72, -48, -72, -36, -24, -48, -24, -16};
    cdouble aP[9], dum; det_adj3(Pp, aP, &dum);
    cdouble A1 = ZERO, A0 = ZERO;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { A1 = cadd(A1, cscale(K[3 * i + j], aP[3 * j + i])); A0 = cadd(A0, cscale(aK[3 * i + j], Pp[3 * j + i])); }
    G->A1 = A1; G->A0 = A0;
}
/* the initial-value term Q(u, v), Girsanov form (cases DU, DT: det(N - u K Sg)) and determinant form */
static cdouble detNuK(const dfl_girs_t *G, int model, cdouble u)
{
    static const double KDU[4] = {1, -1, -1, 0}, KDT[9] = {1, 2, -6, 2, -12, 12, -6, 12, 0};
    int d = G->dim; const double *K = model == 2 ? KDU : KDT;
    cdouble M[9];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
        cdouble ks = ZERO;
        for (int k = 0; k < d; k++) ks = cadd(ks, cscale(K[d * i + k], G->Sg[d * k + j]));
        M[d * i + j] = csub(G->N[d * i + j], cmul(u, ks));
    }
    if (d == 2) return csub(cmul(M[0], M[3]), cmul(M[1], M[2]));
    cdouble A[9], det; det_adj3(M, A, &det); return det;
}
cdouble dfl_girs_detu(int model, double c, cdouble u, cdouble v, dfl_girs_t *G)
{
    dfl_girsanov(model, c, v, G);
    return detNuK(G, model, u);
}
cdouble dfl_qstart(int model, double c, cdouble u, cdouble v)
{
    dfl_girs_t G; dfl_girsanov(model, c, v, &G);
    if (model == 1) {
        cdouble S = cexp(G.lsh), S2 = cmul(S, S);
        cdouble t3 = cdiv(u, cscale(2, cmul(cmul(S2, G.eta), csub(G.eta, u))));
        return cadd(csub(cdiv(v, G.eta), cscale(0.5, u)), t3);
    }
    cdouble d = detNuK(&G, model, u);
    double ck = model == 2 ? c * c : c * c * c * c;
    cdouble f = cadd(cadd(v, cscale(0.5, cmul(u, u))), cscale(c, u));
    cdouble g = dfl_gstat_scaled(model, zof(c, v));
    return cdiv(cscale(2 * ck, cmul(f, g)), cmul(cadd(ONE, cexp(cscale(-2, G.th))), d));
}
cdouble dfl_qstart_d(int model, double c, cdouble u, cdouble v)
{
    cdouble mu, Dt = dfl_Dscaled(model, c, u, v, &mu);
    double ck = model == 2 ? c * c : c * c * c * c;
    cdouble f = cadd(cadd(v, cscale(0.5, cmul(u, u))), cscale(c, u));
    return cdiv(cscale(ck * exp(c), cmul(f, dfl_gstat_scaled(model, zof(c, v)))), Dt);
}

/* ---- t_*, cached ---- */
#define TSMAX 256
static struct { int model; double c; bool stat; double ts; } TSCACHE[TSMAX]; static int ntscache = 0;
static dfl_lock_t tslock = DFL_LOCK_INIT;
static double tstar_g(int model, double c, bool stat, double t)
{
    /* case DK: the sign of D = eta S from the cancellation-free eta and the sign of S = sinh(mu)/mu (sin(th)/th for 2t > c^2);
     * eta alone changes sign at the poles of coth as well, and for c < 0 the first zero of D lies just below the pole
     * (c^2 + pi^2)/2, inside a stretch that the 5% steps of the bracketing can skip (c = -9.5: 61.6 for 49.2) */
    if (model == 1 && !stat) { double q = c * c - 2 * t, sg = q >= 0 ? 1.0 : (sin(sqrt(-q)) >= 0 ? 1.0 : -1.0); return creal(dfl_eta1(c, CPX(t, 0))) * sg; }
    if (!stat && c > DFL_CGIRS) { dfl_girs_t G; dfl_girsanov(model, c, CPX(t, 0), &G); return creal(G.D); }
    return creal(dfl_Dfun(model, c, ZERO, CPX(t, 0), stat));
}
double dfl_tstar(int model, double c, bool stat)
{
    dfl_lock(&tslock);
    for (int k = 0; k < ntscache; k++) if (TSCACHE[k].model == model && TSCACHE[k].c == c && TSCACHE[k].stat == stat) { double r = TSCACHE[k].ts; dfl_unlock(&tslock); return r; }
    dfl_unlock(&tslock);
    double a = 1e-6;
    while (tstar_g(model, c, stat, a) <= 0 && a > 1e-300) a /= 1e3;
    if (tstar_g(model, c, stat, a) <= 0) return NAN;
    double b = a;
    while (tstar_g(model, c, stat, b) > 0) { a = b; b *= 1.05; if (b > 1e4) return NAN; }
    for (int it = 0; it < 100; it++) {
        double m = (a + b) / 2;
        if (tstar_g(model, c, stat, m) > 0) a = m; else b = m;
        if (b - a < 1e-15 * b) break;
    }
    double ts = (a + b) / 2;
    dfl_lock(&tslock);
    if (ntscache < TSMAX) { TSCACHE[ntscache].model = model; TSCACHE[ntscache].c = c; TSCACHE[ntscache].stat = stat; TSCACHE[ntscache].ts = ts; ntscache++; }
    dfl_unlock(&tslock);
    return ts;
}

/* ---- P(t), s1, s2, M0 along the nodes t (t[0] the real point), branches by continuity ---- */
static void unwrap(double *ph, int n) { for (int j = 1; j < n; j++) ph[j] -= 2 * M_PI * round((ph[j] - ph[j - 1]) / (2 * M_PI)); }
void dfl_tparts(int model, double c, int nt, const cdouble *t, bool stat, cdouble *P, cdouble *s1, cdouble *s2, cdouble *M0)
{
    double *ph = malloc(nt * sizeof(double));
    if (model == 1 && !stat) {
        for (int j = 0; j < nt; j++) {
            dfl_pieces_t p = dfl_pieces(zof(c, t[j]));
            s1[j] = c > 0 ? dfl_eta1(c, t[j]) : cdiv(csub(p.C, cscale(c, p.S)), p.S);
            ph[j] = carg(p.S); M0[j] = p.S;                 /* (M0 holds S for the moment) */
        }
        unwrap(ph, nt);
        for (int j = 0; j < nt; j++) { P[j] = cscale(exp(-c / 2) * pow(cabs(M0[j]), -0.5), cexp(CPX(0, -ph[j] / 2))); M0[j] = ZERO; s2[j] = ZERO; }
        free(ph); return;
    }
    cdouble *sa = malloc(nt * sizeof(cdouble)), *sb = malloc(nt * sizeof(cdouble)), *ma2 = malloc(nt * sizeof(cdouble)), *M0lch = malloc(nt * sizeof(cdouble));
    double *ph2 = malloc(nt * sizeof(double));
    bool girs = !stat && c > DFL_CGIRS;
    for (int j = 0; j < nt; j++) {
        if (girs) {
            dfl_girs_t G; dfl_girsanov(model, c, t[j], &G);
            cdouble disc = csqrt(csub(cmul(G.A1, G.A1), cscale(4, cmul(G.A0, G.A2))));
            double sw = creal(cmul(conj(G.A1), disc)) < 0 ? -1 : 1;
            cdouble rb = cdiv(cadd(G.A1, cscale(sw, disc)), cscale(2, G.A0)), rs = cdiv(G.A2, cmul(G.A0, rb));
            sa[j] = rb; sb[j] = rs;
            ma2[j] = G.D; ph[j] = carg(G.D); M0lch[j] = G.lch;           /* (ma2 holds D, ph its argument) */
        } else {
            cdouble s0 = ZERO, sp = CPX(-1, 0), sm = CPX(1, 0);
            cdouble d0 = dfl_Dfun(model, c, s0, t[j], stat), dp = dfl_Dfun(model, c, sp, t[j], stat), dm = dfl_Dfun(model, c, sm, t[j], stat);
            cdouble a2 = csub(cscale(0.5, cadd(dp, dm)), d0), a1 = cscale(0.5, csub(dp, dm)), a0 = d0;
            cdouble disc = csqrt(csub(cmul(a1, a1), cscale(4, cmul(a2, a0))));
            double sw = creal(cmul(conj(a1), disc)) < 0 ? -1 : 1;
            cdouble rb = cdiv(csub(cneg(a1), cscale(sw, disc)), cscale(2, a2)), rs = cdiv(a0, cmul(a2, rb));
            sa[j] = cneg(rb); sb[j] = cneg(rs); ma2[j] = cneg(a2);
        }
    }
    if (creal(sa[0]) < creal(sb[0])) { s1[0] = sa[0]; s2[0] = sb[0]; } else { s1[0] = sb[0]; s2[0] = sa[0]; }
    for (int j = 1; j < nt; j++) {
        if (cabs(csub(sa[j], s1[j - 1])) + cabs(csub(sb[j], s2[j - 1])) <= cabs(csub(sb[j], s1[j - 1])) + cabs(csub(sa[j], s2[j - 1]))) { s1[j] = sa[j]; s2[j] = sb[j]; }
        else { s1[j] = sb[j]; s2[j] = sa[j]; }
    }
    if (girs) {
        unwrap(ph, nt);                                              /* arg D continued along the nodes */
        for (int j = 0; j < nt; j++) ph2[j] = carg(cneg(cmul(s1[j], s2[j])));
        unwrap(ph2, nt);
        for (int j = 0; j < nt; j++) {
            cdouble lm = CPX(log(cabs(cmul(s1[j], s2[j]))), ph2[j]), lD = CPX(log(cabs(ma2[j])), ph[j]);
            M0[j] = cexp(cscale(-0.5, lm));                          /* (-s1 s2)^{-1/2} */
            P[j] = cexp(cadd(CPX(-c / 2, 0), cadd(cscale(-0.5, cadd(M0lch[j], lD)), cscale(0.5, lm))));
        }
    } else {
        for (int j = 0; j < nt; j++) ph[j] = carg(ma2[j]);
        unwrap(ph, nt);
        for (int j = 0; j < nt; j++) P[j] = cscale(pow(cabs(ma2[j]), -0.5), cexp(CPX(0, -ph[j] / 2)));
        for (int j = 0; j < nt; j++) ph2[j] = carg(cneg(cmul(s1[j], s2[j])));
        unwrap(ph2, nt);
        for (int j = 0; j < nt; j++) M0[j] = cscale(pow(cabs(cmul(s1[j], s2[j])), -0.5), cexp(CPX(0, -ph2[j] / 2)));
    }
    free(sa); free(sb); free(ma2); free(M0lch); free(ph); free(ph2);
}
