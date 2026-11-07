/* dfl_gen.c: the single integral with a local null value c0 != 0 or a fixed start g0 != 0: the kernel runs along the
 * parabola X(v) = c0 v^2 + x v + x0 (x0 = 1/2, or (1 + g0^2)/2 in case DK with a fixed start), with one Filon part of
 * the outer integral per positive root of X (and the vertex as a pseudo-edge for c0 > 0), the segments between
 * breakpoints on three-piece steepest-descent paths.  With a fixed start the density of R under the tilt is the
 * noncentral one (nckernel, phialg): elementary in case DK, a one-dimensional l-integral (ncG) in cases DU, DT.  A port
 * of breakpts / inner_gen / segint / pieces_nodes / phase_nodes / descent / outer_gen / nckernel / ncG of DFExact.jl
 * (Raymond Kan's extension of the MATLAB port). */
#include "dflimit.h"
#include "dfl_vmath.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const cdouble ZERO = {0}, ONE = {1};
#define EXPO DFL_EXPO
#define KAPPA DFL_KAPPA
#define WMAX DFL_WMAX
extern int DFL_NODEFAC;
extern double DFL_FILONW;
int DFL_NLNC = 40;                              /* (40 agrees with 384 to 1e-14; 32 loses 3e-13) */
/* shared with dfl_single.c */
cdouble dfl_phi_pos(int kern, cdouble X, cdouble s1, cdouble s2);
cdouble dfl_phi_neg(int kern, cdouble X, cdouble s1, cdouble s2);
cdouble dfl_k0x(cdouble z);
typedef cdouble (*cfn_t)(cdouble x, const void *ctx);
cdouble dfl_gaussexp(cfn_t A, const void *Actx, cdouble t, double q, cdouble sd, double lA, int force, int p);
bool dfl_ray_ok(cdouble sd, cdouble e, double q, double at);
cdouble dfl_psi_half(int kern, cdouble s1, cdouble s2, cdouble M0);
cdouble dfl_filon_panel(const cdouble *G, double a, double b, double y, int N);
double dfl_default_tau(int model, double c, bool stat);

/* ---- the breakpoints ---- */
typedef struct {
    double q, c0, x0, lead;
    cdouble rr[2]; int nr;                     /* the roots of X(v) */
    double b[2], y[2], ell[2]; int kindb[2]; int ne; bool negl[2];
    double ell0; int sgn[3]; bool dbl; double vsad;
} breaks_t;
static cdouble xat(const breaks_t *R, cdouble dv, double b)
{
    cdouble X = CPX(R->lead, 0);
    for (int k = 0; k < R->nr; k++) X = cmul(X, cadd(dv, csub(CPX(b, 0), R->rr[k])));
    return X;
}
static double dXv(const breaks_t *R, double v) { return 2 * R->c0 * v + R->q; }
static void breakpts(double q, double c0, double tau, double x0, breaks_t *R)
{
    memset(R, 0, sizeof *R);
    R->q = q; R->c0 = c0; R->x0 = x0;
    double r[2]; int nrr = 0;
    if (c0 == 0) { R->rr[0] = CPX(-x0 / q, 0); R->nr = 1; R->lead = q; r[0] = -x0 / q; nrr = 1; }
    else {
        double d = q * q - 4 * c0 * x0;
        cdouble sq = d >= 0 ? CPX(sqrt(d), 0) : CPX(0, sqrt(-d));
        R->rr[0] = cdiv(csub(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->rr[1] = cdiv(cadd(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->nr = 2; R->lead = c0;
        if (d >= 0) { r[0] = creal(R->rr[0]); r[1] = creal(R->rr[1]); if (r[0] > r[1]) { double t = r[0]; r[0] = r[1]; r[1] = t; } nrr = 2; }
    }
    double b[2]; int nb = 0;
    for (int k = 0; k < nrr; k++) if (r[k] > 0) b[nb++] = r[k];
    bool noroot = nb == 0;
    /* a far root is dropped for c0 = 0 only: with c0 != 0 X changes sign at the root and the segment beyond it has the other
     * form with the other exponential factor; dropping it ran the near side's form out to infinity (Raymond Kan, Nov 2026) */
    if (c0 == 0) { int nb2 = 0; for (int k = 0; k < nb; k++) if (3 * tau * b[k] * b[k] / 4 <= 40) b[nb2++] = b[k]; nb = nb2; }
    int kindb[2] = {1, 1}; double ell[2] = {INFINITY, INFINITY};
    if (nb == 2 && b[1] - b[0] <= 1e-9 * b[1]) { b[0] = (b[0] + b[1]) / 2; nb = 1; R->dbl = true; }
    else if (nb == 2) ell[0] = ell[1] = b[1] - b[0];
    if (c0 > 0 && q < 0 && noroot) { b[0] = -q / (2 * c0); nb = 1; kindb[0] = 0; ell[0] = sqrt(fmax(x0 - q * q / (4 * c0), 0) / c0); }
    for (int k = 0; k < nb; k++) {
        for (int m = 0; m < R->nr; m++) { double dr = cabs(csub(CPX(b[k], 0), R->rr[m])); if (dr > 1e-9 * fmax(b[k], 1)) ell[k] = fmin(ell[k], dr); }
        ell[k] = fmax(ell[k], 1e-8 * fmax(b[k], 1));
    }
    R->ell0 = INFINITY; for (int m = 0; m < R->nr; m++) R->ell0 = fmin(R->ell0, cabs(R->rr[m]));
    R->ne = nb;
    for (int k = 0; k < nb; k++) { R->b[k] = b[k]; R->y[k] = b[k] * b[k]; R->kindb[k] = kindb[k]; R->ell[k] = ell[k]; }
    double ed[4] = {0, nb > 0 ? b[0] : INFINITY, nb > 1 ? b[1] : INFINITY, INFINITY};
    for (int j = 0; j <= nb; j++) {
        double vm = isfinite(ed[j + 1]) ? (ed[j] + ed[j + 1]) / 2 : ed[j] + 1;
        double sv = creal(xat(R, CPX(vm, 0), 0)); R->sgn[j] = sv < 0 ? -1 : 1;
    }
    for (int k = 0; k < nb; k++) { double mid = (ed[k] + ed[k + 1]) / 2; R->negl[k] = 3 * tau * mid * mid / 4 > 40; }
    R->vsad = NAN;
}

/* ---- the noncentral kernel of a fixed start (see DFExact.jl, "the fixed start g0 != 0 through the single integral") ----
 * Case DK: R + g0^2/2 = J(1)^2/2 with J(1) ~ N(m, 1/s1); phi_t(X~) = pref e^{-s1 m^2/2} e^{-s1 X~} cosh(m s1 sqrt(2 X~))/sqrt(pi X~).
 * Cases DU, DT: R = (X1^2 - X2^2)/2, X1 ~ N(m1, 1/s2), X2 ~ N(m2, -1/s1);
 *   phi_t(X) = pref (2/pi) e^{(s1 m2^2 - s2 m1^2)/2} e^{-s2 X} G(X; a, m2 s1, m1 s2)   (X > 0), with -X, e^{-s1 X} and
 *   the b's swapped for X < 0; G(Y; a, b1, b2) = int_0^inf e^{-a l^2/2} cosh(b1 l) cosh(b2 sqrt(2Y + l^2)) (2Y + l^2)^{-1/2} dl.
 * pref = exp(g0^2 Q(0,t)) multiplies the transform. */
typedef struct { bool nc; int kern; cdouble m1, m2, pref; double g0; } nck_t;
static void nckernel(int model, double c, cdouble t, double g0, cdouble s1, cdouble s2, nck_t *K)
{
    K->nc = true; K->g0 = g0;
    if (model == 1) {
        dfl_girs_t G; dfl_girsanov(1, c, t, &G);
        K->kern = 1; K->m1 = cscale(g0, cdiv(cexp(cneg(G.lsh)), G.eta)); K->m2 = ZERO; K->pref = cexp(cscale(g0 * g0, cdiv(t, G.eta)));
        return;
    }
    cdouble Q0, m1s, m2s;
    dfl_qpieces(model, c, t, g0, cdiv(ONE, s2), cneg(cdiv(ONE, s1)), c > DFL_CGIRS, &Q0, &m1s, &m2s);
    K->kern = 2; K->m1 = csqrt(m1s); K->m2 = csqrt(m2s); K->pref = cexp(Q0);
}
/* G(Y; a, b1, b2): l = al sinh T, al = sqrt|2Y|, Gauss-Legendre in T up to where the Gaussian, against the growth of the
 * cosh factors, is below e^{-41.4}; sqrt(2Y + l^2) = al sqrt(2Y/al^2 + sinh^2 T) (principal branch) */
long DFL_NCG_CALLS = 0, DFL_NCG_SEG = 0, DFL_NCG_RAY = 0;   /* counters, for profiling (DFL_NCG_SEG, DFL_NCG_RAY: the nodes of the segments and of the rays) */
int DFL_NCG_LIBM = 0;                           /* 1: the libm version of ncG (checking) */
/* the libm version: complex exp, cosh, sqrt at each node */
/* the nodes and weights of the l-rule for int_0^lm e^{-a l^2/2 + bb l + i(...)} (...) dl with the edge scale al (the
 * factor (2Y + l^2)^{-1/2} varies on it): the Gaussian times the growth peaks at l* = bb/Re a with the width sigma =
 * (Re a)^{-1/2}.  When the peak is separated from the edge scale (l* - 8 sigma > 2 al, large noncentrality), l = al sinh T
 * with Nl Gauss-Legendre nodes in T on [0, l* - 8 sigma] and Gauss-Legendre in l on [l* - 8 sigma, l* + 8 sigma], with the
 * node count from the phase change (ai = Im a, bi = |Im b_1| + |Im b_2|) across it; otherwise l = al sinh T on [0, lm] with
 * the node count raised where the peak is narrow in T.  Returns the count, at most NLMAX. */
#define NLMAX 256
static int clampn(double n, int lo) { int k = (int) ceil(n / 8) * 8; return k < lo ? lo : k > NLMAX ? NLMAX : k; }
static int lrule(double al, double lm, double bb, double ra, double ai, double bi, int Nl, double *ls, double *ws)
{
    double lstar = bb / ra, sig = 1 / sqrt(ra);
    if (Nl > NLMAX / 2) Nl = NLMAX / 2;
    if (lstar - 8 * sig > 2 * al) {
        double l1 = lstar - 8 * sig, lhi = fmin(lm, lstar + 8 * sig);
        double dphi = fabs(ai) * (lhi * lhi - l1 * l1) / 2 + bi * (lhi - l1);
        int nB = clampn(1.2 * dphi + 24, Nl); if (nB > NLMAX - Nl) nB = NLMAX - Nl;
        const double *wl; const double *zl = dfl_rule(Nl, &wl); const double *wB; const double *zB = dfl_rule(nB, &wB);
        double tm = asinh(l1 / al);
        for (int k = 0; k < Nl; k++) { double T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; }
        for (int k = 0; k < nB; k++) { ls[Nl + k] = l1 + (lhi - l1) * zB[k]; ws[Nl + k] = (lhi - l1) * wB[k]; }
        return Nl + nB;
    }
    double tm = asinh(lm / al), dT = sig / sqrt(al * al + lstar * lstar);
    int n = clampn(3 * tm / dT, Nl);
    const double *wl; const double *zl = dfl_rule(n, &wl);
    for (int k = 0; k < n; k++) { double T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; }
    return n;
}
static cdouble ncG_libm(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
    int Nl = DFL_NLNC;
    double ra = fmax(creal(a), 1e-300), bb = cabs(b1) + cabs(b2);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    double al = fmax(sqrt(cabs(cscale(2, Y))), 1e-8 * lm);
    double ls[NLMAX], ws[NLMAX]; int n = lrule(al, lm, fabs(creal(b1)) + fabs(creal(b2)), ra, cimag(a), fabs(cimag(b1)) + fabs(cimag(b2)), Nl, ls, ws);
    cdouble Y2 = cscale(2 / (al * al), Y), acc = ZERO;
    cdouble Es[NLMAX], rs[NLMAX], fs[NLMAX]; double Eref = -INFINITY;
    for (int k = 0; k < n; k++) {
        double l = ls[k], sh = l / al;
        cdouble r = cscale(al, csqrt(cadd(Y2, CPX(sh * sh, 0))));
        cdouble z1 = cscale(l, b1), z2 = cmul(b2, r);
        if (creal(z1) < 0) z1 = cneg(z1);
        if (creal(z2) < 0) z2 = cneg(z2);
        /* cosh z = e^{z} (1 + e^{-2z})/2 for Re z >= 0: one exponential of the combined exponent */
        Es[k] = cadd(cadd(cscale(-l * l / 2, a), z1), z2); rs[k] = r;
        fs[k] = cscale(0.25, cmul(cadd(ONE, cexp(cscale(-2, z1))), cadd(ONE, cexp(cscale(-2, z2)))));
        if (creal(Es[k]) > Eref) Eref = creal(Es[k]);
    }
    /* the terms as e^{E_k - Eref}: G = S e^{Eref} with S bounded (the exponents reach 2 lam at large noncentrality) */
    for (int k = 0; k < n; k++) acc = cadd(acc, cscale(ws[k], cdiv(cmul(cexp(csub(Es[k], CPX(Eref, 0))), fs[k]), rs[k])));
    *E = Eref;
    return acc;
}
/* the vectorizable version: the same sum in real arithmetic on arrays of length Nl, exp and sincos from dfl_vmath.h, so
 * that the compiler vectorizes the loops (NEON, SSE2; AVX2+FMA on Linux x86 through target_clones) */
#if defined(__linux__) && defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#define DFL_CLONES __attribute__((target_clones("avx2,fma", "default")))
#else
#define DFL_CLONES
#endif
static inline void csqrt_arr(int n, const double *p, const double *q, double *re, double *im)
{
    for (int i = 0; i < n; i++) {
        double pp = p[i], qq = q[i], mag = sqrt(pp * pp + qq * qq), u = sqrt((mag + fabs(pp)) / 2);
        u = u > 1e-300 ? u : 1e-300;
        double aq = fabs(qq), sq = qq < 0 ? -1.0 : 1.0;
        re[i] = pp >= 0 ? u : aq / (2 * u);
        im[i] = pp >= 0 ? qq / (2 * u) : sq * u;
    }
}
/* the factor (1 + e^{-2 z})/2 of cosh z = e^{z} (1 + e^{-2 z})/2, Re z >= 0, from e = e^{-2 Re z}, s = sin(-2 Im z), c = cos(-2 Im z) */
static inline void coshfac_arr(int n, const double *e, const double *s, const double *c, double *re, double *im)
{
    for (int i = 0; i < n; i++) { re[i] = (1 + e[i] * c[i]) / 2; im[i] = e[i] * s[i] / 2; }
}
DFL_CLONES static cdouble ncG_vec(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
    int Nl = DFL_NLNC;
    double ar = creal(a), ai = cimag(a), ra = fmax(ar, 1e-300), bb = cabs(b1) + cabs(b2);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    double al = fmax(sqrt(cabs(cscale(2, Y))), 1e-8 * lm);
    double Y2r = 2 * creal(Y) / (al * al), Y2i = 2 * cimag(Y) / (al * al);
    double b1r = creal(b1), b1i = cimag(b1), b2r = creal(b2), b2i = cimag(b2);
    enum { NM = NLMAX };
    double l[NM], wk[NM], l2[NM], sh[NM];
    double pr[NM], qi[NM], rr[NM], ri[NM];                   /* r = al sqrt(Y2 + sh^2) */
    double x1[NM], y1[NM], x2[NM], y2[NM];                   /* z1 = b1 l, z2 = b2 r, with Re >= 0 */
    double xe[NM], ye[NM], ee[NM], se[NM], ce[NM];           /* the combined exponent E = -a l^2/2 + z1 + z2 */
    double m1[NM], n1[NM], e1[NM], s1[NM], c1[NM], f1r[NM], f1i[NM];   /* (1 + e^{-2 z1})/2 */
    double m2[NM], n2[NM], e2[NM], s2[NM], c2[NM], f2r[NM], f2i[NM];   /* (1 + e^{-2 z2})/2 */
    double accr[NM], acci[NM];
    Nl = lrule(al, lm, fabs(b1r) + fabs(b2r), ra, ai, fabs(b1i) + fabs(b2i), Nl, l, wk);     /* now the count of nodes */
    for (int k = 0; k < Nl; k++) { sh[k] = l[k] / al; l2[k] = l[k] * l[k]; }
    for (int k = 0; k < Nl; k++) { pr[k] = Y2r + sh[k] * sh[k]; qi[k] = Y2i; }
    csqrt_arr(Nl, pr, qi, rr, ri);
    for (int k = 0; k < Nl; k++) {
        rr[k] *= al; ri[k] *= al;
        double zr = b1r * l[k], zi = b1i * l[k]; double sg = zr < 0 ? -1.0 : 1.0; x1[k] = sg * zr; y1[k] = sg * zi;
        zr = b2r * rr[k] - b2i * ri[k]; zi = b2r * ri[k] + b2i * rr[k]; sg = zr < 0 ? -1.0 : 1.0; x2[k] = sg * zr; y2[k] = sg * zi;
        xe[k] = -ar * l2[k] / 2 + x1[k] + x2[k]; ye[k] = -ai * l2[k] / 2 + y1[k] + y2[k];
        m1[k] = -2 * x1[k]; n1[k] = -2 * y1[k]; m2[k] = -2 * x2[k]; n2[k] = -2 * y2[k];
    }
    double Eref = -INFINITY; for (int k = 0; k < Nl; k++) if (xe[k] > Eref) Eref = xe[k];
    for (int k = 0; k < Nl; k++) xe[k] -= Eref;                       /* the terms as e^{E_k - Eref}: G = S e^{Eref} */
    *E = Eref;
    dfl_vexp(Nl, xe, ee); dfl_vsincos(Nl, ye, se, ce);
    dfl_vexp(Nl, m1, e1); dfl_vsincos(Nl, n1, s1, c1); coshfac_arr(Nl, e1, s1, c1, f1r, f1i);
    dfl_vexp(Nl, m2, e2); dfl_vsincos(Nl, n2, s2, c2); coshfac_arr(Nl, e2, s2, c2, f2r, f2i);
    for (int k = 0; k < Nl; k++) {
        double er = ee[k] * ce[k], ei = ee[k] * se[k];                    /* e^E */
        double tr = er * f1r[k] - ei * f1i[k], ti = er * f1i[k] + ei * f1r[k];
        double ur = tr * f2r[k] - ti * f2i[k], ui = tr * f2i[k] + ti * f2r[k];
        double d = rr[k] * rr[k] + ri[k] * ri[k], f = wk[k] / d;                 /* / r */
        accr[k] = (ur * rr[k] + ui * ri[k]) * f; acci[k] = (ui * rr[k] - ur * ri[k]) * f;
    }
    double sr = 0, si = 0; for (int k = 0; k < Nl; k++) { sr += accr[k]; si += acci[k]; }
    return CPX(sr, si);
}
/* G = S e^{E}, S bounded.  Along the ray l = e s, e = e^{-i arg(a)/2}, the Gaussian is e^{-|a| s^2/2} (no oscillation) and
 * r = e (2Y e^{i arg a} + s^2)^{1/2}: the same integral with the arguments (Y e^{i arg a}, |a|, b1 e, b2 e), whose cut in Y
 * lies where a Y is negative real, as the cut of K_0((s2 - s1) X/2) of the central kernel does, so the paths of inner_gen,
 * which avoid that cut, serve (with the real-axis rule the cut would lie at negative real Y) */
static cdouble ncG_rot(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E, int libm)
{
    double ph = carg(a); cdouble e = cexp(CPX(0, -ph / 2));
    Y = cmul(Y, cexp(CPX(0, ph))); a = CPX(cabs(a), 0); b1 = cmul(b1, e); b2 = cmul(b2, e);
    return libm ? ncG_libm(Y, a, b1, b2, E) : ncG_vec(Y, a, b1, b2, E);
}
static cdouble ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
#if defined(__GNUC__) || defined(__clang__)
    __atomic_add_fetch(&DFL_NCG_CALLS, 1, __ATOMIC_RELAXED);
#else
    DFL_NCG_CALLS++;
#endif
    return ncG_rot(Y, a, b1, b2, E, DFL_NCG_LIBM);
}
cdouble dfl_ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm) { double E; cdouble S = ncG_rot(Y, a, b1, b2, &E, libm); return cmul(S, cexp(CPX(E, 0))); }   /* exported for the tests */
cdouble dfl_ncG_S(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm, double *E) { return ncG_rot(Y, a, b1, b2, E, libm); }   /* G = S e^E: S, and E */
/* phialg of cases DU, DT with a fixed start as S e^{E}, S bounded */
static cdouble phialg_scaled(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, double *E)
{
    cdouble a = csub(s2, s1);
    cdouble ex = cscale(0.5, csub(cmul(s1, cmul(nc->m2, nc->m2)), cmul(s2, cmul(nc->m1, nc->m1))));
    cdouble cst = cmul(cscale(2 / M_PI, nc->pref), cexp(ex));
    return sgn > 0 ? cmul(cst, ncG(X, a, cmul(nc->m2, s1), cmul(nc->m1, s2), E)) : cmul(cst, ncG(cneg(X), a, cmul(nc->m1, s2), cmul(nc->m2, s1), E));
}
static cdouble phialg(int kern, const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (nc && nc->nc) {
        if (kern == 1) {
            cdouble m = nc->m1;
            cdouble f = cmul(nc->pref, cexp(cscale(-0.5, cmul(s1, cmul(m, m)))));
            return cdiv(cmul(f, ccosh(cmul(cmul(m, s1), csqrt(cscale(2, X))))), csqrt(cscale(M_PI, X)));
        }
        double E; cdouble S = phialg_scaled(nc, X, sgn, s1, s2, &E);
        return cmul(S, cexp(CPX(E, 0)));
    }
    if (kern == 1) return cdiv(ONE, csqrt(cscale(M_PI, X)));
    cdouble arg = cmul(cscale(0.5, csub(s2, s1)), X);
    return cscale(1 / M_PI, dfl_k0x(sgn > 0 ? arg : cneg(arg)));
}
/* The noncentral density is a Gaussian in sqrt(2 X~) centered at the mean, e^{-s_d X~} cosh(b sqrt(2 X~)) with b = m s_d,
 * so its mass sits away from the edge by the factor e^{lam}, lam = m^2 s_d/2; the paths are built from the exponent
 * E = -t v^2 - s_d X + ncexp, ncexp = log cosh(b sqrt(2 X~)), and the nodes carry phialg e^{-ncexp}, which is bounded. */
static cdouble logcosh(cdouble z)
{
    cdouble zs = creal(z) >= 0 ? z : cneg(z);
    return csub(cadd(zs, clog(cadd(ONE, cexp(cscale(-2, zs))))), CPX(0.6931471805599453, 0));
}
/* tanh z without overflow: 1 - 2/(e^{2z} + 1) for Re z >= 0, odd otherwise */
static cdouble ctanh_safe(cdouble z)
{
    bool neg = creal(z) < 0; if (neg) z = cneg(z);
    cdouble t = csub(ONE, cdiv(CPX(2, 0), cadd(cexp(cscale(2, z)), ONE)));
    return neg ? cneg(t) : t;
}
static cdouble ncb(const nck_t *nc, int sgn, cdouble s1, cdouble s2)
{
    if (nc->kern == 1) return cmul(nc->m1, s1);
    return sgn > 0 ? cmul(nc->m1, s2) : cmul(nc->m2, s1);
}
static cdouble ncexp(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble Xt = sgn > 0 ? X : cneg(X);
    return logcosh(cmul(ncb(nc, sgn, s1, s2), csqrt(cscale(2, Xt))));
}
/* d ncexp / dv = b^2 (tanh z / z) (+-dX/dv), z = b sqrt(2 X~), finite at the edge */
static cdouble ncexp_dv(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, double dXdv)
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble b = ncb(nc, sgn, s1, s2), Xt = sgn > 0 ? X : cneg(X), z = cmul(b, csqrt(cscale(2, Xt)));
    cdouble tz = cabs(z) < 1e-6 ? ONE : cdiv(ctanh_safe(z), z);
    return cmul(cmul(b, b), cscale(sgn > 0 ? dXdv : -dXdv, tz));
}
static cdouble phired(int kern, const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (nc && nc->nc && kern == 2) {
        double E; cdouble S = phialg_scaled(nc, X, sgn, s1, s2, &E);
        return cmul(S, cexp(csub(CPX(E, 0), ncexp(nc, X, sgn, s1, s2))));
    }
    cdouble f = phialg(kern, nc, X, sgn, s1, s2);
    return (nc && nc->nc) ? cmul(f, cexp(cneg(ncexp(nc, X, sgn, s1, s2)))) : f;
}
/* Psi_t(1/2) with a fixed start: P[J(1)^2 <= 1 + g0^2] (case DK), P[R <= 1/2] by the l-rule of Pkap (DU, DT), times the mass */
/* P[R <= 1/2] under the tilt with a fixed start, cases DU, DT: 2 int_0^inf phibar(l; m2, w2) Fbar(sqrt(1 + l^2); m1, w1) dl,
 * X1 ~ N(m1, w1 = 1/s2), X2 ~ N(m2, w2 = -1/s1), by the l-rule (the Gaussian in l peaks at |m2|) */
static cdouble ncP(cdouble s1, cdouble s2, cdouble m1, cdouble m2, int Nl)
{
    cdouble w1 = cdiv(ONE, s2), w2 = cneg(cdiv(ONE, s1)), iw2 = cdiv(ONE, w2), mw = cmul(m2, iw2);
    double ra = fmax(creal(iw2), 1e-300), bb = cabs(mw);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    double ls[NLMAX], ws[NLMAX]; int n = lrule(1.0, lm, fabs(creal(mw)), ra, cimag(iw2), fabs(cimag(mw)), Nl, ls, ws);
    cdouble sq1 = csqrt(cscale(2, w1)), c2 = cdiv(ONE, cscale(2, csqrt(cscale(2 * M_PI, w2)))), P = ZERO;
    for (int k = 0; k < n; k++) {
        double l = ls[k], u = sqrt(1 + l * l);
        cdouble lm2 = csub(CPX(l, 0), m2), lp2 = cadd(CPX(l, 0), m2);
        cdouble f2 = cmul(cadd(cexp(cneg(cdiv(cmul(lm2, lm2), cscale(2, w2)))), cexp(cneg(cdiv(cmul(lp2, lp2), cscale(2, w2))))), c2);
        cdouble F1 = cscale(0.5, cadd(dfl_erf(cdiv(csub(CPX(u, 0), m1), sq1)), dfl_erf(cdiv(cadd(CPX(u, 0), m1), sq1))));
        P = cadd(P, cscale(ws[k], cmul(f2, F1)));
    }
    return cscale(2, P);
}
static cdouble psi_half_nc(const nck_t *nc, cdouble s1, cdouble s2, cdouble M0)
{
    if (nc->kern == 1) {
        double r = sqrt(1 + nc->g0 * nc->g0); cdouble w = csqrt(cdiv(CPX(2, 0), s1));
        cdouble e = cscale(0.5, cadd(dfl_erf(cdiv(csub(CPX(r, 0), nc->m1), w)), dfl_erf(cdiv(cadd(CPX(r, 0), nc->m1), w))));
        return cmul(cdiv(nc->pref, csqrt(s1)), e);
    }
    return cmul(cmul(nc->pref, M0), ncP(s1, s2, nc->m1, nc->m2, DFL_NLNC));
}

/* ---- nodes of the pieces ---- */
typedef cdouble (*efn_t)(double s, const void *ctx);
typedef struct { double *y, *w; int n, cap; } nodes_t;
static void nodes_reserve(nodes_t *N, int extra)
{
    if (N->n + extra <= N->cap) return;
    int cap = N->cap ? N->cap : 1024; while (cap < N->n + extra) cap *= 2;
    N->y = realloc(N->y, cap * sizeof(double)); N->w = realloc(N->w, cap * sizeof(double)); N->cap = cap;
}
static void glnodes(double a, double b, int n, int p, nodes_t *N)
{
    n *= DFL_NODEFAC;
    nodes_reserve(N, n);
    const double *gw; const double *u = dfl_rule(n, &gw); double L = b - a;
    double *y = N->y, *wt = N->w; int cnt = N->n;
    for (int k = 0; k < n; k++) {
        if (p == 1) { y[cnt] = a + L * u[k]; wt[cnt] = L * gw[k]; }
        else if (p == 2) { y[cnt] = a + L * u[k] * u[k]; wt[cnt] = 2 * L * u[k] * gw[k]; }
        else { double u2 = u[k] * u[k]; y[cnt] = a + L * u2 * u2; wt[cnt] = 4 * L * u2 * u[k] * gw[k]; }
        cnt++;
    }
    N->n = cnt;
}
#define NGROW 16
#define NLEV 5
#define NCON 16
/* nodes for int_0^L: the first piece shortened until the exponent falls by at most 4, rises by at most 4 (the noncentral
 * kernels rise away from the edge) and its phase changes by at most 40 rad, then pieces growing by 4 with node counts from
 * the change of the exponent, until it is below -EXPO */
static void pieces_nodes(efn_t Ef, const void *ctx, double L, double first, int p, int nmin, nodes_t *N)
{
    N->n = 0;
    first = fmin(first, L);
    for (int it = 0; it < 60; it++) {
        cdouble E[9]; double mx = -INFINITY; for (int i = 0; i <= 8; i++) { E[i] = Ef(first * i / 8, ctx); mx = fmax(mx, creal(E[i])); }
        double dph = 0; for (int i = 0; i < 8; i++) dph += fabs(cimag(E[i + 1]) - cimag(E[i]));
        if ((creal(E[8]) > -4 && mx - creal(E[0]) < 4 && dph < 40) || first <= 1e-12 * L) break;
        first /= 2;
    }
    glnodes(0, first, nmin, p, N);
    double a = first;
    while (a < L) {
        double bn = fmin(L, 4 * a);
        cdouble E[9]; bool all = true; for (int i = 0; i <= 8; i++) { E[i] = Ef(a + (bn - a) * i / 8, ctx); if (creal(E[i]) >= -EXPO) all = false; }
        if (all) break;
        double dph = 0; for (int i = 0; i < 8; i++) dph += cabs(csub(E[i + 1], E[i]));
        int n = (int) fmax(NGROW, 16 * ceil(ceil(1.2 * dph + 16) / 16));
        int m = (n + 159) / 160; n = 16 * (int) ceil(ceil((double) n / m) / 16);
        for (int k = 0; k < m; k++) glnodes(a + (bn - a) * k / m, a + (bn - a) * (k + 1) / m, n, 1, N);
        if (creal(E[8]) < -EXPO) break;
        a = bn;
    }
}
/* nodes for int_0^1 graded geometrically toward both ends, node counts from the change of the exponent, sub-pieces on
 * which the exponent is below max(E1, E2) - EXPO skipped */
static void phase_nodes(efn_t Ef, const void *ctx, cdouble E1, cdouble E2, nodes_t *N)
{
    N->n = 0;
    /* ss = 0, 2^-NLEV, ..., 2^-1, 1 - 2^-2, ..., 1 - 2^-NLEV, 1 (ascending) */
    double ss[2 * NLEV + 1]; int ns = 0;
    ss[ns++] = 0; for (int k = NLEV; k >= 1; k--) ss[ns++] = pow(2.0, -k);
    for (int k = 2; k <= NLEV; k++) ss[ns++] = 1 - pow(2.0, -k); ss[ns++] = 1;
    double Emax = fmax(creal(E1), creal(E2)) - EXPO + 2;
    for (int k = 0; k < ns - 1; k++) {
        cdouble E[9]; double mx = -INFINITY; for (int i = 0; i <= 8; i++) { E[i] = Ef(ss[k] + (ss[k + 1] - ss[k]) * i / 8, ctx); mx = fmax(mx, creal(E[i])); }
        if (mx < Emax) continue;
        double dph = 0; for (int i = 0; i < 8; i++) dph += cabs(csub(E[i + 1], E[i]));
        int n = (int) fmax(NCON, 8 * ceil(ceil(1.2 * dph + 16) / 8));
        int mm = (n + 159) / 160; n = 8 * (int) ceil(ceil((double) n / mm) / 8);
        for (int i = 0; i < mm; i++) glnodes(ss[k] + (ss[k + 1] - ss[k]) * i / mm, ss[k] + (ss[k + 1] - ss[k]) * (i + 1) / mm, n, 1, N);
    }
}
/* the direction and length of the steepest-descent piece at an end */
static void descent(cdouble gam, cdouble alpha, int dir, double L, double dep, double sg, double deg, cdouble *u, double *l)
{
    cdouble uu = cabs(gam) > 0 ? cdiv(cneg(conj(gam)), CPX(cabs(gam), 0)) : CPX(0, -sg);
    double cmax = cos(deg * M_PI / 180);
    if (creal(uu) * dir < cmax) {
        double si = cimag(uu) > 0 ? 1 : cimag(uu) < 0 ? -1 : -sg;
        uu = CPX(dir * cmax, si * sin(deg * M_PI / 180));
    }
    double g1 = creal(cmul(gam, uu)), g2 = -creal(cmul(alpha, cmul(uu, uu)));
    double ll = L / 2;
    if (fabs(cimag(uu)) > 0) ll = fmin(ll, dep / fabs(cimag(uu)));
    if (g1 < 0) {
        if (g2 > 0) ll = fmin(ll, -g1 / (2 * g2));
        double disc = g1 * g1 - 4 * g2 * EXPO;
        if (g2 == 0) ll = fmin(ll, -EXPO / g1);
        else if (disc >= 0) { double sE = (-g1 - sqrt(disc)) / (2 * g2); if (sE > 0) ll = fmin(ll, sE); }
    }
    *l = fmax(ll, 1e-12 * L); *u = uu;
}

/* ---- the segment integral ---- */
typedef struct {
    int kern, kind; cdouble tj, sd, s1, s2; int sgn; const breaks_t *R; double q, c0;
    double a, b; cdouble ua, ub; double kap, m; /* partition */
    const nck_t *nc;                            /* the noncentral kernel of a fixed start, or NULL */
} sctx;
static cdouble W_of(const sctx *c, cdouble v) { return c->kind == 1 ? cscale(2, cmul(v, v)) : cadd(CPX(c->q, 0), cscale(2 * c->c0, v)); }
static cdouble Ez(const sctx *c, cdouble z, double o) { cdouble v = cadd(CPX(o, 0), z), X = xat(c->R, z, o);
    return cadd(csub(cneg(cmul(c->tj, cmul(v, v))), cmul(c->sd, X)), ncexp(c->nc, X, c->sgn, c->s1, c->s2)); }
static cdouble Fz(const sctx *c, cdouble z, double o) { cdouble X = xat(c->R, z, o);
    if (c->nc && c->nc->nc && c->kern == 2) {                             /* the factor as S e^E with S bounded: one exponential */
        double E; cdouble S = phialg_scaled(c->nc, X, c->sgn, c->s1, c->s2, &E);
        return cmul(cmul(W_of(c, cadd(CPX(o, 0), z)), S), cexp(cadd(csub(Ez(c, z, o), ncexp(c->nc, X, c->sgn, c->s1, c->s2)), CPX(E, 0)))); }
    return cmul(cmul(W_of(c, cadd(CPX(o, 0), z)), cexp(Ez(c, z, o))), phired(c->kern, c->nc, X, c->sgn, c->s1, c->s2)); }
typedef struct { const sctx *c; cdouble u; double o; cdouble E0; } ectx;
static cdouble E_piece(double s, const void *v) { const ectx *e = v; return csub(Ez(e->c, cscale(s, e->u), e->o), e->E0); }
typedef struct { const sctx *c; cdouble leg, dv; } lctx;
static cdouble E_leg(double s, const void *v) { const lctx *e = v; return Ez(e->c, csub(cadd(e->leg, cscale(s, e->dv)), CPX(e->c->a, 0)), e->c->a); }
static cdouble chi_seg(const sctx *c, cdouble v) { return cscale(0.5, cadd(ONE, dfl_erf(cscale(c->kap, csub(v, CPX(c->m, 0)))))); }

static void segint(sctx *c, int pa, int pb, double ella, double ellb, bool ws, bool wl, bool wr, double sg, int ka, int kb, double dep,
                   cdouble *vs, cdouble *vl, cdouble *vr)
{
    double a = c->a, b = c->b, L = b - a; cdouble alpha = cadd(c->tj, cscale(c->c0, c->sd));
    cdouble Eb = Ez(c, ZERO, b), Ea = Ez(c, ZERO, a);
    double dega = 80, fa = ella / 2; if (ka == 0) { dega = 45; fa = ella; }
    double degb = 80, fb = ellb / 2; if (kb == 0) { degb = 45; fb = ellb; }
    cdouble ua, ub; double la, lb;
    descent(cadd(csub(cscale(-2 * a, c->tj), cscale(dXv(c->R, a), c->sd)), ncexp_dv(c->nc, xat(c->R, ZERO, a), c->sgn, c->s1, c->s2, dXv(c->R, a))), alpha, 1, L, dep, sg, dega, &ua, &la);
    descent(cadd(csub(cscale(-2 * b, c->tj), cscale(dXv(c->R, b), c->sd)), ncexp_dv(c->nc, xat(c->R, ZERO, b), c->sgn, c->s1, c->s2, dXv(c->R, b))), alpha, -1, L, dep, sg, degb, &ub, &lb);
    cdouble pa_ = cadd(CPX(a, 0), cscale(la, ua)), pb_ = cadd(CPX(b, 0), cscale(lb, ub));
    nodes_t Nd = {NULL, NULL, 0, 0};
    int capv = 4096, nv = 0; cdouble *fv = malloc(capv * sizeof(cdouble)), *vv = malloc(capv * sizeof(cdouble)), *cv = malloc(capv * sizeof(cdouble));
    /* the integrand is evaluated only where the wanted parts of the partition (chi near b, 1 - chi near a) exceed 1e-15:
     * the edge part skips the nodes near a, the smooth part those near b */
    bool needa = ws || wl, needb = wr;
#define KEEP(V) (chi = chi_seg(c, (V)), (needa && cabs(csub(ONE, chi)) > 1e-15) || (needb && cabs(chi) > 1e-15))
#define PUSHV(FEXPR, V) do { cdouble chi; if (KEEP(V)) { if (nv == capv) { capv *= 2; fv = realloc(fv, capv * sizeof(cdouble)); vv = realloc(vv, capv * sizeof(cdouble)); cv = realloc(cv, capv * sizeof(cdouble)); } fv[nv] = (FEXPR); vv[nv] = (V); cv[nv] = chi; nv++; } } while (0)
    ectx ea = {c, ua, a, Ea}; pieces_nodes(E_piece, &ea, la, fmin(la / 8, fa), pa, 48 + 16 * (pa == 1), &Nd);
    for (int k = 0; k < Nd.n; k++) PUSHV(cmul(Fz(c, cscale(Nd.y[k], ua), a), cscale(Nd.w[k], ua)), cadd(CPX(a, 0), cscale(Nd.y[k], ua)));
    int na_ = Nd.n;
    ectx eb = {c, ub, b, Eb}; pieces_nodes(E_piece, &eb, lb, fmin(lb / 8, fb), pb, 48, &Nd);
    if (getenv("DFL_DEBUG3")) { cdouble sa = ZERO, sb = ZERO, ca = ZERO, cb = ZERO; for (int k = 0; k < na_; k++) { sa = cadd(sa, fv[k]); ca = cadd(ca, cmul(fv[k], chi_seg(c, vv[k]))); }
        for (int k = 0; k < Nd.n; k++) { cdouble f = cneg(cmul(Fz(c, cscale(Nd.y[k], ub), b), cscale(Nd.w[k], ub))); sb = cadd(sb, f); cb = cadd(cb, cmul(f, chi_seg(c, cadd(CPX(b, 0), cscale(Nd.y[k], ub))))); }
        printf("  pieces: a-side %d nodes sum %.12g%+.12gi chi-sum %.12g%+.12gi, b-side %d nodes sum %.12g%+.12gi chi-sum %.12g%+.12gi, first nodes y %.12g %.12g, kap %.6g m %.6g\n", na_, creal(sa), cimag(sa), creal(ca), cimag(ca), Nd.n, creal(sb), cimag(sb), creal(cb), cimag(cb), Nd.y[0], Nd.y[1], c->kap, c->m); }
    for (int k = 0; k < Nd.n; k++) PUSHV(cneg(cmul(Fz(c, cscale(Nd.y[k], ub), b), cscale(Nd.w[k], ub))), cadd(CPX(b, 0), cscale(Nd.y[k], ub)));
    if (cabs(csub(pb_, pa_)) > 0) {
        double Emax = fmax(creal(Ea), creal(Eb)) + 2;
        cdouble legs[4] = {pa_, pb_}; int nl = 2;
        double mx = -INFINITY; for (int k = 0; k <= 32; k++) mx = fmax(mx, creal(Ez(c, csub(cadd(pa_, cscale(k / 32.0, csub(pb_, pa_))), CPX(a, 0)), a)));
        if (mx > Emax) { legs[0] = pa_; legs[1] = CPX(creal(pa_), 0); legs[2] = CPX(creal(pb_), 0); legs[3] = pb_; nl = 4; }
        if (getenv("DFL_DEBUG3")) printf("  legs %d (mx %.6g Emax %.6g Ea %.6g%+.6gi Eb %.6g%+.6gi)\n", nl, mx, Emax, creal(Ea), cimag(Ea), creal(Eb), cimag(Eb));
        for (int k = 0; k < nl - 1; k++) {
            cdouble dv = csub(legs[k + 1], legs[k]); if (cabs(dv) == 0) continue;
            lctx lc = {c, legs[k], dv}; phase_nodes(E_leg, &lc, Ea, Eb, &Nd);
            cdouble legsum = ZERO;
            for (int i = 0; i < Nd.n; i++) { cdouble vc = cadd(legs[k], cscale(Nd.y[i], dv));
                cdouble f = cabs(csub(vc, CPX(b, 0))) < cabs(csub(vc, CPX(a, 0))) ? Fz(c, csub(vc, CPX(b, 0)), b) : Fz(c, csub(vc, CPX(a, 0)), a);
                PUSHV(cmul(f, cscale(Nd.w[i], dv)), vc); legsum = cadd(legsum, cmul(f, cscale(Nd.w[i], dv))); }
            if (getenv("DFL_DEBUG3")) printf("  leg %d: %d nodes, sum %.12g%+.12gi\n", k, Nd.n, creal(legsum), cimag(legsum));
        }
    }
    if (getenv("DFL_DEBUG3")) printf("segint a %.6g b %.6g sgn %d sd %.12g%+.12gi la %.12g lb %.12g ua %.12g%+.12gi ub %.12g%+.12gi pa_ %.12g%+.12gi pb_ %.12g%+.12gi nv %d\n",
                                      a, b, c->sgn, creal(c->sd), cimag(c->sd), la, lb, creal(ua), cimag(ua), creal(ub), cimag(ub), creal(pa_), cimag(pa_), creal(pb_), cimag(pb_), nv);
#if defined(__GNUC__) || defined(__clang__)
    __atomic_add_fetch(&DFL_NCG_SEG, nv, __ATOMIC_RELAXED);
#endif
    *vs = *vl = *vr = ZERO;
    if (ws || wl) { cdouble val = ZERO; for (int k = 0; k < nv; k++) val = cadd(val, cmul(fv[k], csub(ONE, cv[k]))); if (ws) *vs = val; else *vl = val; }
    if (wr) { cdouble val = ZERO; for (int k = 0; k < nv; k++) val = cadd(val, cmul(fv[k], cv[k])); *vr = val; }
#undef PUSHV
#undef KEEP
    free(Nd.y); free(Nd.w); free(fv); free(vv); free(cv);
}

/* ---- the transforms of the general kernel: Is and Ie[k] ---- */
typedef struct { const sctx *c; double a; } actx;
static cdouble A_ray(cdouble rho, const void *v) { const actx *x = v; return cmul(W_of(x->c, cadd(CPX(x->a, 0), rho)), phialg(x->c->kern, x->c->nc, xat(x->c->R, rho, x->a), x->c->sgn, x->c->s1, x->c->s2)); }
typedef struct { const actx *ax; cdouble e, be; double aa; } pctx;
static cdouble pseudo_E(double r, const void *v) { const pctx *p = v; return csub(CPX(-p->aa * r * r, 0), cscale(r, p->be)); }
typedef struct { cfn_t A; const void *Actx; cdouble alpha, beta, e; double aa; } rxctx;
static cdouble rx_f(double r, const void *v) { const rxctx *x = v; cdouble arg = csub(CPX(-x->aa * r * r, 0), cscale(r, cmul(x->beta, x->e)));
    return cmul(cmul(x->A(cscale(r, x->e), x->Actx), cexp(arg)), x->e); }
/* int_0^inf A(r) exp(-alpha r^2 - beta r) dr: along the ray of the Gaussian's decay, or by gaussexp's paths */
cdouble dfl_rayint(cdouble (*f)(double, const void *), const void *ctx, double R, double l, int n1, int n2, int p, double osc);
static cdouble rayexp(cfn_t A, const void *Actx, cdouble alpha, cdouble beta, double lA, int p)
{
    cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
    /* the straight ray when the exponential grows along it by at most e^2 and its phase stays below 150 radians (the
     * level-curve path of gaussexp, taken otherwise when the saddle z = beta / (2 sqrt(alpha)) has Re z >= 0, passes the
     * branch point of its square root at the distance |z| from the origin, where the Gaussian is e^{-|z|^2}: with the cap
     * of 60 radians of ray_ok it took over at |Im z| ~ 4, where that is 1e-7, and the inner transform lost seven digits
     * on a short stretch of the line, 1e-10 in the distribution function in case DK with c0 > 0 at x >= 0; at 150
     * radians the level curve begins at |Im z| ~ 10, where the branch point is below 1e-40) */
    double m = fmax(0, -creal(be));
    if (m * m / (4 * aa) <= 2 && fabs(cimag(be)) * sqrt(EXPO / aa) <= 150) {
        double Rr = sqrt(EXPO / aa);
        if (creal(be) > 0) Rr = fmin(Rr, EXPO / creal(be));
        rxctx x = {A, Actx, alpha, beta, e, aa};
        return dfl_rayint(rx_f, &x, Rr, lA, 48, 32, p, fabs(cimag(be)));
    }
    return dfl_gaussexp(A, Actx, alpha, 1.0, beta, lA, 0, p);
}
/* the ray from a for the noncentral kernel: the straight ray rho = r e in the direction of the Gaussian's decay, or, when
 * the exponential grows along it by more than e^2, the segment to the saddle x0 = -beta/(2 alpha) and the descent line from
 * it; the nodes from pieces_nodes with the full exponent (quadratic plus ncexp) */
typedef struct { const sctx *c; cdouble alpha, beta, e, x0, E0; double a; int mode; } nrctx;
static cdouble Efull_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    return cadd(csub(cneg(cmul(x->alpha, cmul(rho, rho))), cmul(x->beta, rho)), ncexp(c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2)); }
static cdouble Ared_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    return cmul(W_of(c, cadd(CPX(x->a, 0), rho)), phired(c->kern, c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2)); }
static cdouble Ef_nc(double r, const void *v) { const nrctx *x = v;
    if (x->mode == 0) return Efull_nc(x, cscale(r, x->e));
    if (x->mode == 1) return Efull_nc(x, cscale(r, x->x0));
    return csub(Efull_nc(x, cadd(x->x0, cscale(r, x->e))), x->E0); }
static double raylen(efn_t Ef, const void *ctx, double R0)
{
    double Rr = R0;
    for (int it = 0; it < 60; it++) { if (creal(Ef(Rr, ctx)) < -EXPO) break; Rr *= 1.5; }
    return Rr;
}
/* the full exponent's derivative along rho, and the scaled factor (S, E) at rho: Ared = S e^E */
static cdouble ncexp_dv_c(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, cdouble dXdv)   /* complex dX/dv */
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble b = ncb(nc, sgn, s1, s2), Xt = sgn > 0 ? X : cneg(X), z = cmul(b, csqrt(cscale(2, Xt)));
    cdouble tz = cabs(z) < 1e-6 ? ONE : cdiv(ctanh_safe(z), z);
    return cmul(cmul(cmul(b, b), tz), sgn > 0 ? dXdv : cneg(dXdv));
}
static cdouble dEfull_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    cdouble dX = cadd(CPX(c->R->q, 0), cscale(2 * c->R->c0, cadd(CPX(x->a, 0), rho)));
    return cadd(csub(cscale(-2, cmul(x->alpha, rho)), x->beta), ncexp_dv_c(c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2, dX)); }
static cdouble Ared_sc(const nrctx *x, cdouble rho, double *E) { const sctx *c = x->c;
    cdouble X = xat(c->R, rho, x->a), Wv = W_of(c, cadd(CPX(x->a, 0), rho));
    if (c->kern == 1) { *E = 0; return cmul(Wv, phired(c->kern, c->nc, X, c->sgn, c->s1, c->s2)); }   /* case DK: elementary, bounded */
    cdouble S = phialg_scaled(c->nc, X, c->sgn, c->s1, c->s2, E), ne = ncexp(c->nc, X, c->sgn, c->s1, c->s2);
    *E -= creal(ne);                                                   /* the real part; the phase stays in S */
    return cmul(cmul(Wv, S), cexp(CPX(0, -cimag(ne)))); }
/* a straight piece z0 + r d, r in [0, L]: the exponent relative to z0 */
typedef struct { const nrctx *x; cdouble z0, d, Ez0; } spctx;
static cdouble E_sp(double r, const void *v) { const spctx *p = v; return csub(Efull_nc(p->x, cadd(p->z0, cscale(r, p->d))), p->Ez0); }
/* the ray when the full exponent rises along the real axis by more than 2 above its value at the edge (the mass of the
 * noncentral density beyond the breakpoint): the straight piece from 0 to the saddle of the full exponent (Newton on E'
 * from the maximum of Re E on the real axis), then straight pieces in the direction of steepest descent at their start,
 * renewed where the descent along a piece ends, until E has fallen by EXPO.  Returns false when there is no such rise. */
static bool ray_saddle(const nrctx *x, double first, int p, double Rmax, cdouble *out)
{
    double E0 = creal(Efull_nc(x, ZERO)), Emax = -INFINITY, rmax = 0;
    for (int k = 0; k <= 80; k++) { double r = Rmax * pow(1.25, k - 80), Er = creal(Efull_nc(x, CPX(r, 0))); if (Er > Emax) { Emax = Er; rmax = r; } }
    if (!(Emax - E0 > 2)) return false;
    /* the maximum on the real axis refined (Newton on Re E' along the axis, within the scan's cell), so that the saddle is
     * judged against the true height of the bump */
    double rr = rmax;
    for (int it = 0; it < 20; it++) {
        double h = 1e-6 * rr, d1 = creal(dEfull_nc(x, CPX(rr, 0))), d2 = creal(csub(dEfull_nc(x, CPX(rr + h, 0)), dEfull_nc(x, CPX(rr - h, 0)))) / (2 * h);
        if (!(d2 < 0)) break;
        double rn = rr - d1 / d2; if (!(rn > rmax / 1.25 && rn < rmax * 1.25)) break;
        if (fabs(rn - rr) < 1e-10 * rr) { rr = rn; break; }
        rr = rn;
    }
    Emax = fmax(Emax, creal(Efull_nc(x, CPX(rr, 0)))); rmax = rr;
    cdouble rho = CPX(rmax, 0);
    for (int it = 0; it < 30; it++) {
        cdouble d1 = dEfull_nc(x, rho); double h = 1e-6 * cabs(rho);
        cdouble d2 = cscale(1 / (2 * h), csub(dEfull_nc(x, cadd(rho, CPX(h, 0))), dEfull_nc(x, csub(rho, CPX(h, 0)))));
        if (cabs(d2) == 0) break;
        cdouble step = cneg(cdiv(d1, d2));
        if (cabs(step) > cabs(rho)) step = cscale(cabs(rho) / cabs(step), step);
        rho = cadd(rho, step);
        if (cabs(step) < 1e-10 * cabs(rho)) break;
    }
    /* the saddle serves only if it is the bump of the real axis slightly displaced: one far above it (Re E beyond the
     * real-axis maximum) or far from it is not crossed by the real-axis path, and the rotated ray serves instead */
    if (!(isfinite(creal(rho)) && isfinite(cimag(rho)) && creal(rho) > 0 && creal(Efull_nc(x, rho)) > E0 + 1
          && creal(Efull_nc(x, rho)) <= Emax + 1 && fabs(cimag(rho)) <= 0.5 * fabs(creal(rho)))) {
        /* a small rise: the rotated ray serves (its cancellation is at most e^{rise}); a large one: the real-axis maximum
         * is the pivot, and the descent pieces start from it */
        if (!(Emax - E0 > 10)) return false;
        rho = CPX(rmax, 0);
    }
    cdouble val = ZERO; nodes_t Nd = {NULL, NULL, 0, 0};
    cdouble E_s = Efull_nc(x, rho);
    spctx sp = {x, ZERO, rho, ZERO};                                   /* lam in [0, 1] along rho */
    pieces_nodes(E_sp, &sp, 1.0, fmin(1.0 / 8, first / cabs(rho)), p, 48, &Nd);
    for (int k = 0; k < Nd.n; k++) { cdouble z = cscale(Nd.y[k], rho); double E; cdouble S = Ared_sc(x, z, &E);
        val = cadd(val, cscale(Nd.w[k], cmul(S, cexp(cadd(Efull_nc(x, z), CPX(E, 0)))))); }
    val = cmul(val, rho);
    if (getenv("DFL_DEBUG4") && !(creal(val) == creal(val))) printf("  NaN already on the piece to the saddle: rho %g%+gi nodes %d\n", creal(rho), cimag(rho), Nd.n);
    cdouble z0 = rho; double h = 1e-6 * cabs(rho);
    cdouble d2 = cscale(1 / (2 * h), csub(dEfull_nc(x, cadd(rho, CPX(h, 0))), dEfull_nc(x, csub(rho, CPX(h, 0)))));
    cdouble d = cabs(d2) > 0 ? cexp(CPX(0, (M_PI - carg(d2)) / 2)) : ONE; if (creal(d) < 0) d = cneg(d);
    for (int piece = 0; piece < 16; piece++) {
        cdouble Ez0 = Efull_nc(x, z0);
        spctx sq = {x, z0, d, Ez0};
        /* the length: along the line while the exponent falls; stop where it has fallen by EXPO, or at the last point before
         * it rises (the direction is renewed there from the gradient), or at 10 Rmax */
        double L = 1e-3 * cabs(rho), Lend = NAN, Eprev = 0;
        for (int it = 0; it < 200; it++) {
            double Ec = creal(E_sp(L, &sq));
            if (Ec < -EXPO) { Lend = L; break; }
            if (!(Ec == Ec) || Ec > Eprev) { Lend = it == 0 ? L : L / 1.4; break; }
            if (L > 10 * Rmax) { Lend = L; break; }
            Eprev = Ec; L *= 1.4;
        }
        if (isnan(Lend)) Lend = L;
        pieces_nodes(E_sp, &sq, Lend, fmin(Lend / 8, first), 1, 48, &Nd);
        for (int k = 0; k < Nd.n; k++) { cdouble z = cadd(z0, cscale(Nd.y[k], d)); double E; cdouble S = Ared_sc(x, z, &E);
            val = cadd(val, cscale(Nd.w[k], cmul(cmul(S, cexp(cadd(cadd(E_sp(Nd.y[k], &sq), Ez0), CPX(E, 0)))), d))); }
        if (getenv("DFL_DEBUG4")) printf("  piece %d: z0 %g%+gi d %g%+gi Lend %g nodes %d val %g%+gi Ez0 %g%+gi\n", piece, creal(z0), cimag(z0), creal(d), cimag(d), Lend, Nd.n, creal(val), cimag(val), creal(Ez0), cimag(Ez0));
        z0 = cadd(z0, cscale(Lend, d));
        if (creal(csub(Efull_nc(x, z0), E_s)) < -EXPO) break;
        cdouble g = dEfull_nc(x, z0); if (cabs(g) == 0) break;
        d = cscale(1 / cabs(g), cneg(conj(g)));
    }
    free(Nd.y); free(Nd.w);
    if (getenv("DFL_DEBUG4") && !(creal(val) == creal(val))) {
        cdouble z = rho; double E; cdouble S = Ared_sc(x, z, &E);
        printf("ray_saddle NaN: alpha %g%+gi beta %g%+gi rho_s %g%+gi E_s %g%+gi rmax-scan Emax %g E0 %g S(rho_s) %g%+gi E %g d %g%+gi\n",
               creal(x->alpha), cimag(x->alpha), creal(x->beta), cimag(x->beta), creal(rho), cimag(rho), creal(E_s), cimag(E_s), Emax, E0, creal(S), cimag(S), E, creal(d), cimag(d));
    }
    *out = val;
    return true;
}
static cdouble rayexp_nc(const sctx *c, double a, cdouble alpha, cdouble beta, double first, int p)
{
    cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
    nrctx x = {c, alpha, beta, e, ZERO, ZERO, a, 0};
    nodes_t Nd = {NULL, NULL, 0, 0};
    cdouble val = ZERO;
    if (fmax(0, -creal(be)) * fmax(0, -creal(be)) / (4 * aa) <= 2) {
        double Rr = raylen(Ef_nc, &x, (-creal(be) + sqrt(creal(be) * creal(be) + 4 * aa * EXPO)) / (2 * aa));
        if (ray_saddle(&x, first, p, Rr, &val)) return val;
        pieces_nodes(Ef_nc, &x, Rr, fmin(Rr / 8, first), p, 48, &Nd);
        for (int k = 0; k < Nd.n; k++) { cdouble rho = cscale(Nd.y[k], e); double E; cdouble S = Ared_sc(&x, rho, &E); val = cadd(val, cscale(Nd.w[k], cmul(S, cexp(cadd(Efull_nc(&x, rho), CPX(E, 0)))))); }
#if defined(__GNUC__) || defined(__clang__)
        __atomic_add_fetch(&DFL_NCG_RAY, Nd.n, __ATOMIC_RELAXED);
#endif
        free(Nd.y); free(Nd.w);
        return cmul(val, e);
    }
    x.x0 = cdiv(cneg(beta), cscale(2, alpha)); x.mode = 1;
    pieces_nodes(Ef_nc, &x, 1.0, fmin(1.0 / 8, first / cabs(x.x0)), p, 48, &Nd);
    for (int k = 0; k < Nd.n; k++) { cdouble rho = cscale(Nd.y[k], x.x0); double E; cdouble S = Ared_sc(&x, rho, &E); val = cadd(val, cscale(Nd.w[k], cmul(S, cexp(cadd(Efull_nc(&x, rho), CPX(E, 0)))))); }
    val = cmul(val, x.x0);
    x.E0 = Efull_nc(&x, x.x0); x.mode = 2;
    double Rr = raylen(Ef_nc, &x, sqrt(EXPO / aa));
    pieces_nodes(Ef_nc, &x, Rr, fmin(Rr / 8, first), 1, 48, &Nd);
    cdouble v2 = ZERO;
    for (int k = 0; k < Nd.n; k++) { cdouble rho = cadd(x.x0, cscale(Nd.y[k], e)); double E; cdouble S = Ared_sc(&x, rho, &E); v2 = cadd(v2, cscale(Nd.w[k], cmul(S, cexp(cadd(Efull_nc(&x, rho), CPX(E, 0)))))); }
    free(Nd.y); free(Nd.w);
    return cadd(val, cmul(v2, e));
}
static void inner_gen(int kern, const nck_t *nc, cdouble tj, const breaks_t *R, cdouble s1, cdouble s2, cdouble M0, int kind, int part, cdouble *Is, cdouble *Ie)
{
    double q = R->q, c0 = R->c0; int ne = R->ne;
    double sg = cimag(tj) >= 0 ? 1.0 : -1.0;
    *Is = ZERO; for (int k = 0; k < 2; k++) Ie[k] = ZERO;
    bool wants = part == 0 || part == 1;
#define WANT(k) ((part == 0 || part == 2 || part == 10 + (k) + 1) && !R->negl[k])
    if (kind == 2 && wants) *Is = (nc && nc->nc) ? psi_half_nc(nc, s1, s2, M0) : dfl_psi_half(kern, s1, s2, M0);
    double lX = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
    double ed[4] = {0, ne > 0 ? R->b[0] : INFINITY, ne > 1 ? R->b[1] : INFINITY, INFINITY};
    for (int j = 0; j <= ne; j++) {
        double a = ed[j], bb = ed[j + 1]; int sgn = R->sgn[j];
        if (kern == 1 && sgn < 0) continue;
        cdouble sd = (kern == 1 || sgn < 0) ? s1 : s2;
        sctx c = {kern, kind, tj, sd, s1, s2, sgn, R, q, c0, a, bb, ZERO, ZERO, 0, 0, nc};
        if (!isfinite(bb)) {
            if (j == 0) { if (!wants) continue; } else { if (!WANT(j - 1)) continue; }
            cdouble alpha = cadd(tj, cscale(c0, sd)), beta = cadd(cscale(2 * a, alpha), cscale(q, sd));
            cdouble cst = cexp(csub(CPX(-creal(tj) * a * a, -cimag(tj) * a * a), cmul(sd, xat(R, ZERO, a))));
            actx ax = {&c, a};
            double lv = lX / fmax(fabs(dXv(R, a)), 1e-300);
            if (c0 != 0) lv = fmin(lv, sqrt(lX / fabs(c0)));
            lv = fmin(lv, R->ell0 / 16);
            int p = 1; cdouble val;
            if (nc && nc->nc) {
                double first = lv;
                if (j > 0 && R->kindb[j - 1] == 0) first = R->ell[j - 1];
                else if (j > 0) { lv = fmin(lv, R->ell[j - 1] / 16); first = lv; p = 4; }
                val = cmul(cst, rayexp_nc(&c, a, alpha, beta, first, p));
            } else if (j > 0 && R->kindb[j - 1] == 0) {
                cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
                double Rr = (-creal(be) + sqrt(creal(be) * creal(be) + 4 * aa * EXPO)) / (2 * aa);
                pctx pc = {&ax, e, be, aa};
                nodes_t Nd = {NULL, NULL, 0, 0};
                pieces_nodes(pseudo_E, &pc, Rr, fmin(Rr / 8, R->ell[j - 1]), 1, 48, &Nd);
                val = ZERO;
                for (int k = 0; k < Nd.n; k++) val = cadd(val, cscale(Nd.w[k], cmul(A_ray(cscale(Nd.y[k], e), &ax), cexp(pseudo_E(Nd.y[k], &pc)))));
                val = cmul(cmul(val, cst), e);
                free(Nd.y); free(Nd.w);
            } else {
                if (j > 0) { lv = fmin(lv, R->ell[j - 1] / 16); p = 4; }
                val = cmul(cst, rayexp(A_ray, &ax, alpha, beta, lv, p));
            }
            if (j == 0) *Is = cadd(*Is, val); else Ie[j - 1] = cadd(Ie[j - 1], val);
        } else {
            double L = bb - a, h = L / 2, kap = KAPPA / h, m = a + h, dep = L / 8;
            if (j == 0 && R->vsad > 0 && R->vsad < bb) {
                m = fmax(m, (R->vsad + bb) / 2); kap = fmax(kap, KAPPA / (bb - m));
                kap = fmin(kap, 4 * KAPPA / L); m = fmin(m, bb - KAPPA / kap);
                dep = fmin(dep, KAPPA / 4 / kap);
            }
            c.kap = kap; c.m = m;
            int pa = j > 0 ? 4 : 1;
            bool ws_ = j == 0 && wants, wl_ = j > 0 && WANT(j - 1), wr_ = WANT(j);
            if (!(ws_ || wl_ || wr_)) continue;
            double ella = j > 0 ? R->ell[j - 1] : R->ell0; int ka = j > 0 ? R->kindb[j - 1] : 1;
            cdouble vs_, vl_, vr_;
            segint(&c, pa, 1 + 3 * R->kindb[j], ella, R->ell[j], ws_, wl_, wr_, sg, ka, R->kindb[j], dep, &vs_, &vl_, &vr_);
            if (ws_) *Is = cadd(*Is, vs_);
            if (wl_) Ie[j - 1] = cadd(Ie[j - 1], vl_);
            if (wr_) Ie[j] = cadd(Ie[j], vr_);
        }
    }
#undef WANT
}

/* the saddle of the exponent on the first segment at t = tau + i */
static double saddle_loc(int model, double c, double q, double c0, double tau, bool stat, int kern, const breaks_t *R)
{
    if (R->ne == 0) return NAN;
    cdouble ts[2] = {CPX(tau, 0), CPX(tau, 1)}, P[2], s1[2], s2[2], M0[2];
    dfl_tparts(model, c, 2, ts, stat, P, s1, s2, M0);
    cdouble sd = (kern == 1 || R->sgn[0] < 0) ? s1[1] : s2[1];
    cdouble vs = cdiv(cscale(-q, sd), cscale(2, cadd(CPX(tau, 1), cscale(c0, sd))));
    return (creal(vs) > 0 && creal(vs) < R->b[0] && fabs(cimag(vs)) < 0.2 * (R->b[0] - creal(vs))) ? creal(vs) : NAN;
}
static void saddle_phase_gen(int kern, double q, double c0, int n, const cdouble *ts, const cdouble *s1, const cdouble *s2, const breaks_t *R, double *psi)
{
    for (int j = 0; j < n; j++) {
        cdouble sd = (kern == 1 || R->sgn[0] < 0) ? s1[j] : s2[j];
        cdouble alpha = cadd(ts[j], cscale(c0, sd)), beta = cscale(q, sd), z2 = cdiv(cmul(beta, beta), cscale(4, alpha));
        psi[j] = NAN;
        if (R->ne == 0) { cdouble z = cdiv(beta, cscale(2, csqrt(alpha))); if (creal(z) < 0 && creal(z2) > -30) psi[j] = cimag(z2); }
        else if (!isnan(R->vsad)) { cdouble vs = cdiv(cneg(beta), cscale(2, alpha)); if (creal(vs) > 0 && creal(vs) < R->b[0] && creal(z2) > -30) psi[j] = cimag(z2); }
    }
}
static double max_step(int n, const double *psi) { double m = 0; for (int j = 0; j + 1 < n; j++) if (!isnan(psi[j]) && !isnan(psi[j + 1])) m = fmax(m, fabs(psi[j + 1] - psi[j])); return m; }

/* ---- the outer integral for the general kernel ---- */
typedef struct { int kern, kind, part, k; const breaks_t *R; const cdouble *ts, *s1, *s2, *M0, *P; const double *wt; double *g; cdouble *G; double y;
                 int model; double c, g0; } gctx;
/* the transforms at the node j (ts[j]), with the noncentral kernel of a fixed start when g0 != 0 */
static void node_inner(const gctx *o, int j, int part, cdouble *Is, cdouble *Ie)
{
    if (o->g0 == 0) { inner_gen(o->kern, NULL, o->ts[j], o->R, o->s1[j], o->s2[j], o->M0[j], o->kind, part, Is, Ie); return; }
    nck_t K; nckernel(o->model, o->c, o->ts[j], o->g0, o->s1[j], o->s2[j], &K);
    inner_gen(o->kern, &K, o->ts[j], o->R, o->s1[j], o->s2[j], cmul(K.pref, o->M0[j]), o->kind, part, Is, Ie);
}
static void gsmooth_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    node_inner(o, j, 1, &Is, Ie);
    cdouble val = cmul(o->P[j], Is); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val) * o->wt[j]; }
static void gedge_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2], t = o->ts[j + 1];
    node_inner(o, j + 1, 10 + o->k + 1, &Is, Ie);
    cdouble val = cmul(o->P[j + 1], Ie[o->k]); if (o->kind == 2) val = cdiv(val, t); o->G[j] = cmul(val, cexp(cscale(o->y, t))); }
static void guni_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    node_inner(o, j, 0, &Is, Ie);
    cdouble val = cmul(o->P[j], cadd(Is, cadd(Ie[0], Ie[1]))); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val); }

static double outer_gen(int kind, int model, double c, double q, double c0, double g0, breaks_t *R, double tau, bool stat, const dfl_single_opts *o)
{
    int kern = ((model == 1 || model == 4) && !stat) ? 1 : 2; double hu = o->hu, cde = tau; int N = o->N;
    double umax = asinh(WMAX / cde); int nu = (int) floor(umax / hu) + 1;
    cdouble *ts = malloc(nu * sizeof(cdouble)), *P = malloc(nu * sizeof(cdouble)), *s1 = malloc(nu * sizeof(cdouble)), *s2 = malloc(nu * sizeof(cdouble)), *M0 = malloc(nu * sizeof(cdouble));
    double *wt = malloc(nu * sizeof(double)), *g = malloc(nu * sizeof(double)), *psi = malloc(nu * sizeof(double));
    for (int j = 0; j < nu; j++) { double u = j * hu; ts[j] = CPX(tau, cde * sinh(u)); wt[j] = cde * cosh(u); }
    dfl_tparts(model, c, nu, ts, stat, P, s1, s2, M0);
    saddle_phase_gen(kern, q, c0, nu, ts, s1, s2, R, psi);
    double val;
    if (max_step(nu, psi) > 0.25) {
        double w0 = fmin(0.5, tau / 2); int ne = (int) ceil(log2(WMAX / w0)) + 2;
        double *edges0 = malloc(ne * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne - 1] = WMAX;
        int K = 8, nk = (ne - 1) * K + 1;
        cdouble *te0 = malloc(nk * sizeof(cdouble)), *P0 = malloc(nk * sizeof(cdouble)), *s1e = malloc(nk * sizeof(cdouble)), *s2e = malloc(nk * sizeof(cdouble)), *M0e = malloc(nk * sizeof(cdouble));
        for (int p = 0; p < ne - 1; p++) for (int k = 0; k < K; k++) te0[p * K + k] = CPX(tau, edges0[p] + (edges0[p + 1] - edges0[p]) * k / K);
        te0[nk - 1] = CPX(tau, WMAX);
        dfl_tparts(model, c, nk, te0, stat, P0, s1e, s2e, M0e);
        double *psie = malloc(nk * sizeof(double)); saddle_phase_gen(kern, q, c0, nk, te0, s1e, s2e, R, psie);
        int cap_n = 0; int *mm = malloc(ne * sizeof(int)), *nn_ = malloc(ne * sizeof(int));
        for (int p = 0; p < ne - 1; p++) { double dp = 0; for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            mm[p] = (int) fmax(1, ceil(dp / 50)); nn_[p] = (int) fmax(24, 16 * ceil(ceil(3 * dp / mm[p]) / 16)) * DFL_NODEFAC; cap_n += mm[p] * nn_[p]; }
        cdouble *tn = malloc((cap_n + 1) * sizeof(cdouble)); double *wn = malloc(cap_n * sizeof(double)); int nn = 0;
        for (int p = 0; p < ne - 1; p++) {
            double a = edges0[p], b = edges0[p + 1]; const double *gw; const double *u = dfl_rule(nn_[p], &gw);
            for (int k = 0; k < mm[p]; k++) { double aa = a + (b - a) * k / mm[p], bb = a + (b - a) * (k + 1) / mm[p];
                for (int i = 0; i < nn_[p]; i++) { tn[nn + 1] = CPX(tau, aa + (bb - aa) * u[i]); wn[nn] = (bb - aa) * gw[i]; nn++; } }
        }
        tn[0] = CPX(tau, 0);
        cdouble *Pn = malloc((nn + 1) * sizeof(cdouble)), *s1n = malloc((nn + 1) * sizeof(cdouble)), *s2n = malloc((nn + 1) * sizeof(cdouble)), *M0n = malloc((nn + 1) * sizeof(cdouble));
        dfl_tparts(model, c, nn + 1, tn, stat, Pn, s1n, s2n, M0n);
        double *gn = malloc(nn * sizeof(double));
        gctx on = {kern, kind, 1, 0, R, tn + 1, s1n + 1, s2n + 1, M0n + 1, Pn + 1, wn, gn, NULL, 0, model, c, g0};
        dfl_parfor(nn, gsmooth_body, &on);
        val = 0; for (int j = 0; j < nn; j++) val += gn[j]; val /= M_PI;
        if (getenv("DFL_DEBUG")) { int j0 = 0; for (int p = 0; p < ne - 1; p++) { double sp = 0, dp = 0; for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            for (int k = 0; k < mm[p] * nn_[p]; k++) sp += gn[j0++]; printf("    panel %d [%.4g, %.4g] phase %.3g sub %d nodes %d psi0 %.3g sum %.15g\n", p, edges0[p], edges0[p + 1], dp, mm[p], nn_[p], psie[p * K], sp / M_PI); } }
        free(edges0); free(te0); free(P0); free(s1e); free(s2e); free(M0e); free(psie); free(mm); free(nn_); free(tn); free(wn); free(Pn); free(s1n); free(s2n); free(M0n); free(gn);
    } else {
        gctx oc = {kern, kind, 1, 0, R, ts, s1, s2, M0, P, wt, g, NULL, 0, model, c, g0};
        dfl_parfor(nu, gsmooth_body, &oc);
        g[0] *= 0.5; val = 0; for (int j = 0; j < nu; j++) val += g[j]; val *= hu / M_PI;
    }
    free(ts); free(P); free(s1); free(s2); free(M0); free(wt); free(g); free(psi);
    if (getenv("DFL_DEBUG")) printf("  smooth part %.15g (nu %d)\n", val, nu);
    /* one Filon part per breakpoint */
    const double *gw; const double *u = dfl_rule(N, &gw);
    for (int k = 0; k < R->ne; k++) {
        if (R->negl[k]) continue;
        double y = R->y[k], w0 = fmin(fmin(0.5, 1 / y), tau / 2);
        int ne0 = (int) ceil(log2(WMAX / w0)) + 2;
        double *edges0 = malloc(ne0 * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne0; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne0 - 1] = WMAX;
        int cap_e = 1; for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; cap_e += b <= DFL_FILONW / y ? (int) ceil((b - a) * y / 25) : 1; }
        double *orig = malloc((cap_e + 1) * sizeof(double)); int no = 0; orig[0] = 0;
        for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; int m = b <= DFL_FILONW / y ? (int) ceil((b - a) * y / 25) : 1; for (int i = 1; i <= m; i++) orig[++no] = a + (b - a) * i / m; }
        int pf = o->pfac > 1 ? o->pfac : 1; int np = no * pf;
        double *edges = malloc((np + 1) * sizeof(double)); edges[0] = 0;
        for (int p = 0; p < no; p++) for (int r = 1; r <= pf; r++) edges[p * pf + r] = orig[p] + (orig[p + 1] - orig[p]) * r / pf;
        int nw = np * N;
        cdouble *te = malloc((nw + 1) * sizeof(cdouble)); te[0] = CPX(tau, 0);
        for (int p = 0; p < np; p++) for (int i = 0; i < N; i++) te[1 + p * N + i] = CPX(tau, edges[p] + (edges[p + 1] - edges[p]) * u[i]);
        cdouble *Pe = malloc((nw + 1) * sizeof(cdouble)), *s1e = malloc((nw + 1) * sizeof(cdouble)), *s2e = malloc((nw + 1) * sizeof(cdouble)), *M0e = malloc((nw + 1) * sizeof(cdouble));
        dfl_tparts(model, c, nw + 1, te, stat, Pe, s1e, s2e, M0e);
        cdouble *G = malloc(nw * sizeof(cdouble));
        gctx oe = {kern, kind, 2, k, R, te, s1e, s2e, M0e, Pe, NULL, NULL, G, y, model, c, g0};
        dfl_parfor(nw, gedge_body, &oe);
        cdouble acc = ZERO;
        for (int p = 0; p < np; p++) { cdouble fp = dfl_filon_panel(G + p * N, edges[p], edges[p + 1], y, N); acc = cadd(acc, fp);
            if (getenv("DFL_DEBUG2")) printf("    panel %d [%.6g, %.6g] %.15g%+.15gi  G0 %.15g%+.15gi\n", p, edges[p], edges[p + 1], creal(fp), cimag(fp), creal(G[p * N]), cimag(G[p * N])); }
        if (getenv("DFL_DEBUG")) printf("  edge part %d: y %.15g panels %d nodes %d value %.15g\n", k, y, np, nw, creal(cmul(cexp(CPX(-tau * y, 0)), acc)) / M_PI);
        val += creal(cmul(cexp(CPX(-tau * y, 0)), acc)) / M_PI;
        free(edges0); free(orig); free(edges); free(te); free(Pe); free(s1e); free(s2e); free(M0e); free(G);
    }
    return val;
}

/* the value of the density (kind 1) or the distribution function (kind 2) through the general kernel (c0 != 0, a fixed
 * start g0 != 0, or gen); in case DK with c0 > 0 the density is infinite at x = -sqrt(2 c0 (1 + g0^2)) */
double dfl_single_value_gen(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o)
{
    double q = x == 0 ? 1e-12 : x;
    double tau = o->tau > 0 ? o->tau : dfl_default_tau(model, c, stat);
    DFL_NODEFAC = o->nodefac > 0 ? o->nodefac : 1;
    int kern = ((model == 1 || model == 4) && !stat) ? 1 : 2;
    if (stat) g0 = 0;
    double x0 = (g0 != 0 && model == 1) ? (1 + g0 * g0) / 2 : 0.5;
    breaks_t R; breakpts(q, c0, tau, x0, &R);
    if (R.dbl && kern == 1 && kind == 1) return INFINITY;
    R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    if (o->uniform_hw > 0) {
        double hw = o->uniform_hw; int n = (int) floor(WMAX / hw) + 1;
        cdouble *t = malloc(n * sizeof(cdouble)), *P = malloc(n * sizeof(cdouble)), *s1 = malloc(n * sizeof(cdouble)), *s2 = malloc(n * sizeof(cdouble)), *M0 = malloc(n * sizeof(cdouble));
        double *g = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) t[j] = CPX(tau, j * hw);
        dfl_tparts(model, c, n, t, stat, P, s1, s2, M0);
        gctx u = {kern, kind, 0, 0, &R, t, s1, s2, M0, P, NULL, g, NULL, 0, model, c, g0};
        dfl_parfor(n, guni_body, &u);
        g[0] *= 0.5; double v = 0; for (int j = 0; j < n; j++) v += g[j]; v *= hw / M_PI;
        free(t); free(P); free(s1); free(s2); free(M0); free(g);
        return v;
    }
    return outer_gen(kind, model, c, q, c0, g0, &R, tau, stat, o);
}

/* debugging aid: the breakpoints and the transforms at one node */
void dfl_debug_inner_gen(int model, double c, double c0, double x, bool stat, int kind, double tre, double tim, int part)
{
    double q = x == 0 ? 1e-12 : x, tau = dfl_default_tau(model, c, stat);
    int kern = ((model == 1 || model == 4) && !stat) ? 1 : 2;
    breaks_t R; breakpts(q, c0, tau, 0.5, &R); R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    printf("tau %.15g ne %d b %.15g %.15g y %.15g %.15g ell %.15g %.15g kindb %d %d ell0 %.15g sgn %d %d %d negl %d %d dbl %d vsad %.15g\n",
           tau, R.ne, R.b[0], R.b[1], R.y[0], R.y[1], R.ell[0], R.ell[1], R.kindb[0], R.kindb[1], R.ell0, R.sgn[0], R.sgn[1], R.sgn[2], R.negl[0], R.negl[1], R.dbl, R.vsad);
    cdouble ts[2] = {CPX(tau, 0), CPX(tre, tim)}, P[2], s1[2], s2[2], M0[2];
    dfl_tparts(model, c, 2, ts, stat, P, s1, s2, M0);
    cdouble Is, Ie[2];
    inner_gen(kern, NULL, ts[1], &R, s1[1], s2[1], M0[1], kind, part, &Is, Ie);
    printf("part %d t %.15g%+.15gi s1 %.15g%+.15gi s2 %.15g%+.15gi M0 %.15g%+.15gi\nIs %.15g%+.15gi Ie1 %.15g%+.15gi Ie2 %.15g%+.15gi\n",
           part, creal(ts[1]), cimag(ts[1]), creal(s1[1]), cimag(s1[1]), creal(s2[1]), cimag(s2[1]), creal(M0[1]), cimag(M0[1]),
           creal(Is), cimag(Is), creal(Ie[0]), cimag(Ie[0]), creal(Ie[1]), cimag(Ie[1]));
}
