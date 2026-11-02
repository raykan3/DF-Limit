/* dfl_gen.c: the single integral with a local null value c0 != 0: the kernel runs along the parabola
 * X(v) = c0 v^2 + x v + 1/2, with one Filon part of the outer integral per positive root of X (and the vertex as a
 * pseudo-edge for c0 > 0), the segments between breakpoints on three-piece steepest-descent paths.  A port of
 * breakpts / inner_gen / segint / pieces_nodes / phase_nodes / descent / outer_gen of DFExact.jl (Raymond Kan's
 * extension of the MATLAB port). */
#include "dflimit.h"
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
    double q, c0, lead;
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
static void breakpts(double q, double c0, double tau, breaks_t *R)
{
    memset(R, 0, sizeof *R);
    R->q = q; R->c0 = c0;
    double r[2]; int nrr = 0;
    if (c0 == 0) { R->rr[0] = CPX(-1 / (2 * q), 0); R->nr = 1; R->lead = q; r[0] = -1 / (2 * q); nrr = 1; }
    else {
        double d = q * q - 2 * c0;
        cdouble sq = d >= 0 ? CPX(sqrt(d), 0) : CPX(0, sqrt(-d));
        R->rr[0] = cdiv(csub(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->rr[1] = cdiv(cadd(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->nr = 2; R->lead = c0;
        if (d >= 0) { r[0] = creal(R->rr[0]); r[1] = creal(R->rr[1]); if (r[0] > r[1]) { double t = r[0]; r[0] = r[1]; r[1] = t; } nrr = 2; }
    }
    double b[2]; int nb = 0;
    for (int k = 0; k < nrr; k++) if (r[k] > 0) b[nb++] = r[k];
    bool noroot = nb == 0;
    int nb2 = 0; for (int k = 0; k < nb; k++) if (3 * tau * b[k] * b[k] / 4 <= 40) b[nb2++] = b[k]; nb = nb2;
    int kindb[2] = {1, 1}; double ell[2] = {INFINITY, INFINITY};
    if (nb == 2 && b[1] - b[0] <= 1e-9 * b[1]) { b[0] = (b[0] + b[1]) / 2; nb = 1; R->dbl = true; }
    else if (nb == 2) ell[0] = ell[1] = b[1] - b[0];
    if (c0 > 0 && q < 0 && noroot) { b[0] = -q / (2 * c0); nb = 1; kindb[0] = 0; ell[0] = sqrt(fmax(0.5 - q * q / (4 * c0), 0) / c0); }
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

static cdouble phialg(int kern, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (kern == 1) return cdiv(ONE, csqrt(cscale(M_PI, X)));
    cdouble arg = cmul(cscale(0.5, csub(s2, s1)), X);
    return cscale(1 / M_PI, dfl_k0x(sgn > 0 ? arg : cneg(arg)));
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
/* nodes for int_0^L: the first piece shortened until the exponent falls by at most 4 and its phase by at most 40 rad,
 * then pieces growing by 4 with node counts from the change of the exponent, until it is below -EXPO */
static void pieces_nodes(efn_t Ef, const void *ctx, double L, double first, int p, int nmin, nodes_t *N)
{
    N->n = 0;
    first = fmin(first, L);
    for (int it = 0; it < 60; it++) {
        cdouble E[9]; for (int i = 0; i <= 8; i++) E[i] = Ef(first * i / 8, ctx);
        double dph = 0; for (int i = 0; i < 8; i++) dph += fabs(cimag(E[i + 1]) - cimag(E[i]));
        if ((creal(E[8]) > -4 && dph < 40) || first <= 1e-12 * L) break;
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
} sctx;
static cdouble W_of(const sctx *c, cdouble v) { return c->kind == 1 ? cscale(2, cmul(v, v)) : cadd(CPX(c->q, 0), cscale(2 * c->c0, v)); }
static cdouble Ez(const sctx *c, cdouble z, double o) { cdouble v = cadd(CPX(o, 0), z); return csub(cneg(cmul(c->tj, cmul(v, v))), cmul(c->sd, xat(c->R, z, o))); }
static cdouble Fz(const sctx *c, cdouble z, double o) { cdouble X = xat(c->R, z, o); return cmul(cmul(W_of(c, cadd(CPX(o, 0), z)), cexp(Ez(c, z, o))), phialg(c->kern, X, c->sgn, c->s1, c->s2)); }
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
    descent(csub(cscale(-2 * a, c->tj), cscale(dXv(c->R, a), c->sd)), alpha, 1, L, dep, sg, dega, &ua, &la);
    descent(csub(cscale(-2 * b, c->tj), cscale(dXv(c->R, b), c->sd)), alpha, -1, L, dep, sg, degb, &ub, &lb);
    cdouble pa_ = cadd(CPX(a, 0), cscale(la, ua)), pb_ = cadd(CPX(b, 0), cscale(lb, ub));
    nodes_t Nd = {NULL, NULL, 0, 0};
    int capv = 4096, nv = 0; cdouble *fv = malloc(capv * sizeof(cdouble)), *vv = malloc(capv * sizeof(cdouble));
#define PUSHV(F, V) do { if (nv == capv) { capv *= 2; fv = realloc(fv, capv * sizeof(cdouble)); vv = realloc(vv, capv * sizeof(cdouble)); } fv[nv] = (F); vv[nv] = (V); nv++; } while (0)
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
    *vs = *vl = *vr = ZERO;
    if (ws || wl) { cdouble val = ZERO; for (int k = 0; k < nv; k++) val = cadd(val, cmul(fv[k], csub(ONE, chi_seg(c, vv[k])))); if (ws) *vs = val; else *vl = val; }
    if (wr) { cdouble val = ZERO; for (int k = 0; k < nv; k++) val = cadd(val, cmul(fv[k], chi_seg(c, vv[k]))); *vr = val; }
#undef PUSHV
    free(Nd.y); free(Nd.w); free(fv); free(vv);
}

/* ---- the transforms of the general kernel: Is and Ie[k] ---- */
typedef struct { const sctx *c; double a; } actx;
static cdouble A_ray(cdouble rho, const void *v) { const actx *x = v; return cmul(W_of(x->c, cadd(CPX(x->a, 0), rho)), phialg(x->c->kern, xat(x->c->R, rho, x->a), x->c->sgn, x->c->s1, x->c->s2)); }
typedef struct { const actx *ax; cdouble e, be; double aa; } pctx;
static cdouble pseudo_E(double r, const void *v) { const pctx *p = v; return csub(CPX(-p->aa * r * r, 0), cscale(r, p->be)); }
typedef struct { cfn_t A; const void *Actx; cdouble alpha, beta, e; double aa; } rxctx;
static cdouble rx_f(double r, const void *v) { const rxctx *x = v; cdouble arg = csub(CPX(-x->aa * r * r, 0), cscale(r, cmul(x->beta, x->e)));
    return cmul(cmul(x->A(cscale(r, x->e), x->Actx), cexp(arg)), x->e); }
/* int_0^inf A(r) exp(-alpha r^2 - beta r) dr: along the ray of the Gaussian's decay, or by gaussexp's paths */
cdouble dfl_rayint(cdouble (*f)(double, const void *), const void *ctx, double R, double l, int n1, int n2, int p, double osc);
static cdouble rayexp(cfn_t A, const void *Actx, cdouble alpha, cdouble beta, double lA, int p)
{
    cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha);
    if (dfl_ray_ok(beta, e, 1.0, aa)) {
        double Rr = sqrt(EXPO / aa); cdouble be = cmul(beta, e);
        if (creal(be) > 0) Rr = fmin(Rr, EXPO / creal(be));
        rxctx x = {A, Actx, alpha, beta, e, aa};
        return dfl_rayint(rx_f, &x, Rr, lA, 48, 32, p, fabs(cimag(be)));
    }
    return dfl_gaussexp(A, Actx, alpha, 1.0, beta, lA, 0, p);
}
static void inner_gen(int kern, cdouble tj, const breaks_t *R, cdouble s1, cdouble s2, cdouble M0, int kind, int part, cdouble *Is, cdouble *Ie)
{
    double q = R->q, c0 = R->c0; int ne = R->ne;
    double sg = cimag(tj) >= 0 ? 1.0 : -1.0;
    *Is = ZERO; for (int k = 0; k < 2; k++) Ie[k] = ZERO;
    bool wants = part == 0 || part == 1;
#define WANT(k) ((part == 0 || part == 2 || part == 10 + (k) + 1) && !R->negl[k])
    if (kind == 2 && wants) *Is = dfl_psi_half(kern, s1, s2, M0);
    double lX = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
    double ed[4] = {0, ne > 0 ? R->b[0] : INFINITY, ne > 1 ? R->b[1] : INFINITY, INFINITY};
    for (int j = 0; j <= ne; j++) {
        double a = ed[j], bb = ed[j + 1]; int sgn = R->sgn[j];
        if (kern == 1 && sgn < 0) continue;
        cdouble sd = (kern == 1 || sgn < 0) ? s1 : s2;
        sctx c = {kern, kind, tj, sd, s1, s2, sgn, R, q, c0, a, bb, ZERO, ZERO, 0, 0};
        if (!isfinite(bb)) {
            if (j == 0) { if (!wants) continue; } else { if (!WANT(j - 1)) continue; }
            cdouble alpha = cadd(tj, cscale(c0, sd)), beta = cadd(cscale(2 * a, alpha), cscale(q, sd));
            cdouble cst = cexp(csub(CPX(-creal(tj) * a * a, -cimag(tj) * a * a), cmul(sd, xat(R, ZERO, a))));
            actx ax = {&c, a};
            double lv = lX / fmax(fabs(dXv(R, a)), 1e-300);
            if (c0 != 0) lv = fmin(lv, sqrt(lX / fabs(c0)));
            lv = fmin(lv, R->ell0 / 16);
            int p = 1; cdouble val;
            if (j > 0 && R->kindb[j - 1] == 0) {
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
typedef struct { int kern, kind, part, k; const breaks_t *R; const cdouble *ts, *s1, *s2, *M0, *P; const double *wt; double *g; cdouble *G; double y; } gctx;
static void gsmooth_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    inner_gen(o->kern, o->ts[j], o->R, o->s1[j], o->s2[j], o->M0[j], o->kind, 1, &Is, Ie);
    cdouble val = cmul(o->P[j], Is); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val) * o->wt[j]; }
static void gedge_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2], t = o->ts[j + 1];
    inner_gen(o->kern, t, o->R, o->s1[j + 1], o->s2[j + 1], o->M0[j + 1], o->kind, 10 + o->k + 1, &Is, Ie);
    cdouble val = cmul(o->P[j + 1], Ie[o->k]); if (o->kind == 2) val = cdiv(val, t); o->G[j] = cmul(val, cexp(cscale(o->y, t))); }
static void guni_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    inner_gen(o->kern, o->ts[j], o->R, o->s1[j], o->s2[j], o->M0[j], o->kind, 0, &Is, Ie);
    cdouble val = cmul(o->P[j], cadd(Is, cadd(Ie[0], Ie[1]))); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val); }

static double outer_gen(int kind, int model, double c, double q, double c0, breaks_t *R, double tau, bool stat, const dfl_single_opts *o)
{
    int kern = (model == 1 && !stat) ? 1 : 2; double hu = o->hu, cde = tau; int N = o->N;
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
        gctx on = {kern, kind, 1, 0, R, tn + 1, s1n + 1, s2n + 1, M0n + 1, Pn + 1, wn, gn, NULL, 0};
        dfl_parfor(nn, gsmooth_body, &on);
        val = 0; for (int j = 0; j < nn; j++) val += gn[j]; val /= M_PI;
        free(edges0); free(te0); free(P0); free(s1e); free(s2e); free(M0e); free(psie); free(mm); free(nn_); free(tn); free(wn); free(Pn); free(s1n); free(s2n); free(M0n); free(gn);
    } else {
        gctx oc = {kern, kind, 1, 0, R, ts, s1, s2, M0, P, wt, g, NULL, 0};
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
        gctx oe = {kern, kind, 2, k, R, te, s1e, s2e, M0e, Pe, NULL, NULL, G, y};
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

/* the value of the density (kind 1) or the distribution function (kind 2) through the general kernel */
double dfl_single_value_gen(int kind, double x, int model, double c, double c0, bool stat, const dfl_single_opts *o)
{
    double q = x == 0 ? 1e-12 : x;
    double tau = o->tau > 0 ? o->tau : dfl_default_tau(model, c, stat);
    DFL_NODEFAC = o->nodefac > 0 ? o->nodefac : 1;
    int kern = (model == 1 && !stat) ? 1 : 2;
    breaks_t R; breakpts(q, c0, tau, &R);
    if (R.dbl && kern == 1 && kind == 1) return INFINITY;
    R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    if (o->uniform_hw > 0) {
        double hw = o->uniform_hw; int n = (int) floor(WMAX / hw) + 1;
        cdouble *t = malloc(n * sizeof(cdouble)), *P = malloc(n * sizeof(cdouble)), *s1 = malloc(n * sizeof(cdouble)), *s2 = malloc(n * sizeof(cdouble)), *M0 = malloc(n * sizeof(cdouble));
        double *g = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) t[j] = CPX(tau, j * hw);
        dfl_tparts(model, c, n, t, stat, P, s1, s2, M0);
        gctx u = {kern, kind, 0, 0, &R, t, s1, s2, M0, P, NULL, g, NULL, 0};
        dfl_parfor(n, guni_body, &u);
        g[0] *= 0.5; double v = 0; for (int j = 0; j < n; j++) v += g[j]; v *= hw / M_PI;
        free(t); free(P); free(s1); free(s2); free(M0); free(g);
        return v;
    }
    return outer_gen(kind, model, c, q, c0, &R, tau, stat, o);
}

/* debugging aid: the breakpoints and the transforms at one node */
void dfl_debug_inner_gen(int model, double c, double c0, double x, bool stat, int kind, double tre, double tim, int part)
{
    double q = x == 0 ? 1e-12 : x, tau = dfl_default_tau(model, c, stat);
    int kern = (model == 1 && !stat) ? 1 : 2;
    breaks_t R; breakpts(q, c0, tau, &R); R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    printf("tau %.15g ne %d b %.15g %.15g y %.15g %.15g ell %.15g %.15g kindb %d %d ell0 %.15g sgn %d %d %d negl %d %d dbl %d vsad %.15g\n",
           tau, R.ne, R.b[0], R.b[1], R.y[0], R.y[1], R.ell[0], R.ell[1], R.kindb[0], R.kindb[1], R.ell0, R.sgn[0], R.sgn[1], R.sgn[2], R.negl[0], R.negl[1], R.dbl, R.vsad);
    cdouble ts[2] = {CPX(tau, 0), CPX(tre, tim)}, P[2], s1[2], s2[2], M0[2];
    dfl_tparts(model, c, 2, ts, stat, P, s1, s2, M0);
    cdouble Is, Ie[2];
    inner_gen(kern, ts[1], &R, s1[1], s2[1], M0[1], kind, part, &Is, Ie);
    printf("part %d t %.15g%+.15gi s1 %.15g%+.15gi s2 %.15g%+.15gi M0 %.15g%+.15gi\nIs %.15g%+.15gi Ie1 %.15g%+.15gi Ie2 %.15g%+.15gi\n",
           part, creal(ts[1]), cimag(ts[1]), creal(s1[1]), cimag(s1[1]), creal(s2[1]), cimag(s2[1]), creal(M0[1]), cimag(M0[1]),
           creal(Is), cimag(Is), creal(Ie[0]), cimag(Ie[0]), creal(Ie[1]), cimag(Ie[1]));
}
