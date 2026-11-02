/* ltratio_mex.c: MATLAB gateway to the C engine (c/), one MEX file for every operation of the MATLAB programs; the
 * m-files dfx_core.m, tr_limdist.m and tr_limmom.m next to it present the engine under the names and signatures of
 * the MATLAB engines, so that ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom and ltratio_test run unchanged on
 * top of it, and ltratio.m is a self-contained wrapper.  All arguments are double arrays (logical flags as 0/1):
 *
 *   F = ltratio_mex('single', kind, x, model, c, c0, stat, sopt)        the single integral; kind 1 density, 2 cdf
 *   [F, err, status, nbad, nunres, nrefine] = ltratio_mex('double', kind, x, model, c, c0, g0, stat, dopt)
 *   F = ltratio_mex('dist', kind, x, model, c, c0, g0, stat, method)    method 0 automatic, 1 single, 2 double
 *   [q, F, iter, conv] = ltratio_mex('icdf', p, model, c, c0, g0, stat, method, tol)
 *   [m, err] = ltratio_mex('mom', p, model, c, c0, g0, stat)             the moments 1..p
 *   ts = ltratio_mex('tstar', model, c, stat)                            the end of the domain of the mgf of V
 *   n = ltratio_mex('threads'[, n])                                      the number of threads (sets it with n)
 *
 * model 1, 2, 3 (no deterministic terms, intercept, intercept and trend); sopt = [tau hu N nodefac pfac edge_hw
 * uniform_hw gen] and dopt = [M Nv Nl maxlev plen otol stepw errtol blo bhi bmin mumin tdec] (NaN or a shorter
 * vector: the defaults).  Compile with -DDFLIMIT_STUB where MATLAB's headers are not available (mexstub.h). */
#include "../dflimit.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef DFLIMIT_STUB
#include "mexstub.h"
DFLIMIT_DEFINE_APIVERSION
#else
#include "mex.h"
#endif

static double scalar(const mxArray *a, const char *name)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("ltratio:args", "%s must be a real scalar", name);
    return mxGetScalar(a);
}
static int intarg(const mxArray *a, const char *name) { double v = scalar(a, name); if (v != floor(v)) mexErrMsgIdAndTxt("ltratio:args", "%s must be an integer", name); return (int) v; }
static const mxArray *vecarg(const mxArray *a, const char *name, int *n)
{
    if (!mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("ltratio:args", "%s must be a real array", name);
    *n = (int) mxGetNumberOfElements(a); return a;
}
static mxArray *like(const mxArray *a) { return mxCreateDoubleMatrix(mxGetM(a), mxGetN(a), mxREAL); }
/* an optional vector of option values, NaN meaning the default */
static void optvec(const mxArray *a, double *v, int n) { for (int k = 0; k < n; k++) v[k] = NAN; if (a == NULL) return; int m = (int) mxGetNumberOfElements(a); if (!mxIsDouble(a)) mexErrMsgIdAndTxt("ltratio:args", "the option vector must be double"); const double *p = mxGetPr(a); for (int k = 0; k < n && k < m; k++) v[k] = p[k]; }
#define SET(field, v) do { if (!isnan(v)) o.field = (v); } while (0)
static dfl_single_opts sopts(const mxArray *a)
{
    dfl_single_opts o; dfl_single_defaults(&o); double v[8]; optvec(a, v, 8);
    SET(tau, v[0]); SET(hu, v[1]); if (!isnan(v[2])) o.N = (int) v[2]; if (!isnan(v[3])) o.nodefac = (int) v[3]; if (!isnan(v[4])) o.pfac = (int) v[4];
    SET(edge_hw, v[5]); SET(uniform_hw, v[6]); if (!isnan(v[7])) o.gen = v[7] != 0;
    return o;
}
static dfl_double_opts dopts(const mxArray *a)
{
    dfl_double_opts o; dfl_double_defaults(&o); double v[13]; optvec(a, v, 13);
    if (!isnan(v[0])) o.M = (int) v[0]; if (!isnan(v[1])) o.Nv = (int) v[1]; if (!isnan(v[2])) o.Nl = (int) v[2]; if (!isnan(v[3])) o.maxlev = (int) v[3];
    SET(plen, v[4]); SET(otol, v[5]); SET(stepw, v[6]); SET(errtol, v[7]); SET(blo, v[8]); SET(bhi, v[9]); SET(bmin, v[10]); SET(mumin, v[11]); SET(tdec, v[12]);
    return o;
}
#undef SET
static void need(int nrhs, int n, const char *usage) { if (nrhs < n) mexErrMsgIdAndTxt("ltratio:args", "usage: %s", usage); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char op[16], err[512] = "";
    if (nrhs < 1 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("ltratio:args", "the first argument is the operation: single, double, dist, icdf, mom, tstar or threads");
    mxGetString(prhs[0], op, sizeof op);
    if (strcmp(op, "threads") == 0) {
        if (nrhs > 1) dfl_set_nworkers(intarg(prhs[1], "n"));
        plhs[0] = mxCreateDoubleScalar(dfl_nworkers()); return;
    }
    if (strcmp(op, "tstar") == 0) {
        need(nrhs, 3, "ltratio_mex('tstar', model, c, stat)");
        plhs[0] = mxCreateDoubleScalar(dfl_tstar(intarg(prhs[1], "model"), scalar(prhs[2], "c"), nrhs > 3 && scalar(prhs[3], "stat") != 0)); return;
    }
    if (strcmp(op, "mom") == 0) {
        need(nrhs, 3, "ltratio_mex('mom', p, model, c, c0, g0, stat)");
        int p = intarg(prhs[1], "p"); if (p < 1) mexErrMsgIdAndTxt("ltratio:args", "p must be a positive integer");
        plhs[0] = mxCreateDoubleMatrix(1, p, mxREAL); mxArray *e = mxCreateDoubleMatrix(1, p, mxREAL);
        int rc = dfl_moments(intarg(prhs[2], "model"), scalar(prhs[3], "c"), nrhs > 4 ? scalar(prhs[4], "c0") : 0.0, nrhs > 5 ? scalar(prhs[5], "g0") : 0.0,
                             nrhs > 6 && scalar(prhs[6], "stat") != 0, p, mxGetPr(plhs[0]), mxGetPr(e), err, sizeof err);
        if (rc) mexErrMsgIdAndTxt("ltratio:engine", "%s", err);
        if (nlhs > 1) plhs[1] = e; return;
    }
    if (strcmp(op, "single") == 0) {
        need(nrhs, 5, "ltratio_mex('single', kind, x, model, c, c0, stat, sopt)");
        int n; const mxArray *xa = vecarg(prhs[2], "x", &n); plhs[0] = like(xa);
        dfl_single_opts o = sopts(nrhs > 7 ? prhs[7] : NULL);
        int rc = dfl_single(intarg(prhs[1], "kind"), n, mxGetPr(xa), intarg(prhs[3], "model"), scalar(prhs[4], "c"), nrhs > 5 ? scalar(prhs[5], "c0") : 0.0,
                            nrhs > 6 && scalar(prhs[6], "stat") != 0, &o, mxGetPr(plhs[0]), err, sizeof err);
        if (rc) mexErrMsgIdAndTxt("ltratio:engine", "%s", err);
        return;
    }
    if (strcmp(op, "double") == 0) {
        need(nrhs, 5, "ltratio_mex('double', kind, x, model, c, c0, g0, stat, dopt)");
        int n; const mxArray *xa = vecarg(prhs[2], "x", &n); plhs[0] = like(xa);
        dfl_double_opts o = dopts(nrhs > 8 ? prhs[8] : NULL);
        dfl_double_info *info = malloc((n + 1) * sizeof(dfl_double_info));
        int kind = intarg(prhs[1], "kind"); if (kind != 1 && kind != 2) mexErrMsgIdAndTxt("ltratio:args", "kind must be 1 (density) or 2 (distribution function)");
        int model = intarg(prhs[3], "model"); double c = scalar(prhs[4], "c"), c0 = nrhs > 5 ? scalar(prhs[5], "c0") : 0.0, g0 = nrhs > 6 ? scalar(prhs[6], "g0") : 0.0;
        bool stat = nrhs > 7 && scalar(prhs[7], "stat") != 0;
        if (model < 1 || model > 3) mexErrMsgIdAndTxt("ltratio:args", "model must be 1, 2 or 3");
        if (stat && !(c < 0)) mexErrMsgIdAndTxt("ltratio:args", "a stationary start requires c < 0");
        /* finite x only: the m-file handles the rest */
        const double *x = mxGetPr(xa); for (int k = 0; k < n; k++) if (!isfinite(x[k])) mexErrMsgIdAndTxt("ltratio:args", "x must be finite");
        int rc = n > 0 ? dfl_double(kind, n, x, model, c, c0, g0, stat, &o, mxGetPr(plhs[0]), info, err, sizeof err) : 0;
        if (rc) { free(info); mexErrMsgIdAndTxt("ltratio:engine", "%s", err); }
        for (int j = 1; j < 6 && j < nlhs; j++) {
            plhs[j] = like(xa); double *p = mxGetPr(plhs[j]);
            for (int k = 0; k < n; k++) p[k] = j == 1 ? info[k].err : j == 2 ? info[k].status : j == 3 ? info[k].nbad : j == 4 ? info[k].nunres : info[k].nrefine;
        }
        free(info); return;
    }
    if (strcmp(op, "dist") == 0) {
        need(nrhs, 5, "ltratio_mex('dist', kind, x, model, c, c0, g0, stat, method)");
        int n; const mxArray *xa = vecarg(prhs[2], "x", &n); plhs[0] = like(xa); int ns = 0;
        int rc = dfl_dist(intarg(prhs[1], "kind"), n, mxGetPr(xa), intarg(prhs[3], "model"), scalar(prhs[4], "c"), nrhs > 5 ? scalar(prhs[5], "c0") : 0.0,
                          nrhs > 6 ? scalar(prhs[6], "g0") : 0.0, nrhs > 7 && scalar(prhs[7], "stat") != 0, nrhs > 8 ? intarg(prhs[8], "method") : 0,
                          NULL, NULL, mxGetPr(plhs[0]), &ns, err, sizeof err);
        if (rc) mexErrMsgIdAndTxt("ltratio:engine", "%s", err);
        if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(ns);
        return;
    }
    if (strcmp(op, "icdf") == 0) {
        need(nrhs, 4, "ltratio_mex('icdf', p, model, c, c0, g0, stat, method, tol)");
        int n; const mxArray *pa = vecarg(prhs[1], "p", &n); plhs[0] = like(pa);
        mxArray *Fq = like(pa), *it = like(pa), *cv = like(pa); int *iter = malloc((n + 1) * sizeof(int)), *conv = malloc((n + 1) * sizeof(int));
        int rc = dfl_icdf(n, mxGetPr(pa), intarg(prhs[2], "model"), scalar(prhs[3], "c"), nrhs > 4 ? scalar(prhs[4], "c0") : 0.0, nrhs > 5 ? scalar(prhs[5], "g0") : 0.0,
                          nrhs > 6 && scalar(prhs[6], "stat") != 0, nrhs > 7 ? intarg(prhs[7], "method") : 0, nrhs > 8 ? scalar(prhs[8], "tol") : 1e-10,
                          NULL, NULL, mxGetPr(plhs[0]), mxGetPr(Fq), iter, conv, err, sizeof err);
        for (int k = 0; k < n; k++) { mxGetPr(it)[k] = iter[k]; mxGetPr(cv)[k] = conv[k]; }
        free(iter); free(conv);
        if (rc) mexErrMsgIdAndTxt("ltratio:engine", "%s", err);
        if (nlhs > 1) plhs[1] = Fq; if (nlhs > 2) plhs[2] = it; if (nlhs > 3) plhs[3] = cv;
        return;
    }
    mexErrMsgIdAndTxt("ltratio:args", "unknown operation %s", op);
}
