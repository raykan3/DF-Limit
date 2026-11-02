/* dfl_vmath.h: exp and sincos on arrays, written as straight-line branch-free loops so that the compiler vectorizes
 * them (NEON on Apple silicon, AVX2/AVX-512 on x86 with -O3 -mavx2 -mfma, SSE2 otherwise; MSVC /O2 vectorizes them
 * too).  Accuracy about 1 ulp: exp by the reduction x = k ln 2 + r, |r| <= ln2/2, a degree-13 Taylor polynomial for
 * e^r and 2^k built from the exponent bits; sin and cos by the Cody-Waite reduction x = j pi/2 + r (three parts,
 * exact up to |x| ~ 1e5), |r| <= pi/4, Taylor polynomials of degree 15 and 16, and the quadrant from j mod 4.
 * The inputs are clamped to [-745, 709] for exp.  Used by the l-rule of the noncentral kernel (dfl_gen.c). */
#ifndef DFL_VMATH_H
#define DFL_VMATH_H
#include <stdint.h>
#include <string.h>
#include <math.h>

static inline double dfl_bits_to_double(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

/* y[i] = exp(x[i]) */
static inline void dfl_vexp(int n, const double *x, double *y)
{
    const double log2e = 1.4426950408889634, ln2hi = 0.693147180369123816490, ln2lo = 1.90821492927058770002e-10;
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        xi = xi > 709.0 ? 709.0 : xi; xi = xi < -745.0 ? -745.0 : xi;
        double kf = (double) (long) (xi * log2e + (xi >= 0 ? 0.5 : -0.5));   /* round to nearest */
        double r = (xi - kf * ln2hi) - kf * ln2lo;
        /* e^r, |r| <= 0.3466: Taylor to degree 13 (error 2e-18) */
        double p = 1.0 / 6227020800.0;
        p = p * r + 1.0 / 479001600.0;
        p = p * r + 1.0 / 39916800.0;
        p = p * r + 1.0 / 3628800.0;
        p = p * r + 1.0 / 362880.0;
        p = p * r + 1.0 / 40320.0;
        p = p * r + 1.0 / 5040.0;
        p = p * r + 1.0 / 720.0;
        p = p * r + 1.0 / 120.0;
        p = p * r + 1.0 / 24.0;
        p = p * r + 1.0 / 6.0;
        p = p * r + 0.5;
        p = p * r + 1.0;
        p = p * r + 1.0;
        long k = (long) kf;
        /* 2^k for k in [-1074, 1023]: split into two factors so that the subnormal range is reachable */
        long k1 = k < -1022 ? -1022 : k, k2 = k - k1;
        double s1 = dfl_bits_to_double((uint64_t) (k1 + 1023) << 52);
        double s2 = dfl_bits_to_double((uint64_t) (k2 + 1023) << 52);
        y[i] = p * s1 * s2;
    }
}

/* s[i] = sin(x[i]), c[i] = cos(x[i]) */
static inline void dfl_vsincos(int n, const double *x, double *s, double *c)
{
    const double twooverpi = 0.636619772367581343076;
    const double p1 = 1.5707963267341256e+00, p2 = 6.0771005065061922e-11, p3 = 2.0222662487959502e-21;   /* pi/2 in three parts */
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double jf = (double) (long) (xi * twooverpi + (xi >= 0 ? 0.5 : -0.5));
        double r = ((xi - jf * p1) - jf * p2) - jf * p3;
        double r2 = r * r;
        /* sin r: degree 15, cos r: degree 16, |r| <= pi/4 */
        double ps = -1.0 / 1307674368000.0;
        ps = ps * r2 + 1.0 / 6227020800.0;
        ps = ps * r2 - 1.0 / 39916800.0;
        ps = ps * r2 + 1.0 / 362880.0;
        ps = ps * r2 - 1.0 / 5040.0;
        ps = ps * r2 + 1.0 / 120.0;
        ps = ps * r2 - 1.0 / 6.0;
        ps = ps * r2 * r + r;
        double pc = 1.0 / 20922789888000.0;
        pc = pc * r2 - 1.0 / 87178291200.0;
        pc = pc * r2 + 1.0 / 479001600.0;
        pc = pc * r2 - 1.0 / 3628800.0;
        pc = pc * r2 + 1.0 / 40320.0;
        pc = pc * r2 - 1.0 / 720.0;
        pc = pc * r2 + 1.0 / 24.0;
        pc = pc * r2 - 0.5;
        pc = pc * r2 + 1.0;
        long j = (long) jf; int q = (int) ((unsigned long) j & 3);
        /* quadrant: 0: (sin, cos), 1: (cos, -sin), 2: (-sin, -cos), 3: (-cos, sin) */
        double sv = (q & 1) ? pc : ps, cv = (q & 1) ? ps : pc;
        double sgs = (q == 2 || q == 3) ? -1.0 : 1.0, sgc = (q == 1 || q == 2) ? -1.0 : 1.0;
        s[i] = sgs * sv; c[i] = sgc * cv;
    }
}
#endif
