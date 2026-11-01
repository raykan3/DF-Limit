// trlim.js -- the local-to-unity limit of the least squares t-ratio in the
// Gaussian AR(1) (Bao, Broda and Kan, "The Exact Limiting Distribution of the
// Dickey-Fuller t-Ratio"): its distribution function, density and moments, a
// JavaScript port of the MATLAB programs of the LTRATIO package: DFX_CORE (the
// single integral of Theorem 2 of the paper, zero and stationary starts, any
// local null value c0), TR_LIMDIST, TR_CFLIM, TR_JBSTART, TR_NSSTART (the double
// integral of Theorem 3, used for a fixed start g0 != 0, with the initial-value
// terms in closed form, Proposition 2) and TR_LIMMOM (the moments).  Complex
// numbers are [re, im].  Needs the table NT (ntcoef.js).
//
//   const E = TRL.engine(cas, c, { c0, g0 });   cas: DK DU DT SK SU ST
//   E.point(x)  -> { F, f }   cdf and density of the limit at x (E.method: 'single' or 'double')
//   E.moments(4) -> [E t, E t^2, E t^3, E t^4]
//   TRL.single(cas, c, { c0 }).cdf(x), .pdf(x)   the single integral directly
'use strict';
const TRL = (function () {
  // ---------------- complex arithmetic ----------------
  const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  function cdiv(a, b) {
    if (Math.abs(b[0]) >= Math.abs(b[1])) { const r = b[1] / b[0], d = b[0] + b[1] * r; return [(a[0] + a[1] * r) / d, (a[1] - a[0] * r) / d]; }
    const r = b[0] / b[1], d = b[0] * r + b[1]; return [(a[0] * r + a[1]) / d, (a[1] * r - a[0]) / d];
  }
  const csc = (a, s) => [a[0] * s, a[1] * s];
  const cabs = (a) => Math.hypot(a[0], a[1]);
  const carg = (a) => Math.atan2(a[1], a[0]);
  const clog = (a) => [Math.log(cabs(a)), carg(a)];
  const cexp = (a) => { const e = Math.exp(a[0]); return a[1] === 0 ? [e, 0] : [e * Math.cos(a[1]), e * Math.sin(a[1])]; };
  const cneg = (a) => [-a[0], -a[1]];
  const imul = (a) => [-a[1], a[0]];          // i a
  const cinv = (a) => cdiv([1, 0], a);
  const csq = (a) => cmul(a, a);
  function csqrt(a) {                          // principal root
    const r = cabs(a); if (r === 0) return [0, 0];
    if (a[0] >= 0) { const re = Math.sqrt((r + a[0]) / 2); return [re, a[1] / (2 * re)]; }
    const im = Math.sqrt((r - a[0]) / 2); const re = Math.abs(a[1]) / (2 * im);
    return [re, (a[1] < 0 || Object.is(a[1], -0)) ? -im : im];
  }
  function cpowi(a, m) {                       // a^m, integer m >= 0
    let r = [1, 0], b = a, e = m;
    while (e > 0) { if (e & 1) r = cmul(r, b); e = Math.floor(e / 2); if (e > 0) b = cmul(b, b); }
    return r;
  }
  const isfin = (a) => isFinite(a[0]) && isFinite(a[1]);
  const wrap = (t) => t - 2 * Math.PI * Math.round(t / (2 * Math.PI));

  // ---------------- special functions ----------------
  // Faddeeva function w(z) = exp(-z^2) erfc(-iz), Weideman (1994), N = 36
  const FAD = (function () {
    const N = 36, M = 2 * N, M2 = 2 * M, L = Math.sqrt(N / Math.SQRT2);
    const fv = [0];
    for (let k = -M + 1; k <= M - 1; k++) { const t = L * Math.tan(k * Math.PI / M / 2); fv.push(Math.exp(-t * t) * (L * L + t * t)); }
    const len = fv.length, half = len / 2, sh = fv.slice(half).concat(fv.slice(0, half));
    const a = [];
    for (let k = 0; k <= N; k++) { let s = 0; for (let j = 0; j < len; j++) s += sh[j] * Math.cos(2 * Math.PI * j * k / len); a.push(s / M2); }
    return { coef: a.slice(1, N + 1).reverse(), L };
  })();
  function wup(z) {
    const L = FAD.L, iz = imul(z), den = [L - iz[0], -iz[1]];
    const Z = cdiv([L + iz[0], iz[1]], den);
    let p = [FAD.coef[0], 0];
    for (let k = 1; k < FAD.coef.length; k++) p = cadd(cmul(p, Z), [FAD.coef[k], 0]);
    return cadd(csc(cdiv(p, cmul(den, den)), 2), cdiv([1 / Math.sqrt(Math.PI), 0], den));
  }
  function faddeeva(z) { return z[1] >= 0 ? wup(z) : csub(csc(cexp(cneg(csq(z))), 2), wup(cneg(z))); }
  function erfc_r(x) {                         // real erfc, x >= 0 (continued fraction / series)
    if (x < 2) {
      // erf by its series
      let s = 0, t = x, k = 0; const x2 = x * x;
      do { s += t / (2 * k + 1); k++; t *= -x2 / k; } while (Math.abs(t) > 1e-17 * Math.abs(s) && k < 200);
      return 1 - 2 / Math.sqrt(Math.PI) * s;
    }
    // Lentz continued fraction for erfc
    let f = x, C = x, D = 0;
    for (let k = 1; k < 300; k++) {
      const an = k / 2; D = x + an * D; D = 1 / D; C = x + an / C; const del = C * D; f *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return Math.exp(-x * x) / (Math.sqrt(Math.PI) * f);
  }
  function erfr(x) { return x >= 0 ? 1 - erfc_r(x) : erfc_r(-x) - 1; }
  function erfz(z) {
    if (z[1] === 0) return [erfr(z[0]), 0];
    return csub([1, 0], cmul(cexp(cneg(csq(z))), faddeeva(imul(z))));
  }
  // exponentially scaled K0: e^z K0(z), z off the negative real axis
  function besselk0s(z) {
    if (cabs(z) < 1) {
      // K0 = -(log(z/2) + gamma) I0 + sum (z^2/4)^k/(k!)^2 H_k
      const q = csc(csq(z), 0.25); let t = [1, 0], I0 = [1, 0], S = [0, 0], H = 0;
      for (let k = 1; k < 40; k++) { t = csc(cmul(t, q), 1 / (k * k)); H += 1 / k; I0 = cadd(I0, t); S = cadd(S, csc(t, H)); if (cabs(t) < 1e-18) break; }
      const K0 = cadd(cneg(cmul(cadd(clog(csc(z, 0.5)), [0.5772156649015329, 0]), I0)), S);
      return cmul(K0, cexp(z));
    }
    const r = cabs(z);
    if (r > 30) {
      // asymptotic series e^z K0(z) ~ (pi/(2z))^(1/2) sum_k (-1)^k ((2k-1)!!)^2/(k! 8^k z^k)
      const iz = cinv(z); let t = [1, 0], s = [1, 0];
      for (let k = 1; k < 30; k++) { t = csc(cmul(t, iz), -((2 * k - 1) * (2 * k - 1)) / (8 * k)); s = cadd(s, t); if (cabs(t) < 1e-17) break; }
      return cmul(csqrt(csc(iz, Math.PI / 2)), s);
    }
    // e^z K0(z) = int_0^inf 2 exp(-v^2)/sqrt(v^2 + 2z) dv (u = v^2 in int e^-u/sqrt(u(u+2z)) du),
    // an even analytic integrand: trapezoidal rule on the whole line, its singularities at
    // distance >= |z|^(1/2) cos(arg(z)/2) from the real axis
    const h = Math.min(0.42, 0.15 * Math.sqrt(r) * Math.max(0.3, Math.cos(carg(z) / 2) / 0.7071)), z2 = csc(z, 2);
    let s = csc(cinv(csqrt(z2)), 0.5);
    for (let v = h; v <= 6.5; v += h) s = cadd(s, csc(cinv(csqrt([v * v + z2[0], z2[1]])), Math.exp(-v * v)));
    return csc(s, 2 * h);
  }
  // Gauss-Legendre nodes and weights on [0,1]
  function gl01(m) {
    const x = new Array(m), w = new Array(m);
    for (let i = 1; i <= m; i++) {
      let z = Math.cos(Math.PI * (i - 0.25) / (m + 0.5)), pp = 1;
      for (let it = 0; it < 100; it++) {
        let p1 = 1, p2 = 0;
        for (let jj = 1; jj <= m; jj++) { const p3 = p2; p2 = p1; p1 = ((2 * jj - 1) * z * p2 - (jj - 1) * p3) / jj; }
        pp = m * (z * p1 - p2) / (z * z - 1);
        const z1 = z; z = z1 - p1 / pp; if (Math.abs(z - z1) < 1e-15) break;
      }
      x[m - i] = (1 + z) / 2; w[m - i] = 1 / ((1 - z * z) * pp * pp);
    }
    return { x, w };
  }
  function lgamma(x) { // Lanczos, x > 0
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5; for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // ---------------- the engine ----------------
  function engine(cas0, c, opts) {
    opts = opts || {};
    cas0 = cas0.toUpperCase();
    const stA = cas0[0] === 'S';
    if (stA && !(c < 0)) throw new Error('A stationary start requires c < 0.');
    const cas = stA ? 'D' + cas0[1] : cas0;
    const c0 = (opts.c0 === undefined || opts.c0 === null) ? c : opts.c0;
    let g0 = stA ? 0 : (opts.g0 || 0);
    if (cas !== 'DK' && c === 0) g0 = 0;            // DU, DT do not depend on the start at c = 0
    const s02 = stA ? -1 / (2 * c) : 0;
    const j = { DK: 1, DU: 2, DT: 4 }[cas], b = -c;
    // series coefficients of z^k lambda^m at this c (powers of b = -c)
    const T = NT[j], AS = T.map(row => [0, 1, 2].map(mm => { if (mm >= row.length) return 0; let s = 0, bp = 1; for (let qq = 0; qq < row[mm].length; qq++) { s += row[mm][qq] * bp; bp *= b; } return s; }));
    const st = stA && cas === 'DK', nz = st || (g0 !== 0 && cas === 'DK'), nzR = cas !== 'DK' && (stA || g0 !== 0);
    const useg = !(nzR && !stA) && !(cas === 'DK' && !st);
    const M = opts.M || 64, Nv = opts.Nv || 48, Nl = opts.Nl || 32;
    const GLm = gl01(M), GLl = gl01(Nl);
    let nbad = 0;

    function coefs(z) {
      // F0, F1, F2 of D e^{-c}, scaled by exp(-ls); ls = 0 (series) or theta = (-z)^(1/2)
      if (cabs(z) < 4) {
        const F = [0, 1, 2].map(mm => { let p = [0, 0]; for (let k = AS.length - 1; k >= 0; k--) p = cadd(cmul(p, z), [AS[k][mm], 0]); return p; });
        return { F, ls: [0, 0] };
      }
      const X = csc(cadd(z, [b * b, 0]), 0.5), th = csqrt(cneg(z)), E = cexp(csc(th, -2));
      const C = csc(cadd([1, 0], E), 0.5), S = cdiv(csub([1, 0], E), csc(th, 2)), one = cexp(cneg(th));
      const Cm1 = csub(C, one), iz = cinv(z), iz2 = csq(iz), iz3 = cmul(iz2, iz);
      let F0, F1, F2;
      if (j === 1) { F0 = cadd(C, csc(S, b)); F1 = S; F2 = [0, 0]; }
      else if (j === 2) {
        F2 = cadd(cmul(S, iz), csc(cmul(Cm1, iz2), 2));
        F1 = csc(F2, -b * b);
        F0 = csub(csub(cmul(cmul(csub(csc(X, 2), [b * b * b, 0]), S), iz), csc(cmul(C, iz), b * b)), csc(cmul(cmul(X, Cm1), iz2), 4 * b));
      } else {
        const Tt = cmul(cadd(S, cmul(Cm1, iz)), iz3), b2 = b * b, b4 = b2 * b2;
        F2 = cadd(cadd(csc(cmul(S, iz2), -4 * (b2 + 3 * b + 27)), cmul(cadd(csc(X, 192), [288 * (b + 1), 0]), Tt)),
          cadd(csc(cmul(C, iz3), -32 * (b2 + 3 * b + 6)), csc(cmul(one, iz3), -16 * (b2 + 3 * b - 3))));
        F1 = cadd(cadd(csc(cmul(S, iz2), b4), csc(Tt, -24 * b4)), cadd(csc(cmul(C, iz3), 8 * b4), csc(cmul(one, iz3), 4 * b4)));
        F0 = cadd(cadd(cmul(cmul(csub([b4 * b, 0], csc(X, 8 * (b2 - 3 * b - 3))), S), iz2), csc(cmul(csq(X), Tt), -96 * (b + 1))),
          cadd(cmul(cadd(csc(iz2, b4), csc(cmul(X, iz3), 16 * b2 * b)), C), csc(cmul(cmul(X, one), iz3), 8 * b2 * (b + 3))));
      }
      return { F: [F0, F1, F2], ls: th };
    }
    // sign of D(0, 2s - c^2) at real s and log M_B(s) (zero start)
    function realD(s) { const { F, ls } = coefs([2 * s - b * b, 0]); return { Dr: cmul(cexp([0, ls[1]]), F[0])[0], lD: -b + ls[0] + Math.log(cabs(F[0])) }; }   // Dr: the sign of D
    function domainend() {
      const ss = [];
      for (let w = -Math.abs(c); w <= 4 * Math.PI + 0.01 + 1e-12; w += 0.01) { const z = Math.sign(w) * w * w, s = (z + c * c) / 2; if (s > 0) ss.push(s); }
      let k = -1; for (let i = 0; i < ss.length; i++) if (realD(ss[i]).Dr <= 0) { k = i; break; }
      if (k < 0) return Infinity;
      let lo = k > 0 ? ss[k - 1] : 0, hi = ss[k];
      for (let it = 0; it < 100; it++) { const md = (lo + hi) / 2; if (realD(md).Dr > 0) lo = md; else hi = md; if (hi - lo < 1e-14 * Math.max(1, hi)) break; }
      return lo;
    }
    const smax0 = domainend();
    function LB0(s) {                         // log M_B at real s, zero start; ok: s in the domain
      const r = realD(s); return { L: -r.lD / 2, ok: s < smax0 && r.Dr > 0 };
    }
    function jbstart(s) {                     // TR_JBSTART at the mgf argument s of B
      const lam = cneg(s), mu = csqrt(cadd([c * c, 0], csc(lam, 2)));
      const e = cexp(csc(mu, -2));
      let q = cdiv(csub([1, 0], e), mu);
      if (cabs(mu) < 1e-4) q = cadd(csub([2, 0], csc(mu, 2)), csc(csq(mu), 4 / 3));
      const den = csub(cadd([1, 0], e), csc(q, c));
      return { aq: cdiv(cmul(s, q), den), mq: cdiv(csc(cexp(cneg(mu)), 2), den) };
    }
    // the initial-value term of log E exp(lam A' + s B) with a fixed start g0:
    // log M = log M_0 + g0^2 Q(lam, s), Q in closed form (TR_NSSTART, Proposition 2
    // of the paper).  Cases DU, DT (Nabeya and Sorensen's determinants):
    //    Q = c^(2k) e^c (s + lam^2/2 + c lam) g(z)/D(lam, s),  k = 1 (DU), 2 (DT),
    // z = 2s - c^2, D e^-c e^-ls = F0 - F1 lam + F2 lam^2 the scaled coefficients of
    // coefs(), and g(z) entire: its Taylor series for |z| < 4, and beyond the closed
    // forms in theta = (-z)^(1/2) with e^theta taken out.  Case DK (the Riccati
    // solution of TR_JBSTART): Q = aq - lam/2 + lam mq^2/(2 (1 - 2 lam kap_1)).
    const GCOEF = (function () {
      const k = (cas === 'DT') ? 2 : 1, a = [];
      const lf = (n) => { let v = 0; for (let i = 2; i <= n; i++) v += Math.log(i); return v; };
      for (let jj = 0; jj < 24; jj++) {
        const sg = (jj % 2) ? -1 : 1;
        if (k === 1) a.push(sg * (2 * jj + 2) * Math.exp(-lf(2 * jj + 4)));
        else a.push(sg * (Math.exp(-lf(2 * jj + 5)) - 8 * Math.exp(-lf(2 * jj + 6)) + 24 * Math.exp(-lf(2 * jj + 7)) - 24 * Math.exp(-lf(2 * jj + 8))));
      }
      return a;
    })();
    function logg(z) {                        // log g(z), any branch (it is exponentiated)
      if (cabs(z) < 4) {
        let g = [0, 0], p = [1, 0];
        for (let m = 0; m < GCOEF.length; m++) { g = cadd(g, csc(p, GCOEF[m])); p = cmul(p, z); }
        return clog(g);
      }
      const th = csqrt(cneg(z)), e = cexp(csc(th, -2)), em = cexp(cneg(th));
      const Cs = csc(cadd([1, 0], e), 0.5), Ss = cdiv(csub([1, 0], e), csc(th, 2));   // C e^-theta, S e^-theta
      const iz = cinv(z), iz2 = csq(iz), Cm = csub(Cs, em);
      let gs;
      if (cas === 'DT') {
        const iz3 = cmul(iz2, iz), iz4 = csq(iz2);
        gs = csub(csub(csc(cmul(Cs, iz3), 12), cmul(cmul(cadd([24, 0], csc(z, 4)), Cm), iz4)), cmul(cmul(csub([24, 0], z), Ss), iz3));
      } else {
        gs = csub(cneg(cmul(Ss, iz)), csc(cmul(Cm, iz2), 2));
      }
      return cadd(th, clog(gs));
    }
    function qstart(lam, s) {                 // Q(lam, s) for the case cas at (complex) lam, s
      const z = csub(csc(s, 2), [c * c, 0]), { F, ls } = coefs(z);
      const Dt = cadd(csub(F[0], cmul(F[1], lam)), cmul(F[2], csq(lam)));   // D e^(-c-ls)
      if (cas === 'DK') {
        const { aq, mq } = jbstart(s);
        return cadd(csub(aq, csc(lam, 0.5)), cdiv(cmul(cmul(lam, csq(mq)), F[0]), csc(Dt, 2)));
      }
      const k = (cas === 'DT') ? 2 : 1, ck = Math.pow(c, 2 * k);
      const pre = cadd(cadd(s, csc(csq(lam), 0.5)), csc(lam, c));
      return csc(cdiv(cmul(pre, cexp(csub(logg(z), ls))), Dt), ck);
    }
    // initial-value part of log E exp(lam A' + s B) (fixed start par, cases DU, DT)
    function lpar(lam, s, par) { return csc(qstart(lam, s), par * par); }
    function LB(s) {                          // log M_B at real s with the initial-value terms
      let { L, ok } = LB0(s);
      if (nzR) {
        let Q0 = 0, iv = ok && isFinite(s) && s < smax0;
        if (iv) { const Qv = lpar([0, 0], [s, 0], 1); Q0 = Qv[0]; iv = isfin(Qv); }
        if (stA) { const d1 = 1 - 2 * s02 * Q0; ok = ok && iv && d1 > 0; L -= Math.log(Math.abs(d1)) / 2; }
        else { ok = ok && iv; L += g0 * g0 * Q0; }
      }
      if (nz) {
        const { aq } = jbstart([s, 0]);
        ok = ok && isfin(aq) && Math.abs(aq[1]) < 1e-8 * Math.max(1, cabs(aq));
        if (st) { const d1 = 1 - 2 * aq[0] * s02; ok = ok && d1 > 0; L -= Math.log(Math.abs(d1)) / 2; }
        else L += g0 * g0 * aq[0];
      }
      return { L, ok };
    }
    // log M_B along a path column from its real start (continuous branch), and the weights
    function qeval(s) {
      const z = csub(csc(s, 2), [b * b, 0]), th = csqrt(cneg(z)), { F, ls } = coefs(z);
      return { lv: cadd(clog(F[0]), csub(ls, th)), th, F };
    }
    function kapOf(F) {
      const p1 = cdiv(F[1], csc(F[0], 2)), p2 = cdiv(F[2], csc(F[0], 4));
      if (j === 1) return [p1, [0, 0]];
      const dsc = csqrt(csub(csq(p1), csc(p2, 4)));
      let k1 = csc(cadd(p1, dsc), 0.5), k2 = csc(csub(p1, dsc), 0.5);
      if (k2[0] > k1[0]) { const t = k1; k1 = k2; k2 = t; }
      return [k1, k2];
    }
    function pathB(S) {
      const N = S.length, ev = S.map(qeval);
      const seg = (sa, ea, sb, eb, depth) => {
        const d = wrap(eb.lv[1] - ea.lv[1]);
        if (Math.abs(d) <= Math.PI / 4 || depth >= 30) return d;
        const sm = csc(cadd(sa, sb), 0.5), em = qeval(sm);
        return seg(sa, ea, sm, em, depth + 1) + seg(sm, em, sb, eb, depth + 1);
      };
      const L = new Array(N), k1 = new Array(N), k2 = new Array(N);
      let ph = ev[0].lv[1];
      const m0 = Math.round((ph + ev[0].th[1]) / (2 * Math.PI));
      for (let i = 0; i < N; i++) {
        if (i > 0) ph += seg(S[i - 1], ev[i - 1], S[i], ev[i], 0);
        const lt = [ev[i].lv[0] + ev[i].th[0], ph + ev[i].th[1] - 2 * Math.PI * m0];
        L[i] = csc([lt[0] - b, lt[1]], -0.5);
        const kk = kapOf(ev[i].F); k1[i] = kk[0]; k2[i] = kk[1];
      }
      return { L, k1, k2 };
    }
    function unwrapLog(arr) {                 // continuous log along a column
      const out = arr.map(clog);
      for (let i = 1; i < out.length; i++) out[i][1] = out[i - 1][1] + wrap(out[i][1] - out[i - 1][1]);
      return out;
    }
    function roots2(p1, p2) {
      const dsc = csqrt(csub(csq(p1), csc(p2, 4)));
      let k1 = csc(cadd(p1, dsc), 0.5), k2 = csc(csub(p1, dsc), 0.5);
      if (k2[0] > k1[0]) { const t = k1; k1 = k2; k2 = t; }
      return [k1, k2];
    }
    function qpieces(S, gg, s1, s2) {
      const la = cdiv([0.25, 0], s1), lb = cdiv([-0.25, 0], s2);
      const Q0 = lpar([0, 0], S, gg), Qa = csub(lpar(la, S, gg), Q0), Qb = csub(lpar(lb, S, gg), Q0);
      const A11 = cdiv(csc(la, 0.5), csub([1, 0], cmul(la, s1))), A12 = cneg(cdiv(csc(la, 0.5), cadd([1, 0], cmul(la, s2))));
      const A21 = cdiv(csc(lb, 0.5), csub([1, 0], cmul(lb, s1))), A22 = cneg(cdiv(csc(lb, 0.5), cadd([1, 0], cmul(lb, s2))));
      const dt = csub(cmul(A11, A22), cmul(A12, A21));
      return { Q0, m1s: cdiv(csub(cmul(Qa, A22), cmul(A12, Qb)), dt), m2s: cdiv(csub(cmul(A11, Qb), cmul(A21, Qa)), dt) };
    }
    function nodepar(S) {
      const { L, k1, k2 } = pathB(S), N = S.length;
      let mq = null, m2 = null;
      if (nzR) {
        if (stA) {
          const pcs = S.map((s, i) => qpieces(s, 1, csc(k1[i], 2), csc(k2[i], -2)));
          const d1 = pcs.map(pc => csub([1, 0], csc(pc.Q0, 2 * s02))), ld = unwrapLog(d1);
          for (let i = 0; i < N; i++) {
            L[i] = csub(L[i], csc(ld[i], 0.5));
            const st2 = cdiv([s02, 0], d1[i]);
            const p1 = cadd(cadd(k1[i], k2[i]), csc(cmul(st2, csub(pcs[i].m1s, pcs[i].m2s)), 0.5));
            const p2 = cadd(cmul(k1[i], k2[i]), csc(cmul(st2, csub(cmul(pcs[i].m1s, k2[i]), cmul(pcs[i].m2s, k1[i]))), 0.5));
            const kk = roots2(p1, p2); k1[i] = kk[0]; k2[i] = kk[1];
          }
        } else {
          mq = new Array(N); m2 = new Array(N);
          for (let i = 0; i < N; i++) {
            const pc = qpieces(S[i], g0, csc(k1[i], 2), csc(k2[i], -2));
            L[i] = cadd(L[i], pc.Q0); mq[i] = pc.m1s; m2[i] = pc.m2s;
          }
        }
      }
      if (nz) {
        const js = S.map(jbstart);
        if (st) {
          const d1 = js.map(v => csub([1, 0], csc(v.aq, 2 * s02))), ld = unwrapLog(d1);
          for (let i = 0; i < N; i++) {
            L[i] = csub(L[i], csc(ld[i], 0.5));
            const t2 = cdiv([s02, 0], d1[i]), v = csc(k1[i], 2);
            const p1 = csc(csub(cadd(cmul(csq(js[i].mq), t2), v), t2), 0.5), p2 = csc(cmul(t2, v), -0.25);
            const kk = roots2(p1, p2); k1[i] = kk[0]; k2[i] = kk[1];
          }
        } else {
          mq = new Array(N);
          for (let i = 0; i < N; i++) { L[i] = cadd(L[i], csc(js[i].aq, g0 * g0)); mq[i] = js[i].mq; }
        }
      }
      return { L, k1, k2, mq, m2 };
    }
    // density of R at a (real) under the tilt with the weights k1, k2 (cases with useg)
    function gden(a, k1, k2) {
      const s2 = cinv(csc(k1, 2)), s1 = cinv(csc(k2, 2));
      const al = csc(cadd(s1, s2), 0.5), be = csc(csub(s2, s1), 0.5), z = csc(be, Math.abs(a));
      return cdiv(cmul(cexp(csub(csc(al, -a), z)), besselk0s(z)), csc(cmul(csqrt(k1), csqrt(cneg(k2))), 2 * Math.PI));
    }
    // P[R <= a] (pdf: its derivative in a) under the tilt; central (mq: null), DK, or noncentral (m2)
    function Pk(k1, k2, a, mq, m2, pdfm) {
      if (m2) return Pnc(csc(k1, 2), csc(k2, -2), a, csqrt(mq), csqrt(m2), pdfm);
      if (cas === 'DK' && !st) {
        if (g0 !== 0) {
          const m = csc(mq, g0), r = Math.sqrt(2 * a + g0 * g0), sg = csqrt(csc(k1, 4));
          const u1 = cdiv(csub([r, 0], m), sg), u2 = cdiv(cadd([r, 0], m), sg);
          if (pdfm) return csc(cdiv(cadd(cexp(cneg(csq(u1))), cexp(cneg(csq(u2)))), csc(sg, Math.sqrt(Math.PI) / 2)), 1 / (2 * r));
          return csc(cadd(erfz(u1), erfz(u2)), 0.5);
        }
        if (pdfm) return cdiv(cexp(cdiv([-a, 0], csc(k1, 2))), csqrt(csc(k1, 2 * Math.PI * a)));
        return erfz(csqrt(cdiv([a, 0], csc(k1, 2))));
      }
      // central rank two, s1 = 2 k1, s2 = -2 k2
      const s1 = csc(k1, 2), s2 = csc(k2, -2);
      const lm = Math.sqrt(2 * 41.4 / Math.max(cinv(s2)[0], 1e-12));
      const zl = GLl.x, wl = GLl.w;
      const is2 = cinv(csc(s2, 2)), nrm = cinv(csqrt(csc(s2, 2 * Math.PI))), i2s1 = cinv(csc(s1, 2)), rs1 = csqrt(csc(s1, 2));
      let P = [0, 0];
      if (a >= 0) {
        const al = Math.max(Math.sqrt(2 * a), 1e-8 * lm), tm = Math.asinh(lm / al);
        for (let k = 0; k < Nl; k++) {
          const t = zl[k] * tm, Lg = al * Math.sinh(t), dL = al * Math.cosh(t) * wl[k] * tm;
          const ph = cmul(cexp(csc(is2, -Lg * Lg)), nrm), u2 = 2 * a + Lg * Lg;
          const v = pdfm ? csc(cdiv(cmul(ph, cexp(csc(i2s1, -u2))), rs1), 2 / Math.sqrt(Math.PI) / Math.sqrt(u2))
            : cmul(ph, erfz(csc(csqrt(i2s1), Math.sqrt(u2))));
          P = cadd(P, csc(v, dL));
        }
      } else {
        const l0 = Math.sqrt(-2 * a), um = Math.sqrt(Math.max(lm * lm - l0 * l0, 0) + 1e-30);
        for (let k = 0; k < Nl; k++) {
          const U = zl[k] * um, WU = wl[k] * um, Lg = Math.sqrt(U * U + l0 * l0);
          const ph = cmul(cexp(csc(is2, -Lg * Lg)), nrm);
          const v = pdfm ? csc(cdiv(cmul(ph, cexp(csc(i2s1, -U * U))), rs1), 2 / Math.sqrt(Math.PI) / Lg)
            : csc(cmul(ph, erfz(cdiv([U, 0], rs1))), U / Lg);
          P = cadd(P, csc(v, WU));
        }
      }
      return csc(P, 2);
    }
    function Pnc(s1, s2, a, m1, m2, pdfm) {
      const lm = Math.sqrt(2 * 41.4 / Math.max(cinv(s2)[0], 1e-12)) + cabs(m2);
      const zl = GLl.x, wl = GLl.w, r2s1 = csqrt(csc(s1, 2)), r2s2 = csqrt(csc(s2, 2));
      const F1 = (u) => csc(cadd(erfz(cdiv(csub([u, 0], m1), r2s1)), erfz(cdiv(cadd([u, 0], m1), r2s1))), 0.5);
      const f1 = (u) => cdiv(cadd(cexp(cneg(csq(cdiv(csub([u, 0], m1), r2s1)))), cexp(cneg(csq(cdiv(cadd([u, 0], m1), r2s1))))), csc(r2s1, Math.sqrt(Math.PI)));
      const f2 = (l) => cdiv(cadd(cexp(cneg(csq(cdiv(csub([l, 0], m2), r2s2)))), cexp(cneg(csq(cdiv(cadd([l, 0], m2), r2s2))))), csc(r2s2, 2 * Math.sqrt(Math.PI)));
      let P = [0, 0];
      if (a >= 0) {
        const al = Math.max(Math.sqrt(2 * a), 1e-8 * lm), tm = Math.asinh(lm / al);
        for (let k = 0; k < Nl; k++) {
          const t = zl[k] * tm, Lg = al * Math.sinh(t), dL = al * Math.cosh(t) * wl[k] * tm, u = Math.sqrt(2 * a + Lg * Lg);
          const v = pdfm ? csc(cmul(f2(Lg), f1(u)), 1 / u) : cmul(f2(Lg), F1(u));
          P = cadd(P, csc(v, dL));
        }
      } else {
        const l0 = Math.sqrt(-2 * a), um = Math.sqrt(Math.max(lm * lm - l0 * l0, 0) + 1e-30);
        for (let k = 0; k < Nl; k++) {
          const U = zl[k] * um, WU = wl[k] * um, Lg = Math.sqrt(U * U + l0 * l0);
          const v = pdfm ? csc(cmul(f2(Lg), f1(U)), 1 / Lg) : csc(cmul(f2(Lg), F1(U)), U / Lg);
          P = cadd(P, csc(v, WU));
        }
      }
      return csc(P, 2);
    }

    // the real domain end with the initial-value terms, E B, untilted weights
    let smax = smax0;
    if (nz || nzR) {
      let hi = isFinite(smax) ? smax : 1e3;
      if (!LB(hi * (1 - 1e-9)).ok) { let lo = 0; for (let it = 0; it < 80; it++) { const md = (lo + hi) / 2; if (LB(md).ok) lo = md; else hi = md; } smax = lo; }
    }
    const hh = 1e-5, EB = (LB(hh).L - LB(-hh).L) / (2 * hh);
    let k10 = null, k20 = null, mq0 = null;
    if (useg) { const np0 = nodepar([[0, 0]]); k10 = np0.k1[0]; k20 = np0.k2[0]; mq0 = np0.mq ? np0.mq[0] : null; }

    // saddle point of log M_B(s) - s b and the path parameters, for one b (cached)
    const pathCache = new Map();
    function saddleExact(bb) {
      let s = 0, slo = -Infinity, shi = smax, sbd = smax, g2 = 1;
      for (let it = 0; it < 200; it++) {
        const r0 = LB(s), bad = !r0.ok || s >= smax, L0 = r0.L;
        if (bad) { shi = s; sbd = Math.min(sbd, s); }
        let e = Math.min(1e-4 * Math.max(1, Math.abs(s)), 0.1 * (sbd - s)), okp = false, Lp = L0, Lm = L0;
        if (!bad) for (let k = 0; k < 30; k++) {
          const rp = LB(s + e), rm = LB(s - e); Lp = rp.L; Lm = rm.L; okp = rp.ok && s + e < smax;
          if (okp) break; e /= 10; sbd = Math.min(sbd, s + 10 * e);
        }
        const g1 = (Lp - Lm) / (2 * e) - bb, g2a = (Lp - 2 * L0 + Lm) / (e * e), gd = !bad && okp;
        if (gd && g1 < 0) slo = Math.max(slo, s);
        if (gd && g1 >= 0) shi = Math.min(shi, s);
        if (gd && g2a > 0) g2 = g2a;
        let sn = (gd && g2a > 0) ? s - g1 / g2a : NaN;
        const A0 = slo, B0 = shi;
        if (!(sn > A0 && sn < B0)) {
          if (isFinite(A0) && isFinite(B0)) sn = (A0 + B0) / 2;
          else if (!isFinite(A0)) sn = B0 - Math.max(1, 2 * Math.abs(B0));
          else sn = A0 + Math.max(1, Math.abs(A0));
        }
        if (gd && (Math.abs(g1) < 1e-9 * bb || (B0 - A0) < 1e-12 * (1 + Math.abs(s)))) break;
        s = sn;
      }
      if (nzR && !stA && isFinite(smax0)) s = Math.min(s, smax0 - 0.02 * Math.max(1, Math.abs(smax0)));
      for (let pass = 0; pass < 2; pass++) {
        const a1 = 1 / Math.sqrt(g2), e = Math.min(0.05 * a1, 0.3 * (sbd - s));
        const L0 = LB(s).L, rp = LB(s + e), rm = LB(s - e);
        const g2n = (rp.L - 2 * L0 + rm.L) / (e * e);
        if (rp.ok && s + e < smax && rm.ok && g2n > 0 && isFinite(g2n)) g2 = g2n;
      }
      return { s, g2, sbd };
    }
    // the saddle points on a grid in log b (only an approximate s* is needed: any real s in
    // the domain gives the same integral; s* and the width a1 just place and scale the path)
    const LGLO = Math.log(EB) - 8.5, LGH = 0.1, LGN = 151;
    let SAD = null;
    function saddle(bb) {
      if (!SAD) { SAD = []; for (let i = 0; i < LGN; i++) SAD.push(saddleExact(Math.exp(LGLO + i * LGH))); }
      const t = (Math.log(bb) - LGLO) / LGH;
      if (!(t >= 0 && t <= LGN - 1)) return saddleExact(bb);
      const i0 = Math.min(LGN - 2, Math.floor(t)), w = t - i0, P0 = SAD[i0], P1 = SAD[i0 + 1];
      return { s: (1 - w) * P0.s + w * P1.s, g2: Math.exp((1 - w) * Math.log(P0.g2) + w * Math.log(P1.g2)), sbd: Math.min(P0.sbd, P1.sbd) };
    }
    function pathFor(bb, keep) {
      if (keep && pathCache.has(bb)) return pathCache.get(bb);
      let { s, g2, sbd } = saddle(bb);
      const a1 = 1 / Math.sqrt(g2);
      if (useg && Math.abs(s) < a1) s = (s >= 0 && a1 < 0.5 * sbd) ? a1 : -a1;   // away from the pole at 0 (cdf)
      const mu = Math.max(a1 / 2, 4.5), sth = a1 / (2 * mu), thm = Math.sqrt(41.5 / (bb * mu));
      const P = { s, mu, sth, thm, cols: {} };
      if (keep) pathCache.set(bb, P);
      return P;
    }
    function column(P, NN) {
      if (P.cols[NN]) return P.cols[NN];
      const ht = Math.asinh(P.thm / P.sth) / NN, S = [], W = [];
      for (let k = 0; k <= NN; k++) {
        const th = P.sth * Math.sinh(k * ht); let w = P.sth * Math.cosh(k * ht) * ht; if (k === 0) w /= 2;
        S.push([P.s + P.mu * th * th, -2 * P.mu * th]);
        W.push(csc([1, th], 2 * P.mu / Math.PI * w));
      }
      const np = nodepar(S);
      for (let k = 0; k <= NN; k++) { if (np.k1[k][0] <= 0) nbad++; if ((cas !== 'DK' || st) && np.k2[k][0] >= 0) nbad++; }
      const col = { S, W, np };
      P.cols[NN] = col;
      return col;
    }
    // (1/2 pi i) path integrals at b for the density (f) and distribution (F) forms
    function pathint(bb, a, NN, keep) {
      const P = pathFor(bb, keep), col = column(P, NN), { S, W, np } = col;
      let If = 0, IF = 0, Ihf = 0, IhF = 0; const sb = Math.sqrt(bb);
      for (let k = 0; k <= NN; k++) {
        const ex = cexp(csub(np.L[k], csc(S[k], bb)));
        let gf, gF;
        if (useg) { const g = gden(a, np.k1[k], np.k2[k]); gf = csc(cmul(ex, g), sb); gF = cdiv(cmul(ex, g), S[k]); }
        else {
          const m2 = np.m2 ? np.m2[k] : null, mq = np.mq ? np.mq[k] : null;
          gf = csc(cmul(ex, Pk(np.k1[k], np.k2[k], a, mq, m2, true)), sb);
          gF = cmul(ex, Pk(np.k1[k], np.k2[k], a, mq, m2, false));
        }
        const vf = cmul(gf, W[k])[0], vF = cmul(gF, W[k])[0];
        If += vf; IF += vF;
        if (k % 2 === 0) { Ihf += 2 * vf; IhF += 2 * vF; }
      }
      return { If, IF, Ihf, IhF, s: P.s };
    }
    function Hb(bs, x, keep) {
      const nb = bs.length, Hf = new Array(nb).fill(0), HF = new Array(nb).fill(0), res = new Array(nb).fill(null), as = new Array(nb);
      for (let i = 0; i < nb; i++) {
        const bb = bs[i], a = 0.5 + x * Math.sqrt(bb) + c0 * bb; as[i] = a;
        if (cas === 'DK' && !st && !(a + g0 * g0 / 2 > 0)) continue;
        res[i] = pathint(bb, a, Nv, keep[i]);
      }
      let scf = 1e-300, scF = 1e-300;
      for (const r of res) if (r) { scf = Math.max(scf, Math.abs(r.If)); scF = Math.max(scF, Math.abs(r.IF)); }
      for (let i = 0; i < nb; i++) {
        let r = res[i]; if (!r) continue;
        for (let rf = 1; rf <= 2; rf++) {
          if (Math.abs(r.If - r.Ihf) <= 1e-9 * scf && Math.abs(r.IF - r.IhF) <= 1e-9 * scF) break;
          r = pathint(bs[i], as[i], Nv * (1 << rf), keep[i]);
        }
        let IF = r.IF;
        if (useg) {
          if (r.s < 0) IF += gden(as[i], k10, k20)[0];
          IF *= x / (2 * Math.sqrt(bs[i])) + c0;
        }
        Hf[i] = r.If; HF[i] = IF;
      }
      return { Hf, HF };
    }
    // the limit at x: distribution function F and density f
    function point(x) {
      const lo = Math.log(EB) - 8, hi = Math.log(EB) + 6, blo = Math.exp(lo);
      const q0 = (cas === 'DK' && !st) ? (1 + g0 * g0) / 2 : 0.5;
      let sr = [];
      if (c0 === 0) sr = [-q0 / x];
      else { const dd = x * x - 4 * c0 * q0; if (dd >= 0) { const r = Math.sqrt(dd); sr = [(-x + r) / (2 * c0), (-x - r) / (2 * c0)]; } }
      const bk = sr.filter(v => v > 0 && isFinite(v)).map(v => Math.log(v * v)).filter(v => v > lo && v < hi).sort((u, v) => u - v);
      // fixed pieces of length 2 (their path columns are kept for all x); the piece that
      // contains a zero of a(b) is split there, with the double cosine map on both sides
      const NP = 7, A = [], B = [], dbl = [];
      for (let k = 0; k < NP; k++) {
        const a0 = lo + k * (hi - lo) / NP, b0 = lo + (k + 1) * (hi - lo) / NP;
        const inn = bk.filter(v => v > a0 && v < b0);
        if (!inn.length) { A.push(a0); B.push(b0); dbl.push(0); continue; }
        const br = [a0].concat(inn, [b0]);
        for (let q = 1; q < br.length; q++) { if (br[q] - br[q - 1] < 1e-10) continue; A.push(br[q - 1]); B.push(br[q]); dbl.push(useg ? 1 : -1); }
      }
      const zm = GLm.x, wm = GLm.w, bs = [], wt = [], keep = [];
      for (let pc = 0; pc < A.length; pc++) {
        for (let k = 0; k < M; k++) {
          let t, dt;
          if (dbl[pc] === 1) { const t1 = (1 - Math.cos(Math.PI * zm[k])) / 2; t = (1 - Math.cos(Math.PI * t1)) / 2; dt = (Math.PI / 2) ** 2 * Math.sin(Math.PI * t1) * Math.sin(Math.PI * zm[k]); }
          else { t = (1 - Math.cos(Math.PI * zm[k])) / 2; dt = Math.PI / 2 * Math.sin(Math.PI * zm[k]); }
          const lb = A[pc] + t * (B[pc] - A[pc]), bb = Math.exp(lb);
          bs.push(bb); wt.push(wm[k] * dt * (B[pc] - A[pc]) * bb); keep.push(dbl[pc] === 0);
        }
      }
      const { Hf, HF } = Hb(bs, x, keep);
      let f = 0, F = 0;
      for (let i = 0; i < bs.length; i++) { f += wt[i] * Hf[i]; F += wt[i] * HF[i]; }
      if (useg) F += Pk(k10, k20, 0.5 + x * Math.sqrt(blo) + c0 * blo, mq0, null, false)[0];
      return { F, f };
    }

    // ---------------- moments (TR_LIMMOM) ----------------
    function circle(s, r) {
      const Nu = 64, lD = [], lS = [], Ls = [];
      for (let k = 0; k < Nu; k++) {
        const U = [r * Math.cos(2 * Math.PI * k / Nu), r * Math.sin(2 * Math.PI * k / Nu)], V = csub([-s, 0], csc(U, c0));
        const z = csub(csc(V, 2), [b * b, 0]), cf = coefs(z), lam = cneg(U);
        const Dt = cadd(cadd(cf.F[0], cmul(cf.F[1], lam)), cmul(cf.F[2], csq(lam)));
        lD.push(clog(Dt)); Ls.push(csub(csc(U, -0.5), csc(cadd([c, 0], cf.ls), 0.5)));
        if (stA || g0 !== 0) {
          const q = qstart(U, V);              // log E exp(U A' + V B), the start term at g0 = 1
          lS.push(stA ? clog(csub([1, 0], csc(q, 2 * s02))) : csc(q, g0 * g0));
        }
      }
      const cont = (A) => {                 // unwrap along the circle; null if it winds or is too coarse
        if (Math.abs(A[0][1]) > 1e-8) return false;
        for (let k = 1; k < Nu; k++) { const d = wrap(A[k][1] - A[k - 1][1]); if (Math.abs(d) > Math.PI / 3) return false; A[k][1] = A[k - 1][1] + d; }
        const d = wrap(A[0][1] - A[Nu - 1][1]); return Math.abs(A[Nu - 1][1] + d - A[0][1]) < 1e-6;
      };
      if (!cont(lD)) return null;
      if (stA && !cont(lS)) return null;
      const f = [];
      for (let k = 0; k < Nu; k++) {
        let L = csub(Ls[k], csc(lD[k], 0.5));
        if (stA) L = csub(L, csc(lS[k], 0.5)); else if (g0 !== 0) L = cadd(L, lS[k]);
        f.push(cexp(L));
      }
      return f;
    }
    function moments(p) {
      const Nu = 64, h = 0.05, tlo = -60, thi = 8;
      const w = []; for (let k = 0; k < Nu; k++) w.push([Math.cos(2 * Math.PI * k / Nu), Math.sin(2 * Math.PI * k / Nu)]);
      const sums = new Array(p).fill(0), sums2 = new Array(p).fill(0);
      let idx = 0;
      for (let t = tlo; t <= thi + 1e-9; t += h, idx++) {
        const s = Math.exp(t);
        const { F } = coefs([-2 * s - b * b, 0]), kk = kapOf(F), kmax = Math.max(cabs(kk[0]), cabs(kk[1]));
        let r = 0.4 / (2 * kmax);
        if (c0 !== 0) r = Math.min(r, 0.4 * (s + smax0) / Math.abs(c0));
        if (stA || g0 !== 0) r = Math.min(r, 0.25);
        // the mgf on the circle |u| = r; log D (and the start factor) continued along the circle from
        // the real point u = r, where D > 0; a nonzero winding means a zero of D inside: halve r
        let f = null;
        for (let tries = 0; tries < 8 && !f; tries++, r /= 2) f = circle(s, r);
        if (!f) throw new Error('moments: no zero-free circle in u.');
        r *= 2;
        let fact = 1;
        for (let jj = 1; jj <= p; jj++) {
          fact *= jj;
          let acc = 0; for (let k = 0; k < Nu; k++) acc += cmul(f[k], cpowi([w[k][0], -w[k][1]], jj))[0];
          const Mj = fact / (Nu * Math.pow(r, jj)) * acc, g = Math.pow(s, jj / 2) * Mj;
          sums[jj - 1] += g; if (idx % 2 === 0) sums2[jj - 1] += g;
        }
      }
      const m = [], err = [];
      for (let jj = 1; jj <= p; jj++) { const G = Math.exp(lgamma(jj / 2)); m.push(h * sums[jj - 1] / G); err.push(Math.abs(h * sums[jj - 1] - 2 * h * sums2[jj - 1]) / G); }
      return { m, err };
    }
    // the representation: the single integral (Theorem 2) with a zero or a stationary start,
    // the double integral (Theorem 3) with a fixed start g0 != 0 or on request (opts.method)
    const useSingle = g0 === 0 && opts.method !== 'double';
    const S1 = useSingle ? single(cas0, c, { c0, tau: opts.tau, hu: opts.hu, N: opts.N, wmax: opts.wmax }) : null;
    const pointAny = useSingle ? S1.point : point;
    return { point: pointAny, pointDouble: point, moments, method: useSingle ? 'single' : 'double',
      info: () => ({ EB, smax, smax0, nbad, useg, cache: pathCache.size, method: useSingle ? 'single' : 'double' }),
      _d: { pathFor, column, pathint, nodepar, LB, LB0, Pk, gden, Hb, coefs, qstart } };
  }
  // =====================================================================================
  // The single integral (Theorem 2 of the paper; a port of DFX_CORE of the LTRATIO
  // package): the inversion in the argument of U is explicit, phi_t(X) a scaled chi^2_1
  // density (case DK) or a Bessel K_0 density, and one integral along the line
  // Re t = tau remains, split into a smooth part (trapezoidal rule in w = tau sinh u, or
  // Gauss-Legendre panels following the phase of the saddle factor) and one Filon part
  // per breakpoint of the kernel along the parabola X(v) = c0 v^2 + q v + 1/2.
  // Zero and stationary starts; a fixed start g0 != 0 uses the double integral above.

  // ---- shared coefficient builders ----
  function makeCoefs(cas, c) {
    const j = { DK: 1, DU: 2, DT: 4 }[cas], b = -c;
    const T = NT[j], AS = T.map(row => [0, 1, 2].map(mm => { if (mm >= row.length) return 0; let s = 0, bp = 1; for (let qq = 0; qq < row[mm].length; qq++) { s += row[mm][qq] * bp; bp *= b; } return s; }));
    function coefs(z) {
      // F0, F1, F2 of D e^{-c}, scaled by exp(-ls); ls = 0 (series) or theta = (-z)^(1/2)
      if (cabs(z) < 4) {
        const F = [0, 1, 2].map(mm => { let p = [0, 0]; for (let k = AS.length - 1; k >= 0; k--) p = cadd(cmul(p, z), [AS[k][mm], 0]); return p; });
        return { F, ls: [0, 0] };
      }
      const X = csc(cadd(z, [b * b, 0]), 0.5), th = csqrt(cneg(z)), E = cexp(csc(th, -2));
      const C = csc(cadd([1, 0], E), 0.5), S = cdiv(csub([1, 0], E), csc(th, 2)), one = cexp(cneg(th));
      const Cm1 = csub(C, one), iz = cinv(z), iz2 = csq(iz), iz3 = cmul(iz2, iz);
      let F0, F1, F2;
      if (j === 1) { F0 = cadd(C, csc(S, b)); F1 = S; F2 = [0, 0]; }
      else if (j === 2) {
        F2 = cadd(cmul(S, iz), csc(cmul(Cm1, iz2), 2));
        F1 = csc(F2, -b * b);
        F0 = csub(csub(cmul(cmul(csub(csc(X, 2), [b * b * b, 0]), S), iz), csc(cmul(C, iz), b * b)), csc(cmul(cmul(X, Cm1), iz2), 4 * b));
      } else {
        const Tt = cmul(cadd(S, cmul(Cm1, iz)), iz3), b2 = b * b, b4 = b2 * b2;
        F2 = cadd(cadd(csc(cmul(S, iz2), -4 * (b2 + 3 * b + 27)), cmul(cadd(csc(X, 192), [288 * (b + 1), 0]), Tt)),
          cadd(csc(cmul(C, iz3), -32 * (b2 + 3 * b + 6)), csc(cmul(one, iz3), -16 * (b2 + 3 * b - 3))));
        F1 = cadd(cadd(csc(cmul(S, iz2), b4), csc(Tt, -24 * b4)), cadd(csc(cmul(C, iz3), 8 * b4), csc(cmul(one, iz3), 4 * b4)));
        F0 = cadd(cadd(cmul(cmul(csub([b4 * b, 0], csc(X, 8 * (b2 - 3 * b - 3))), S), iz2), csc(cmul(csq(X), Tt), -96 * (b + 1))),
          cadd(cmul(cadd(csc(iz2, b4), csc(cmul(X, iz3), 16 * b2 * b)), C), csc(cmul(cmul(X, one), iz3), 8 * b2 * (b + 3))));
      }
      return { F: [F0, F1, F2], ls: th };
    }
    return { coefs, j, b };
  }
  // g(z) of the initial-value term (Nabeya-Sorensen), scaled by e^-ls as coefs: gsc(z) = g(z) e^-ls
  function makeG(cas) {
    const k = (cas === 'DT') ? 2 : 1, a = [];
    const lf = (n) => { let v = 0; for (let i = 2; i <= n; i++) v += Math.log(i); return v; };
    for (let jj = 0; jj < 24; jj++) {
      const sg = (jj % 2) ? -1 : 1;
      if (k === 1) a.push(sg * (2 * jj + 2) * Math.exp(-lf(2 * jj + 4)));
      else a.push(sg * (Math.exp(-lf(2 * jj + 5)) - 8 * Math.exp(-lf(2 * jj + 6)) + 24 * Math.exp(-lf(2 * jj + 7)) - 24 * Math.exp(-lf(2 * jj + 8))));
    }
    function gsc(z) {
      if (cabs(z) < 4) { let g = [0, 0], p = [1, 0]; for (let m = 0; m < a.length; m++) { g = cadd(g, csc(p, a[m])); p = cmul(p, z); } return g; }
      const th = csqrt(cneg(z)), e = cexp(csc(th, -2)), em = cexp(cneg(th));
      const Cs = csc(cadd([1, 0], e), 0.5), Ss = cdiv(csub([1, 0], e), csc(th, 2));
      const iz = cinv(z), iz2 = csq(iz), Cm = csub(Cs, em);
      if (k === 2) {
        const iz3 = cmul(iz2, iz), iz4 = csq(iz2);
        return csub(csub(csc(cmul(Cs, iz3), 12), cmul(cmul(cadd([24, 0], csc(z, 4)), Cm), iz4)), cmul(cmul(csub([24, 0], z), Ss), iz3));
      }
      return csub(cneg(cmul(Ss, iz)), csc(cmul(Cm, iz2), 2));
    }
    return { gsc, k };
  }

  // ---- the Girsanov forms of cases DU, DT at large c (IIPAR_DFX) ----
  const IIP_T = [1.0, -0.33333333333333333, 0.13333333333333333, -0.053968253968253968, 0.021869488536155203, -0.0088632355299021966, 0.003592128036572481, -0.0014558343870513183, 0.00059002744094558598, -0.00023912911424355248, 9.6915379569294503e-5, -3.9278323883316834e-5, 1.5918905069328965e-5, -6.4516892156554308e-6, 2.6147711512907546e-6, -1.0597268320104654e-6, 4.2949110782738059e-7, -1.7406618963571648e-7, 7.0546369464009683e-8, -2.8591366623052539e-8, 1.1587644432798852e-8, -4.6962953982309016e-9, 1.9033368339312759e-9, -7.7139336353590623e-10, 3.126339545892087e-10, -1.2670576930305401e-10, 5.1351914080393677e-11, -2.0812146867700474e-11, 8.4348454190943383e-12, -3.4185140868111558e-12, 1.3854715742948469e-12, -5.6151047924146801e-13, 2.2757162553728748e-13, -9.2231305851395317e-14, 3.7379940310967389e-14, -1.5149519187148605e-14, 6.1398688626168138e-15, -2.4883951222762789e-15, 1.0085085566354096e-15, -4.0873312268690138e-16, 1.6565329513786284e-16, -6.7136751750487117e-17, 2.7209500613044588e-17, -1.1027595233722371e-17, 4.4693160072374125e-18];
  const IIP_E = [1.0, -0.5, 0.20833333333333333, -0.084722222222222222, 0.034350198412698413, -0.013922233245149912, 0.0056424968100315323, -0.0022868190951648293, 0.00092681292737742184, -0.0003756231338525945, 0.00015223432221797662, -6.1698246877700525e-5, 2.5005357609459247e-5, -1.0134289721572026e-5, 4.1072729198567004e-6, -1.664615015128028e-6, 6.7464305456632013e-7, -2.7342253129896734e-7, 1.1081397802278207e-7, -4.4911213669537136e-8, 1.820182931124577e-8, -7.3769235610848765e-9, 2.9897545073926764e-9, -1.2117018619561619e-9, 4.9108426750009155e-10, -1.9902895700495881e-10, 8.066339801136952e-11, -3.2691643852499818e-11, 1.3249424201396147e-11, -5.3697893706595739e-12, 2.1762936597810882e-12, -8.8201859824934e-13, 3.5746867347671487e-13, -1.4487659644686842e-13, 5.871627293628006e-14, -2.3796809091881837e-14, 9.6444834564008509e-15, -3.9087619176859166e-15, 1.5841615363041244e-15, -6.4203648775599251e-16, 2.6020758752402585e-16, -1.0545816304260601e-16, 4.2740583616893926e-17, -1.7322106086512011e-17, 7.0203851674541611e-18, -2.8452549391662691e-18];
  const cpoly = (cf, start, q) => { let acc = [0, 0]; for (let i = cf.length - 1; i >= start; i--) acc = cadd(cmul(acc, q), [cf[i], 0]); return acc; };
  function det3(M) {
    return csub(cadd(cmul(M[0][0], csub(cmul(M[1][1], M[2][2]), cmul(M[1][2], M[2][1]))),
      cmul(M[0][2], csub(cmul(M[1][0], M[2][1]), cmul(M[1][1], M[2][0])))),
      cmul(M[0][1], csub(cmul(M[1][0], M[2][2]), cmul(M[1][2], M[2][0]))));
  }
  function adj3(M) {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    A[0][0] = csub(cmul(M[1][1], M[2][2]), cmul(M[1][2], M[2][1])); A[0][1] = cneg(csub(cmul(M[0][1], M[2][2]), cmul(M[0][2], M[2][1]))); A[0][2] = csub(cmul(M[0][1], M[1][2]), cmul(M[0][2], M[1][1]));
    A[1][0] = cneg(csub(cmul(M[1][0], M[2][2]), cmul(M[1][2], M[2][0]))); A[1][1] = csub(cmul(M[0][0], M[2][2]), cmul(M[0][2], M[2][0])); A[1][2] = cneg(csub(cmul(M[0][0], M[1][2]), cmul(M[0][2], M[1][0])));
    A[2][0] = csub(cmul(M[1][0], M[2][1]), cmul(M[1][1], M[2][0])); A[2][1] = cneg(csub(cmul(M[0][0], M[2][1]), cmul(M[0][1], M[2][0]))); A[2][2] = csub(cmul(M[0][0], M[1][1]), cmul(M[0][1], M[1][0]));
    return A;
  }
  function iipar(s, c, cas) {
    // s: complex mgf argument of V; returns {th, lcd, D, A2, A1, A0, dS, lch}
    const q = csub([c * c, 0], csc(s, 2)), th = csqrt(q);
    const del = (c > 0) ? cdiv(csc(s, -2), cadd(th, [c, 0])) : csub(th, [c, 0]);
    const sm = cabs(th) < 1, e2 = cexp(csc(th, -2));
    let g1, g2, g3, g4, g5, t1, lch;
    if (sm) {
      g1 = cpoly(IIP_T, 0, q); g2 = cneg(cpoly(IIP_E, 1, q)); g3 = cneg(cpoly(IIP_T, 1, q)); g4 = cpoly(IIP_E, 2, q); g5 = cpoly(IIP_T, 2, q);
      // the polys start at index 'start': p(q) = sum_{i>=start} cf[i] q^(i-start)
      t1 = csub([1, 0], cmul(th, g1)); lch = clog(csc(cadd(cexp(th), cexp(cneg(th))), 0.5));
    } else {
      const tn = cdiv(csub([1, 0], e2), cadd([1, 0], e2)), sc = cdiv(csc(cexp(cneg(th)), 2), cadd([1, 0], e2));
      const th2 = csq(th), th3 = cmul(th2, th), th4 = csq(th2), th5 = cmul(th4, th);
      g1 = cdiv(tn, th); g2 = cdiv(csub([1, 0], sc), th2); g3 = cdiv(csub(th, tn), th3);
      g4 = cdiv(cadd(csub(th2, [2, 0]), csc(sc, 2)), csc(th4, 2)); g5 = cdiv(cadd(csub(csc(th3, 1 / 3), th), tn), th5);
      t1 = cdiv(csc(e2, 2), cadd([1, 0], e2)); lch = cadd(th, clog(csc(cadd([1, 0], e2), 0.5)));
    }
    let N11 = sm ? csub([1, 0], csc(g1, c)) : cdiv(cadd(del, csc(t1, c)), th);
    let D, A2, A1, A0, dS;
    if (cas === 'DU') {
      const s11 = g1, s12 = g2, s22 = g3;
      dS = csub(cmul(s11, s22), csq(s12));
      D = cadd(cmul(N11, cadd([1, 0], csc(cmul(s, s22), 2))), csc(cmul(s, csq(s12)), 2 * c));
      A2 = cdiv(D, dS);
      A1 = cadd(cdiv(csub(s11, csc(s12, 2)), dS), csc(s, 2));
      A0 = [-1, 0];
    } else {
      const S = [[g1, g2, g3], [g2, g3, g4], [g3, g4, g5]];
      const Qd = [[[c, 0], [0, 0], [0, 0]], [[0, 0], csc(s, -8), csc(s, 12)], [[0, 0], csc(s, 12), csc(s, -24)]];
      const Nm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let jj = 0; jj < 3; jj++) {
        let v = cadd(cadd(cmul(Qd[i][0], S[0][jj]), cmul(Qd[i][1], S[1][jj])), cmul(Qd[i][2], S[2][jj]));
        Nm[i][jj] = csub(i === jj ? [1, 0] : [0, 0], v);
      }
      Nm[0][0] = N11;
      D = det3(Nm); dS = det3(S);
      A2 = cdiv(D, dS);
      const Si = adj3(S), Pp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let jj = 0; jj < 3; jj++) Pp[i][jj] = csub(cdiv(Si[i][jj], dS), Qd[i][jj]);
      Pp[0][0] = cdiv(cadd(cadd(cmul(Nm[0][0], Si[0][0]), cmul(Nm[0][1], Si[1][0])), cmul(Nm[0][2], Si[2][0])), dS);
      const K = [[1, 2, -6], [2, -12, 12], [-6, 12, 0]];
      const aK = [[K[1][1] * K[2][2] - K[1][2] * K[2][1], -(K[0][1] * K[2][2] - K[0][2] * K[2][1]), K[0][1] * K[1][2] - K[0][2] * K[1][1]],
        [-(K[1][0] * K[2][2] - K[1][2] * K[2][0]), K[0][0] * K[2][2] - K[0][2] * K[2][0], -(K[0][0] * K[1][2] - K[0][2] * K[1][0])],
        [K[1][0] * K[2][1] - K[1][1] * K[2][0], -(K[0][0] * K[2][1] - K[0][1] * K[2][0]), K[0][0] * K[1][1] - K[0][1] * K[1][0]]];
      const aP = adj3(Pp);
      A1 = [0, 0]; A0 = [0, 0];
      for (let i = 0; i < 3; i++) for (let jj = 0; jj < 3; jj++) { A1 = cadd(A1, csc(aP[jj][i], K[i][jj])); A0 = cadd(A0, csc(Pp[jj][i], aK[i][jj])); }
    }
    return { th, del, D, A2, A1, A0, dS, lch };
  }

  // ---- the single-integral engine ----
  function single(cas0, c, opts) {
    opts = opts || {};
    cas0 = cas0.toUpperCase();
    const stA = cas0[0] === 'S';
    if (stA && !(c < 0)) throw new Error('A stationary start requires c < 0.');
    const cas = stA ? 'D' + cas0[1] : cas0;
    const model = { DK: 1, DU: 2, DT: 3 }[cas];
    const c0 = (opts.c0 === undefined || opts.c0 === null) ? 0 : opts.c0;
    const { coefs } = makeCoefs(cas, c), G = (cas === 'DK') ? null : makeG(cas);
    const kern = (model === 1 && !stA) ? 1 : 2;
    const EXPO = 60, KAPPA = 6, NGROW = 16, NLEV = 5, NCON = 16;
    const nodefac = opts.nodefac || 1;
    const Z = [0, 0], ONE = [1, 0];
    const rcache = new Map();
    function rule(n) { if (!rcache.has(n)) rcache.set(n, gl01(n)); return rcache.get(n); }
    // D(s,t) scaled by e^(-c-ls): at s (mgf argument of U) and t, with the stationary term
    function Dsc(s, t) {
      const z = csub(csc(t, 2), [c * c, 0]), cf = coefs(z);
      let Dt = cadd(csub(cf.F[0], cmul(cf.F[1], s)), cmul(cf.F[2], csq(s)));
      if (stA) {
        if (model === 1) {
          // e^c {F0 + S (2t + s^2)/(2c)}: F1 = S
          Dt = cadd(cf.F[0], cdiv(cmul(cf.F[1], cadd(csc(t, 2), csq(s))), [2 * c, 0]));
        } else {
          const ck = (model === 2) ? c : c * c * c;
          Dt = cadd(Dt, csc(cmul(cadd(cadd(t, csc(csq(s), 0.5)), csc(s, c)), G.gsc(z)), ck));
        }
      }
      return { Dt, ls: cf.ls, F: cf.F };
    }
    function Dreal(t) { const d = Dsc(Z, [t, 0]); return cmul(cexp([0, d.ls[1]]), d.Dt)[0]; }   // the sign of D(0,t) (the real scaling e^{Re ls} > 0)
    function eta1(t) {
      const mu = csqrt([c * c - 2 * t, 0]);
      if (cabs(mu) < 1e-4) return 1 + csq(mu)[0] / 3 - c;
      const e2 = cexp(csc(mu, -2));
      return cadd(cdiv([-2 * t, 0], cadd(mu, [c, 0])), cdiv(csc(cmul(mu, e2), 2), csub(ONE, e2)))[0];
    }
    function tstar() {
      let g;
      if (model === 1 && !stA) g = eta1;
      else if (!stA && c > 2) g = (t) => iipar([t, 0], c, cas).D[0];
      else g = Dreal;
      let a = 1e-6;
      while (!(g(a) > 0) && a > 1e-300) a /= 100;
      if (!(g(a) > 0)) throw new Error('dfx: D(0,t) not positive near t = 0');
      let b = a;
      while (g(b) > 0) { a = b; b *= 1.05; if (b > 1e4) throw new Error('dfx: no zero of D(0,t) below 1e4'); }
      for (let it = 0; it < 100; it++) { const m = (a + b) / 2; if (g(m) > 0) a = m; else b = m; if (b - a < 1e-15 * b) break; }
      return (a + b) / 2;
    }
    let tau = opts.tau;
    if (!(tau > 0)) tau = (c <= 0 && !stA) ? 1 : Math.min(1, tstar() / 2);
    const unwrapc = (ph) => { for (let i = 1; i < ph.length; i++) ph[i] -= 2 * Math.PI * Math.round((ph[i] - ph[i - 1]) / (2 * Math.PI)); return ph; };
    // P(t), s1(t), s2(t), M0(t) on a grid of t (t[0] the real point), branches by continuity
    function tparts(ts) {
      const nt = ts.length, P = new Array(nt), s1 = new Array(nt), s2 = new Array(nt), M0 = new Array(nt);
      if (model === 1 && !stA) {
        // s1 = mu coth mu - c (cancellation-free for c > 0), P = e^(-c/2) S^(-1/2) with S = sinh mu/mu
        const lS = new Array(nt);
        for (let i = 0; i < nt; i++) {
          const t = ts[i], z = [c * c - 2 * t[0], -2 * t[1]], mu = csqrt(z);
          const cf = coefs(csub(csc(t, 2), [c * c, 0]));      // F1 = S e^-ls
          lS[i] = cadd(clog(cf.F[1]), cf.ls);
          if (c > 0) {
            const e2 = cexp(csc(mu, -2));
            let ep = cdiv(csc(cmul(mu, e2), 2), csub(ONE, e2));
            if (cabs(mu) < 1e-4) ep = cadd(csub(ONE, mu), csc(csq(mu), 1 / 3));
            s1[i] = cadd(cdiv(csc(t, -2), cadd(mu, [c, 0])), ep);
          } else {
            s1[i] = cdiv(cf.F[0], cf.F[1]);                   // (C - c S)/S
          }
          s2[i] = Z;
        }
        const ph = unwrapc(lS.map(v => v[1]));
        for (let i = 0; i < nt; i++) { P[i] = cexp([-c / 2 - lS[i][0] / 2, -ph[i] / 2]); M0[i] = cmul(P[i], cexp(csc(clog(s1[i]), -0.5))); }
        return { P, s1, s2, M0 };
      }
      if (!stA && c > 2) {
        // the Girsanov forms: A2 nu^2 - A1 nu + A0 = 0 has the roots nu = 2 kap = 1/s
        const lDs = new Array(nt), lms = new Array(nt), lch = new Array(nt);
        for (let i = 0; i < nt; i++) {
          const Q = iipar(ts[i], c, cas);
          const disc = csqrt(csub(csq(Q.A1), csc(cmul(Q.A0, Q.A2), 4)));
          const sw = Math.sign(cmul([Q.A1[0], -Q.A1[1]], disc)[0]) || 1;
          const rb = cdiv(cadd(Q.A1, csc(disc, sw)), csc(Q.A0, 2)), rs = cdiv(Q.A2, cmul(Q.A0, rb));
          let sa = rb, sb = rs;
          if (i === 0) { if (sa[0] < sb[0]) { s1[0] = sa; s2[0] = sb; } else { s1[0] = sb; s2[0] = sa; } }
          else if (cabs(csub(sa, s1[i - 1])) + cabs(csub(sb, s2[i - 1])) <= cabs(csub(sb, s1[i - 1])) + cabs(csub(sa, s2[i - 1]))) { s1[i] = sa; s2[i] = sb; }
          else { s1[i] = sb; s2[i] = sa; }
          lDs[i] = clog(Q.D); lms[i] = clog(cneg(cmul(s1[i], s2[i]))); lch[i] = Q.lch;
        }
        const phD = unwrapc(lDs.map(v => v[1])), phm = unwrapc(lms.map(v => v[1]));
        for (let i = 0; i < nt; i++) {
          M0[i] = cexp([-lms[i][0] / 2, -phm[i] / 2]);
          P[i] = cexp([-c / 2 - (lch[i][0] + lDs[i][0]) / 2 + lms[i][0] / 2, -(lch[i][1] + phD[i]) / 2 + phm[i] / 2]);
        }
        return { P, s1, s2, M0 };          // (s1 < 0 < s2 at the real point, the convention of DFX_CORE)
      }
      // D = a2 lam^2 + a1 lam + a0 (lam = -s), scaled; a2 from three values
      const lma = new Array(nt), lmm = new Array(nt);
      for (let i = 0; i < nt; i++) {
        const t = ts[i];
        const d0 = Dsc(Z, t), dp = Dsc([1, 0], t).Dt, dm = Dsc([-1, 0], t).Dt;   // at s = 0, -1 (lam = 1), 1 (lam = -1)
        // Dsc takes s; lam = -s: dp := D(lam = 1) = Dsc(s = -1)... keep MATLAB's naming: dp = D(lam=+1), dm = D(lam=-1)
        const Dp = dm, Dm = dp;                                 // (Dsc(-1) is lam = +1)
        const a0 = d0.Dt, a2 = csub(csc(cadd(Dp, Dm), 0.5), a0), a1 = csc(csub(Dp, Dm), 0.5);
        const disc = csqrt(csub(csq(a1), csc(cmul(a2, a0), 4)));
        const sw = Math.sign(cmul([a1[0], -a1[1]], disc)[0]) || 1;
        const rb = cdiv(cneg(cadd(a1, csc(disc, sw))), csc(a2, 2)), rs = cdiv(a0, cmul(a2, rb));
        const sa = cneg(rb), sb = cneg(rs);
        if (i === 0) { if (sa[0] < sb[0]) { s1[0] = sa; s2[0] = sb; } else { s1[0] = sb; s2[0] = sa; } }
        else if (cabs(csub(sa, s1[i - 1])) + cabs(csub(sb, s2[i - 1])) <= cabs(csub(sb, s1[i - 1])) + cabs(csub(sa, s2[i - 1]))) { s1[i] = sa; s2[i] = sb; }
        else { s1[i] = sb; s2[i] = sa; }
        const ma2 = cneg(a2);                                  // the scaled leading coefficient -a2 e^(c+ls)
        lma[i] = cadd(clog(ma2), [c + d0.ls[0], d0.ls[1]]);
        lmm[i] = clog(cneg(cmul(s1[i], s2[i])));
      }
      if (Math.abs(lma[0][1]) > 1e-8) throw new Error('dfx: prefactor not positive at the real point');
      const pha = unwrapc(lma.map(v => v[1])), phm = unwrapc(lmm.map(v => v[1]));
      for (let i = 0; i < nt; i++) { P[i] = cexp([-lma[i][0] / 2, -pha[i] / 2]); M0[i] = cexp([-lmm[i][0] / 2, -phm[i] / 2]); }
      return { P, s1, s2, M0 };
    }

    // ---- e^z K_0(z), phi_t ----
    const k0x = besselk0s;
    function phi_pos(X, s1, s2) {
      if (kern === 1) return cdiv(cexp(cneg(cmul(s1, X))), csqrt(csc(X, Math.PI)));
      return csc(cmul(k0x(cmul(csc(csub(s2, s1), 0.5), X)), cexp(cneg(cmul(s2, X)))), 1 / Math.PI);
    }
    function phi_neg(X, s1, s2) {
      if (kern === 1) return Z;
      return csc(cmul(k0x(cmul(csc(csub(s2, s1), -0.5), X)), cexp(cneg(cmul(s1, X)))), 1 / Math.PI);
    }
    function phialg(X, sgn, s1, s2) {
      if (kern === 1) return cinv(csqrt(csc(X, Math.PI)));
      return csc(k0x(cmul(csc(csub(s2, s1), sgn > 0 ? 0.5 : -0.5), X)), 1 / Math.PI);
    }
    function phisg(X, sgn, s1, s2) { return sgn > 0 ? phi_pos(X, s1, s2) : phi_neg(X, s1, s2); }

    // ---- quadrature ----
    function glnodes(a, b, n, p) {            // nodes/weights on [a,b] with y = a + (b-a) u^p; real a, b, L
      n = n * nodefac;
      const { x: u, w: gw } = rule(n), L = b - a, y = new Array(n), wt = new Array(n);
      for (let i = 0; i < n; i++) {
        if (p === 1) { y[i] = a + L * u[i]; wt[i] = L * gw[i]; }
        else if (p === 2) { y[i] = a + L * u[i] * u[i]; wt[i] = 2 * L * u[i] * gw[i]; }
        else { const u3 = u[i] * u[i] * u[i]; y[i] = a + L * u3 * u[i]; wt[i] = 4 * L * u3 * gw[i]; }
      }
      return { y, wt };
    }
    function glsum(f, a, b, n, p) { const { y, wt } = glnodes(a, b, n, p); let acc = Z; for (let i = 0; i < y.length; i++) acc = cadd(acc, csc(f(y[i]), wt[i])); return acc; }
    const nodes = (n, osc, L) => Math.max(n, 16 * Math.ceil(Math.ceil(1.2 * osc * L) / 16));
    function rayint(f, R, l, n1, n2, p, osc) {   // the Gaussian ray from r = 0 (real r)
      osc = osc || 0;
      let a = Math.min(R, 8 * l);
      let acc = glsum(f, 0, a, n1, p);
      while (a < R && a > 0) {
        const b = Math.min(R, 4 * a);
        acc = cadd(acc, glsum(f, a, b, nodes(n2, osc, b - a), 1));
        a = b;
      }
      return acc;
    }
    const cap = (x) => x / Math.sqrt(1 + x * x);
    function levelf(s, A, vmin, sd, t, q) {
      const v = vmin * Math.sinh(s), dv = vmin * Math.cosh(s);
      const rt = cmul(sd, csqrt(cadd(ONE, cdiv(csc(t, 4 * v * v / (q * q)), csq(sd)))));
      const x = cdiv([2 * v * v, 0], cadd(sd, rt));
      return cmul(A(x), cdiv([Math.exp(-v * v) * 2 * v * dv, 0], rt));
    }
    function gaussexp(A, t, q, sd, lA, force, p) {
      // int_0^inf A(x) exp(-t x^2/q^2 - sd x) dx on the saddle path (Re z < 0) or the level curve
      force = force || 0; p = p || 1;
      const at = cabs(t), e = cexp([0, -carg(t) / 2]);
      const x0 = cdiv(csc(sd, -q * q), csc(t, 2)), u0 = cdiv(csc(csq(sd), -q * q), csc(t, 4));
      if (force === 1 || (force === 0 && cmul(sd, e)[0] < 0)) {
        const ru = u0[0];
        const lam_max = ru > EXPO ? 1 - Math.sqrt(1 - EXPO / ru) : 1;
        const f1 = (lam) => cmul(cmul(A(csc(x0, lam)), cexp(csc(u0, -lam * (2 - lam)))), x0);
        const l = Math.min(lA / cabs(x0), 1 / (2 * cabs(u0) + 1));
        let acc = rayint(f1, lam_max, l, 48, 32, p, 2 * Math.abs(u0[1]));
        if (ru > EXPO) return acc;
        const f2 = (r) => cmul(cmul(A(cadd(x0, csc(e, r))), cexp([-u0[0] - at * r * r / (q * q), -u0[1]])), e);
        return cadd(acc, rayint(f2, q * Math.sqrt(EXPO / at), lA, 48, 32, 1, 0));
      }
      const v0 = cabs(sd) * q / (2 * Math.sqrt(at)), vA = Math.sqrt(lA * cabs(sd));
      const vmin = Math.max(Math.min(v0, vA, 1) / 2, 1e-4);
      const S = Math.asinh(Math.sqrt(EXPO) / vmin);
      const lf = (s) => levelf(s, A, vmin, sd, t, q);
      if (p === 1) return glsum(lf, 0, S, 16 * Math.ceil(Math.ceil(13 * S) / 16), 1);
      const S1 = Math.min(1, S);
      let acc = glsum(lf, 0, S1, 32, p);
      if (S > 1) acc = cadd(acc, glsum(lf, 1, S, 16 * Math.ceil(Math.ceil(13 * (S - 1)) / 16), 1));
      return acc;
    }
    function ray_ok(sd, e, q, at) {
      const se = cmul(sd, e);
      return Math.pow(Math.max(0, -se[0]), 2) * q * q / (4 * at) <= 2 && Math.abs(se[1]) * q * Math.sqrt(EXPO / at) <= 60;
    }
    function rayexp(A, alpha, beta, lA, p) {
      if (!(alpha[0] > 0)) throw new Error('dfx: Re alpha <= 0 on a ray');
      const e = cexp([0, -carg(alpha) / 2]), aa = cabs(alpha);
      if (ray_ok(beta, e, 1, aa)) {
        let Rr = Math.sqrt(EXPO / aa);
        const be = cmul(beta, e);
        if (be[0] > 0) Rr = Math.min(Rr, EXPO / be[0]);
        const f = (r) => cmul(cmul(A(csc(e, r)), cexp(csub([-aa * r * r, 0], csc(be, r)))), e);
        return rayint(f, Rr, lA, 48, 32, p, Math.abs(be[1]));
      }
      return gaussexp(A, alpha, 1, beta, lA, 0, p);
    }
    // Psi_t(1/2)
    function psi_half(s1, s2, M0) {
      if (kern === 1) return cdiv(erfz(csqrt(csc(s1, 0.5))), csqrt(s1));
      const l = 1 / (cabs(csub(s2, s1)) + 2);
      if (cabs(s2) >= cabs(s1) / 4) {
        const e2 = cexp([0, -carg(s2)]);
        return csub(M0, rayint((r) => cmul(phi_pos(cadd([0.5, 0], csc(e2, r)), s1, s2), e2), 36 / cabs(s2), l, 40, 32, 1, 0));
      }
      const e1 = cexp([0, -carg(cneg(s1))]);
      return cadd(rayint((X) => phi_pos([X, 0], s1, s2), 0.5, l, 40, 32, 4, 0),
        rayint((r) => cmul(phi_neg(csc(e1, -r), s1, s2), e1), 36 / cabs(s1), l, 40, 32, 4, 0));
    }

    // ---- the breakpoints of the kernel along the parabola ----
    function breakpts(q) {
      const R = { q, c0 };
      let rr, lead, r;
      if (c0 === 0) { rr = [[-1 / (2 * q), 0]]; lead = q; r = [-1 / (2 * q)]; }
      else {
        const d = q * q - 2 * c0;
        const sq = d >= 0 ? [Math.sqrt(d), 0] : [0, Math.sqrt(-d)];
        rr = [csc(csub([-q, 0], sq), 1 / (2 * c0)), csc(cadd([-q, 0], sq), 1 / (2 * c0))]; lead = c0;
        r = d >= 0 ? [rr[0][0], rr[1][0]].sort((u, v) => u - v) : [];
      }
      R.rr = rr; R.lead = lead;
      R.Xat = (dv, b) => { let X = [lead, 0]; for (const rk of rr) X = cmul(X, cadd(dv, csub([b, 0], rk))); return X; };   // dv complex, b real
      // roots so far out that the Gaussian e^(-t v^2) is below e^-40 there (3 tau b^2/4 > 40)
      // are dropped: the part near them is negligible, and the segment up to them would be
      // integrated at a cost that grows with b (for q -> 0- the root is at -1/(2q))
      const noroot = r.filter(v => v > 0).length === 0;
      let b = r.filter(v => v > 0 && 3 * tau * v * v / 4 <= 40), kindb = b.map(() => 1), ell = b.map(() => Infinity);
      R.double = false;
      if (b.length === 2 && b[1] - b[0] <= 1e-9 * b[1]) { b = [(b[0] + b[1]) / 2]; kindb = [1]; ell = [Infinity]; R.double = true; }
      else if (b.length === 2) { ell = [b[1] - b[0], b[1] - b[0]]; }
      if (c0 > 0 && q < 0 && noroot) { b = [-q / (2 * c0)]; kindb = [0]; ell = [Math.sqrt(Math.max(0.5 - q * q / (4 * c0), 0) / c0)]; }
      R.dX = (v) => 2 * c0 * v + q;
      for (let k = 0; k < b.length; k++) {
        const dr = rr.map(rk => cabs(csub([b[k], 0], rk))).filter(v => v > 1e-9 * Math.max(b[k], 1));
        if (dr.length) ell[k] = Math.min(ell[k], Math.min(...dr));
      }
      ell = ell.map((v, k) => Math.max(v, 1e-8 * Math.max(b[k], 1)));
      R.ell0 = Math.min(...rr.map(cabs));
      R.b = b; R.y = b.map(v => v * v); R.kindb = kindb; R.ell = ell; R.ne = b.length;
      const ed = [0].concat(b, [Infinity]); R.sgn = [];
      for (let jj = 0; jj < b.length + 1; jj++) {
        const vm = isFinite(ed[jj + 1]) ? (ed[jj] + ed[jj + 1]) / 2 : ed[jj] + 1;
        const sv = Math.sign(R.Xat([vm, 0], 0)[0]); R.sgn.push(sv !== 0 ? sv : 1);
      }
      R.negl = b.map(() => false);
      return R;
    }
    function saddle_loc(q, R) {
      if (R.ne === 0) return NaN;
      const tp = tparts([[tau, 0], [tau, 1]]);
      const sd = (kern === 1 || R.sgn[0] < 0) ? tp.s1[1] : tp.s2[1];
      const vs = cdiv(csc(sd, -q), csc(cadd([tau, 1], csc(sd, c0)), 2));
      if (!(vs[0] > 0 && vs[0] < R.b[0] && Math.abs(vs[1]) < 0.2 * (R.b[0] - vs[0]))) return NaN;
      return vs[0];
    }
    function saddle_phase_gen(q, ts, s1, s2, R) {
      const psi = new Array(ts.length).fill(NaN);
      for (let i = 0; i < ts.length; i++) {
        const sd = (kern === 1 || R.sgn[0] < 0) ? s1[i] : s2[i];
        const alpha = cadd(ts[i], csc(sd, c0)), beta = csc(sd, q), z2 = cdiv(csq(beta), csc(alpha, 4));
        let m;
        if (R.ne === 0) { const z = cdiv(beta, csc(csqrt(alpha), 2)); m = z[0] < 0 && z2[0] > -30; }
        else { const vs = cdiv(beta, csc(alpha, -2)); m = vs[0] > 0 && vs[0] < R.b[0] && z2[0] > -30; if (isNaN(R.vsad)) m = false; }
        if (m) psi[i] = z2[1];
      }
      return psi;
    }
    function phase_steps(psi) { const st = []; for (let i = 1; i < psi.length; i++) { const d = Math.abs(psi[i] - psi[i - 1]); st.push(isNaN(d) ? 0 : d); } return st; }

    // ---- the nodes of the segment pieces ----
    function pieces_nodes(Ef, L, first, p, nmin) {
      first = Math.min(first, L);
      for (let k = 0; k < 60; k++) {
        const E = []; for (let i = 0; i <= 8; i++) E.push(Ef(first * i / 8));
        let dph = 0; for (let i = 1; i <= 8; i++) dph += Math.abs(E[i][1] - E[i - 1][1]);
        if ((E[8][0] > -4 && dph < 40) || first <= 1e-12 * L) break;
        first /= 2;
      }
      let { y, wt } = glnodes(0, first, nmin, p);
      let a = first;
      while (a < L) {
        const bn = Math.min(L, 4 * a);
        const E = []; for (let i = 0; i <= 8; i++) E.push(Ef(a + (bn - a) * i / 8));
        if (E.every(v => v[0] < -EXPO)) break;
        let dph = 0; for (let i = 1; i <= 8; i++) dph += cabs(csub(E[i], E[i - 1]));
        let n = Math.max(NGROW, 16 * Math.ceil((1.2 * dph + 16) / 16));
        const m = Math.ceil(n / 160); n = 16 * Math.ceil(n / m / 16);
        for (let k = 1; k <= m; k++) { const g = glnodes(a + (bn - a) * (k - 1) / m, a + (bn - a) * k / m, n, 1); y = y.concat(g.y); wt = wt.concat(g.wt); }
        if (E[8][0] < -EXPO) break;
        a = bn;
      }
      return { y, wt };
    }
    function phase_nodes(Ef, E1, E2) {
      const g = []; for (let k = -NLEV; k <= -1; k++) g.push(Math.pow(2, k));
      const ss = [0].concat(g, g.slice(0, -1).reverse().map(v => 1 - v), [1]), ns = ss.length - 1;
      const Emax = Math.max(E1[0], E2[0]) - EXPO + 2;
      let y = [], wt = [];
      for (let k = 0; k < ns; k++) {
        const E = []; for (let i = 0; i <= 8; i++) E.push(Ef(ss[k] + (ss[k + 1] - ss[k]) * i / 8));
        if (Math.max(...E.map(v => v[0])) < Emax) continue;
        let dph = 0; for (let i = 1; i <= 8; i++) dph += cabs(csub(E[i], E[i - 1]));
        let n = Math.max(NCON, 8 * Math.ceil((1.2 * dph + 16) / 8));
        const m = Math.ceil(n / 160); n = 8 * Math.ceil(n / m / 8);
        for (let i = 1; i <= m; i++) { const gg = glnodes(ss[k] + (ss[k + 1] - ss[k]) * (i - 1) / m, ss[k] + (ss[k + 1] - ss[k]) * i / m, n, 1); y = y.concat(gg.y); wt = wt.concat(gg.wt); }
      }
      return { y, wt };
    }
    function descent(gam, alpha, dir, L, dep, sg, deg) {
      let u = cabs(gam) > 0 ? csc([gam[0], -gam[1]], -1 / cabs(gam)) : [0, -sg];
      const cmax = Math.cos(deg * Math.PI / 180);
      if (u[0] * dir < cmax) { let si = Math.sign(u[1]); if (si === 0) si = -sg; u = [dir * cmax, si * Math.sin(deg * Math.PI / 180)]; }
      const g1 = cmul(gam, u)[0], g2 = -cmul(alpha, csq(u))[0];
      let l = L / 2;
      if (Math.abs(u[1]) > 0) l = Math.min(l, dep / Math.abs(u[1]));
      if (g1 < 0) {
        if (g2 > 0) l = Math.min(l, -g1 / (2 * g2));
        const disc = g1 * g1 - 4 * g2 * EXPO;
        if (g2 === 0) l = Math.min(l, -EXPO / g1);
        else if (disc >= 0) { const sE = (-g1 - Math.sqrt(disc)) / (2 * g2); if (sE > 0) l = Math.min(l, sE); }
      }
      return { u, l: Math.max(l, 1e-12 * L) };
    }
    function segint(tj, sd, sgn, s1, s2, W, R, a, b, chi, pa, pb, ella, ellb, ws, wl, wr, sg, ka, kb, dep) {
      const L = b - a, Xat = R.Xat;
      const Ez = (z, o) => csub(csc(cmul(tj, csq(cadd([o, 0], z))), -1), cmul(sd, Xat(z, o)));
      const Fz = (z, o) => cmul(cmul(W(cadd([o, 0], z)), cexp(Ez(z, o))), phialg(Xat(z, o), sgn, s1, s2));
      const Eb = Ez(Z, b), Ea = Ez(Z, a);
      const alpha = cadd(tj, csc(sd, c0));
      let dega = 80, fa = ella / 2; if (ka === 0) { dega = 45; fa = ella; }
      let degb = 80, fb = ellb / 2; if (kb === 0) { degb = 45; fb = ellb; }
      const da = descent(csub(csc(tj, -2 * a), csc(sd, R.dX(a))), alpha, 1, L, dep, sg, dega);
      const db = descent(csub(csc(tj, -2 * b), csc(sd, R.dX(b))), alpha, -1, L, dep, sg, degb);
      const ua = da.u, la = da.l, ub = db.u, lb = db.l;
      const pa_ = cadd([a, 0], csc(ua, la)), pb_ = cadd([b, 0], csc(ub, lb));
      const fv = [], v = [];
      let g = pieces_nodes((s) => csub(Ez(csc(ua, s), a), Ea), la, Math.min(la / 8, fa), pa, 48 + 16 * (pa === 1 ? 1 : 0));
      for (let i = 0; i < g.y.length; i++) { fv.push(cmul(Fz(csc(ua, g.y[i]), a), csc(ua, g.wt[i]))); v.push(cadd([a, 0], csc(ua, g.y[i]))); }
      g = pieces_nodes((s) => csub(Ez(csc(ub, s), b), Eb), lb, Math.min(lb / 8, fb), pb, 48);
      for (let i = 0; i < g.y.length; i++) { fv.push(cneg(cmul(Fz(csc(ub, g.y[i]), b), csc(ub, g.wt[i])))); v.push(cadd([b, 0], csc(ub, g.y[i]))); }
      if (cabs(csub(pb_, pa_)) > 0) {
        const Emax = Math.max(Ea[0], Eb[0]) + 2;
        let legs = [pa_, pb_];
        let rise = false;
        for (let i = 0; i <= 32; i++) { const pt = cadd(pa_, csc(csub(pb_, pa_), i / 32)); if (Ez(csub(pt, [a, 0]), a)[0] > Emax) { rise = true; break; } }
        if (rise) legs = [pa_, [pa_[0], 0], [pb_[0], 0], pb_];
        for (let k = 0; k < legs.length - 1; k++) {
          const dv = csub(legs[k + 1], legs[k]);
          if (cabs(dv) === 0) continue;
          const pn = phase_nodes((s) => Ez(csub(cadd(legs[k], csc(dv, s)), [a, 0]), a), Ea, Eb);
          for (let i = 0; i < pn.y.length; i++) {
            const vc = cadd(legs[k], csc(dv, pn.y[i]));
            const nb = cabs(csub(vc, [b, 0])) < cabs(csub(vc, [a, 0]));
            const fval = nb ? Fz(csub(vc, [b, 0]), b) : Fz(csub(vc, [a, 0]), a);
            fv.push(cmul(fval, csc(dv, pn.wt[i]))); v.push(vc);
          }
        }
      }
      let vs = Z, vl = Z, vr = Z;
      for (let i = 0; i < v.length; i++) {
        const cw = chi(v[i]);
        if (ws || wl) { const t = cmul(fv[i], csub(ONE, cw)); if (ws) vs = cadd(vs, t); else vl = cadd(vl, t); }
        if (wr) vr = cadd(vr, cmul(fv[i], cw));
      }
      return { vs, vl, vr };
    }
    // ---- the inner transforms of the general kernel ----
    function inner_gen(tj, R, s1, s2, M0, kind, part) {
      const q = R.q, ne = R.ne, b = R.b;
      const sg = tj[1] >= 0 ? 1 : -1;
      let Is = Z; const Ie = new Array(ne).fill(Z);
      const W = kind === 1 ? (v) => csc(csq(v), 2) : (v) => cadd([q, 0], csc(v, 2 * c0));
      const wants = part === 0 || part === 1;
      const want = (k) => (part === 0 || part === 2 || part === 10 + k) && !R.negl[k];
      if (kind === 2 && wants) Is = psi_half(s1, s2, M0);
      const lX = kern === 1 ? 1 / (cabs(s1) + 2) : 1 / (cabs(csc(csub(s2, s1), 0.5)) + 2);
      const ed = [0].concat(b, [Infinity]);
      for (let jj = 0; jj < ne + 1; jj++) {
        const a = ed[jj], bb = ed[jj + 1], sgn = R.sgn[jj];
        if (kern === 1 && sgn < 0) continue;
        const sd = (kern === 1 || sgn < 0) ? s1 : s2;
        if (!isFinite(bb)) {
          if (jj === 0) { if (!wants) continue; } else if (!want(jj - 1)) continue;
          const alpha = cadd(tj, csc(sd, c0)), beta = cadd(csc(alpha, 2 * a), csc(sd, q));
          const cst = cexp(csub(csc(tj, -a * a), cmul(sd, R.Xat(Z, a))));
          const A = (rho) => cmul(W(cadd([a, 0], rho)), phialg(R.Xat(rho, a), sgn, s1, s2));
          let lv = lX / Math.max(Math.abs(R.dX(a)), 1e-300);
          if (c0 !== 0) lv = Math.min(lv, Math.sqrt(lX / Math.abs(c0)));
          lv = Math.min(lv, R.ell0 / 16);
          let p = 1, val;
          if (jj > 0 && R.kindb[jj - 1] === 0) {
            const e = cexp([0, -carg(alpha) / 2]), aa = cabs(alpha), be = cmul(beta, e);
            const Rr = (-be[0] + Math.sqrt(be[0] * be[0] + 4 * aa * EXPO)) / (2 * aa);
            const pn = pieces_nodes((r) => csub([-aa * r * r, 0], csc(be, r)), Rr, Math.min(Rr / 8, R.ell[jj - 1]), 1, 48);
            let acc = Z;
            for (let i = 0; i < pn.y.length; i++) { const s = pn.y[i]; acc = cadd(acc, csc(cmul(A(csc(e, s)), cexp(csub([-aa * s * s, 0], csc(be, s)))), pn.wt[i])); }
            val = cmul(cmul(cst, acc), e);
          } else {
            if (jj > 0) { lv = Math.min(lv, R.ell[jj - 1] / 16); p = 4; }
            val = cmul(cst, rayexp(A, alpha, beta, lv, p));
          }
          if (jj === 0) Is = cadd(Is, val); else Ie[jj - 1] = cadd(Ie[jj - 1], val);
        } else {
          const L = bb - a, h = L / 2; let kap = KAPPA / h, m = a + h, dep = L / 8;
          if (jj === 0 && R.vsad > 0 && R.vsad < bb) {
            m = Math.max(m, (R.vsad + bb) / 2); kap = Math.max(kap, KAPPA / (bb - m));
            kap = Math.min(kap, 4 * KAPPA / L); m = Math.min(m, bb - KAPPA / kap);
            dep = Math.min(dep, KAPPA / 4 / kap);
          }
          const chi = (v) => csc(cadd(ONE, erfz(csc(csub(v, [m, 0]), kap))), 0.5);
          const pa = jj > 0 ? 4 : 1;
          const ws_ = jj === 0 && wants, wl_ = jj > 0 && want(jj - 1), wr_ = want(jj);
          if (!(ws_ || wl_ || wr_)) continue;
          const ella = jj > 0 ? R.ell[jj - 1] : R.ell0, ka = jj > 0 ? R.kindb[jj - 1] : 1;
          const r = segint(tj, sd, sgn, s1, s2, W, R, a, bb, chi, pa, 1 + 3 * R.kindb[jj], ella, R.ell[jj], ws_, wl_, wr_, sg, ka, R.kindb[jj], dep);
          if (ws_) Is = cadd(Is, r.vs);
          if (wl_) Ie[jj - 1] = cadd(Ie[jj - 1], r.vl);
          if (wr_) Ie[jj] = cadd(Ie[jj], r.vr);
        }
      }
      return { Is, Ie };
    }

    // ---- Filon's method ----
    const PMcache = new Map();
    function filonPM(N) {
      if (PMcache.has(N)) return PMcache.get(N);
      const { x: u, w: gw } = rule(N), x = u.map(v => 2 * v - 1), ww = gw.map(v => 2 * v);
      const Pm = []; for (let i = 0; i < N; i++) { Pm.push(new Array(N).fill(0)); Pm[i][0] = 1; if (N > 1) Pm[i][1] = x[i]; }
      for (let n = 1; n <= N - 2; n++) for (let i = 0; i < N; i++) Pm[i][n + 1] = ((2 * n + 1) * x[i] * Pm[i][n] - n * Pm[i][n - 1]) / (n + 1);
      // PM[n][i] = Pm[i][n] ww[i] (2n+1)/2 * 2 (-i)^n   (complex)
      const PM = [];
      for (let n = 0; n < N; n++) {
        const row = [], ph = [[1, 0], [0, -1], [-1, 0], [0, 1]][n % 4];
        for (let i = 0; i < N; i++) row.push(csc(ph, Pm[i][n] * ww[i] * (2 * n + 1)));
        PM.push(row);
      }
      PMcache.set(N, PM); return PM;
    }
    function sphj(nmax, x) {                   // spherical Bessel j_n(x), n = 0..nmax, x > 0
      const j = new Array(nmax + 1);
      if (x < nmax + 10) {                     // Miller's downward recurrence
        let M = nmax + Math.max(20, Math.ceil(x)) + 20; let jp = 0, jc = 1e-300; const tmp = new Array(M + 1);
        tmp[M] = jc;
        for (let n = M; n >= 1; n--) { const jm = (2 * n + 1) / x * jc - jp; jp = jc; jc = jm; tmp[n - 1] = jc; if (Math.abs(jc) > 1e250) { for (let k = n - 1; k <= M; k++) tmp[k] *= 1e-250; jc *= 1e-250; jp *= 1e-250; } }
        const sc = (Math.sin(x) / x) / tmp[0];
        for (let n = 0; n <= nmax; n++) j[n] = tmp[n] * sc;
      } else {
        j[0] = Math.sin(x) / x; if (nmax >= 1) j[1] = (Math.sin(x) / x - Math.cos(x)) / x;
        for (let n = 1; n < nmax; n++) j[n + 1] = (2 * n + 1) / x * j[n] - j[n - 1];
      }
      return j;
    }
    function filon_panel(G, a, b, y, N) {     // int_a^b G(w) e^(-i y w) dw, G at the N GL nodes (complex)
      const PM = filonPM(N), L = b - a, Om = y * L / 2;
      let jn; if (Om === 0) { jn = new Array(N).fill(0); jn[0] = 1; } else jn = sphj(N - 1, Om);
      let acc = Z;
      for (let n = 0; n < N; n++) { let cn = Z; for (let i = 0; i < N; i++) cn = cadd(cn, cmul(PM[n][i], G[i])); acc = cadd(acc, csc(cn, jn[n])); }
      return csc(cmul(cexp([0, -y * (a + b) / 2]), acc), L / 2);
    }

    // ---- the outer integral ----
    function outer_gen(q, R, kd) {
      const WMAX = opts.wmax || 4900, hu = opts.hu || 0.075, N = opts.N || 20, cde = tau;
      const weight = kd === 1 ? (t) => ONE : (t) => cinv(t);
      const inner = (t, s1, s2, M0, part) => inner_gen(t, R, s1, s2, M0, kd, part);
      const umax = Math.asinh(WMAX / cde);
      const us = []; for (let u = 0; u <= umax + 1e-12; u += hu) us.push(u);
      let ts = us.map(u => [tau, cde * Math.sinh(u)]);
      let tp = tparts(ts);
      const psi = saddle_phase_gen(q, ts, tp.s1, tp.s2, R);
      let val = 0;
      if (Math.max(0, ...phase_steps(psi)) > 0.25) {
        const w0 = Math.min(0.5, tau / 2);
        const edges0 = [0]; for (let k = 0; k <= Math.ceil(Math.log2(WMAX / w0)); k++) edges0.push(w0 * Math.pow(2, k)); edges0[edges0.length - 1] = WMAX;
        const K = 8, wk = [];
        for (let p = 0; p < edges0.length - 1; p++) for (let i = 0; i < K; i++) wk.push(edges0[p] + (edges0[p + 1] - edges0[p]) * i / K);
        wk.push(WMAX);
        const te0 = wk.map(w => [tau, w]), tpe = tparts(te0);
        const steps = phase_steps(saddle_phase_gen(q, te0, tpe.s1, tpe.s2, R));
        const wn = [], wt = [];
        for (let p = 0; p < edges0.length - 1; p++) {
          const a = edges0[p], b = edges0[p + 1];
          let dp = 0; for (let i = 0; i < K; i++) dp += steps[p * K + i];
          const m = Math.max(1, Math.ceil(dp / 50));
          const n = Math.max(24, 16 * Math.ceil(Math.ceil(3 * dp / m) / 16)) * nodefac * (opts.glfac || 1);
          const { x: u, w: gw } = rule(n);
          for (let k = 1; k <= m; k++) { const aa = a + (b - a) * (k - 1) / m, bb = a + (b - a) * k / m; for (let i = 0; i < n; i++) { wn.push(aa + (bb - aa) * u[i]); wt.push((bb - aa) * gw[i]); } }
        }
        ts = [[tau, 0]].concat(wn.map(w => [tau, w])); tp = tparts(ts);
        for (let jj = 0; jj < wn.length; jj++) {
          const t = ts[jj + 1], Is = inner(t, tp.s1[jj + 1], tp.s2[jj + 1], tp.M0[jj + 1], 1).Is;
          val += cmul(cmul(tp.P[jj + 1], Is), weight(t))[0] * wt[jj];
        }
        val /= Math.PI;
      } else {
        let acc = 0;
        for (let jj = 0; jj < ts.length; jj++) {
          const Is = inner(ts[jj], tp.s1[jj], tp.s2[jj], tp.M0[jj], 1).Is;
          let g = cmul(cmul(tp.P[jj], Is), weight(ts[jj]))[0] * cde * Math.cosh(us[jj]);
          if (jj === 0) g *= 0.5;
          acc += g;
        }
        val = acc * hu / Math.PI;
      }
      const u20 = rule(N).x;
      for (let k = 0; k < R.ne; k++) {
        if (R.negl[k]) continue;
        const y = R.y[k], w0 = Math.min(0.5, 1 / y, tau / 2);
        const edges0 = [0]; for (let kk = 0; kk <= Math.ceil(Math.log2(WMAX / w0)); kk++) edges0.push(w0 * Math.pow(2, kk)); edges0[edges0.length - 1] = WMAX;
        let edges = [0];
        for (let p = 0; p < edges0.length - 1; p++) {
          const a = edges0[p], b = edges0[p + 1], m = b <= 300 / y ? Math.ceil((b - a) * y / 25) : 1;
          for (let i = 1; i <= m; i++) edges.push(a + (b - a) * i / m);
        }
        const pfac = opts.pfac || 1;
        if (pfac > 1) { const e2 = []; for (let p = 0; p < edges.length - 1; p++) for (let i = 0; i < pfac; i++) e2.push(edges[p] + (edges[p + 1] - edges[p]) * i / pfac); edges = e2.concat([edges[edges.length - 1]]); }
        const np = edges.length - 1, wn = [];
        for (let p = 0; p < np; p++) for (let i = 0; i < N; i++) wn.push(edges[p] + (edges[p + 1] - edges[p]) * u20[i]);
        const te = [[tau, 0]].concat(wn.map(w => [tau, w])), tpe = tparts(te);
        const Gv = [];
        for (let jj = 0; jj < wn.length; jj++) {
          const t = te[jj + 1], Ie = inner(t, tpe.s1[jj + 1], tpe.s2[jj + 1], tpe.M0[jj + 1], 10 + k).Ie[k];
          Gv.push(cmul(cmul(cmul(tpe.P[jj + 1], Ie), weight(t)), cexp(csc(t, y))));
        }
        let acc = Z;
        for (let p = 0; p < np; p++) acc = cadd(acc, filon_panel(Gv.slice(p * N, (p + 1) * N), edges[p], edges[p + 1], y, N));
        val += csc(acc, Math.exp(-tau * y))[0] / Math.PI;
      }
      return val;
    }
    function onevalue(kd, q) {
      if (q === 0) q = 1e-12;
      const R = breakpts(q);
      if (R.double && kern === 1 && kd === 1) return Infinity;
      const ed = [0].concat(R.b);
      R.negl = []; for (let i = 0; i < R.b.length; i++) { const mid = (ed[i] + ed[i + 1]) / 2; R.negl.push(3 * tau * mid * mid / 4 > 40); }
      R.vsad = saddle_loc(q, R);
      return outer_gen(q, R, kd);
    }
    function point(x) { return { F: onevalue(2, x), f: onevalue(1, x) }; }
    function cdf(x) { return onevalue(2, x); }
    function pdf(x) { return onevalue(1, x); }
    return { point, cdf, pdf, tau, info: () => ({ tau, kern }), _d: { tparts, breakpts, inner_gen, tstar, Dsc, iipar: (s) => iipar(s, c, cas) } };
  }

  return { engine, single, _t: { erfz, besselk0s, faddeeva, gl01, makeCoefs, iipar } };
})();
if (typeof module !== 'undefined') module.exports = TRL;
