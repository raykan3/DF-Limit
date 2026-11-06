function [F,info] = tr_limdist(x,cas,c,opts,mode)
%TR_LIMDIST  Limiting distribution and density of the least squares t-ratio.
%
%   [F,info] = TR_LIMDIST(x,cas,c,opts,mode), mode 'cdf' (default) or 'pdf':
%   the limit as n -> Inf of P[t <= x] (or its density) under phi = 1 + c/n,
%   no drift, with a fixed start y0 = g0 sigma sqrt(n) (cases 'DK', 'DU', 'DT';
%   opts.g0, default 0) or a stationary start (cases 'SK', 'SU', 'ST', c < 0;
%   J(0) ~ N(0,-1/(2c)), the limit of y0/(sigma sqrt(n))).  t is the t-ratio for
%   the null value phi0 = 1 + c0/n (opts.c0, default c: t centered at the true
%   phi; c0 = 0: the Dickey-Fuller test of phi = 1 under local alternatives).
%
%   With A, B the limiting functionals of TR_CFLIM and A' = A + c B,
%      t => (A + (c - c0) B)/sqrt(B),
%      P[t <= x] = int_0^Inf E[delta(B - b) 1{R <= a(b)}] db,
%   where R = A' + 1/2 and a(b) = 1/2 + x sqrt(b) + c0 b.  R is a quadratic
%   form of rank one (case DK: R = J(1)^2/2) or two (cases DU, DT) in Gaussian
%   functionals of the process; under the tilt exp(s B) these remain Gaussian
%   (with complex covariances off the real axis) and R has the weights kap_1,
%   kap_2 of TR_CFLIM (form 'B'): E[exp(s B + i u R)] = M_B(s) prod_k (1 -
%   2 i u kap_k)^(-1/2).  Both the distribution function and the density are
%   double integrals (Theorem 3 of the paper):
%
%   (i) case DK (zero or fixed start):
%      E[delta(B - b) 1{R <= a}] = (1/(2 pi i)) int M_B(s) P e^(-s b) ds,
%   P = erf(sqrt(a/(2 kap_1))) for a > 0 and 0 otherwise, the analytic
%   continuation of P[kap_1 Z^2 <= a]; the density uses dP/da.
%   With a zero start (unless opts.oldDK) the product is in closed form: by
%   Girsanov's theorem the law under c is that of Brownian motion W weighted
%   by exp(c (W(1)^2 - 1)/2 - c^2 B/2), and with Levy's formula
%      M_B(s) P = E[e^(s B); R <= a] = e^(-c/2) (th/sinh th)^(1/2) sqrt(a) E(a eta),
%   th = sqrt(c^2 - 2 s), eta = th coth th - c, E(t) = erf(sqrt t)/sqrt t
%   (entire), analytic for s < (c^2 + pi^2)/2.  Unlike M_B it has no
%   singularity at the end s_max of the domain of M_B; for c > 0 the saddle
%   point of the whole integrand (R is close to c B) can lie beyond s_max,
%   where the contour through the saddle point of M_B fails.  The inversion
%   runs through the saddle point of the whole integrand, on the best (by a
%   nested-rule error estimate) of three paths: a parabola opening into a
%   wedge in s, a vertical line in the th-plane, and f_B minus the same
%   integral with erfc in place of erf; the logarithm of the integrand is
%   evaluated free of cancellation in th - c and th coth th - th.  The
%   density uses e^(-c/2) (th/sinh th)^(1/2) e^(-a eta)/sqrt(pi a).  The
%   outer integral covers [min(3e-3, E B e^-8), E B e^6] (B has a heavy left
%   tail for large c), with pieces graded toward the zeros of a(b); for c0 >
%   0 these merge at x = -sqrt(2 c0), where the density has a logarithmic
%   singularity (returned as Inf).
%
%   (ii) cases DU, DT (zero start) and SK, SU, ST (stationary start): with
%   g_s(r) the density of R under the tilt in closed form, for real weights
%   that of kap_1 Z_1^2 + kap_2 Z_2^2, s_k = 1/(2 kap_k),
%      g_s(r) = exp(-(s_1 + s_2) r/2) K0((s_1 - s_2)|r|/2)/(2 pi sqrt(kap_1) sqrt(-kap_2)),
%   and h(r,b) = (1/(2 pi i)) int M_B(s) g_s(r) e^(-s b) ds the joint density of (R, B),
%      f(x) = int_0^Inf sqrt(b) h(a(b), b) db,
%      F(x) = P[R <= 1/2] + int_0^Inf a'(b) G(a(b), b) db,
%      G(r,b) = (1/(2 pi i)) int M_B(s) g_s(r) e^(-s b)/s ds  (Re s > 0),
%   a'(b) = x/(2 sqrt(b)) + c0, by an integration by parts in b (G(r,b) is the
%   density of R on {B > b}); a path crossing the real axis left of 0 adds the
%   residue g_0(r).  P[R <= 1/2] = P(kap_1(0), kap_2(0); 1/2), with
%      P(kap_1, kap_2; a) = 2 int_{l0}^Inf phi(l; s_2) erf(sqrt((2a + l^2)/(2 s_1))) dl,
%   s_1 = 2 kap_1, s_2 = -2 kap_2, phi(l; s) = exp(-l^2/(2s))/sqrt(2 pi s),
%   l0 = sqrt(max(0,-2a)) (Gauss-Legendre, Nl nodes).
%   Cases DU, DT with a zero start (unless opts.oldII): by Girsanov's theorem,
%   under the tilt exp(-th^2/2 int W^2) of Brownian motion X = (W(1), int W[,
%   int r W]) is Gaussian with covariance Sig(th) (tanh, sech in closed form),
%   e^(sB) gains exp(X'QX/2) and R = X'KX/2, and
%      M_B(s) g_s(r) = e^(-c/2) (pi^2 (-A0) cosh th det Sig)^(-1/2)
%                      exp(-A1 r/(2 A0)) K0(-S|r|/(2 A0)),
%   with A2 nu^2 - A1 nu + A0 = 0 the equation of the weights nu = 2 kap_k
%   (A2 = det(Sig^-1 - Q)), S = (A1^2 - 4 A0 A2)^(1/2); the end of the domain
%   of M_B (A2 = 0) is not a singularity, and the analytic continuation ends
%   at cosh th = 0 or S = 0.  The inversions run through the saddle point of
%   the whole integrand on the better of the two paths of case DK, the
%   determinant 1 - c tanh(th)/th evaluated as (th - c + c(1 - tanh th))/th;
%   the crossing point of the path for G is kept one width from the pole at
%   0, with the residue added on its left; the outer integral covers [3e-3,
%   E B e^6] (E B of case DK, an upper bound).
%
%   Cases DU, DT with a fixed start g0 ~= 0: R is a noncentral form, its
%   density has no closed form, and form (i) is used with the noncentral P
%   (a one-dimensional integral; a triple integral overall).
%
%   The inversions in b run along a parabola through the saddle point s* of
%   log M_B(s) - s b that opens to the right around the singularities of M_B
%   (all on the real axis), with log M_B continued along it (TR_CFLIM, form
%   'BC'); for G the crossing point is kept at least one saddle-point width
%   from the pole at 0.  The outer integral over b is a cosine-mapped
%   Gauss-Legendre rule in log b over [E B e^-8, E B e^6], with breakpoints
%   at the zeros of a(b) (the support edge in case DK, logarithmic
%   singularities of g otherwise; the cosine map is applied twice on the
%   pieces that end there).
%
%   Initial conditions (J the Ornstein-Uhlenbeck process dJ = cJ dr + dW,
%   J(0) = g0).  Case DK, fixed start: R = (J(1)^2 - g0^2)/2, under the B-tilt
%   J(1) is Gaussian with mean g0 mq(s) and variance 2 kap_1, and log M_B gains
%   g0^2 aq(s) (closed forms, TR_JBSTART); P = Pbar((2a + g0^2)^(1/2); g0 mq,
%   2 kap_1), Pbar(r; m, v) = P[X^2 <= r^2], X ~ N(m, v).  Case SK: averaging
%   over J(0) ~ N(0, s0^2), s0^2 = -1/(2c), log M_B gains -log(1 - 2 aq s0^2)/2
%   and R = (J(1)^2 - J(0)^2)/2 is a central form of rank two in the tilted
%   pair (J(0), J(1)).  Cases DU, DT, fixed start: given the tilt, R = (X1^2 -
%   X2^2)/2 with X_k independent Gaussian of variances 2 kap_1, -2 kap_2 and
%   means m_k proportional to g0; log M_B gains g0^2 q(s), and q and m_1^2,
%   m_2^2 come from the closed form of the initial-value term (TR_NSSTART,
%   from the determinants of Nabeya and Sorensen) at lambda = 0 and two
%   values of lambda; P is the
%   rule above with the density of X2 and the distribution function of X1^2
%   symmetrized in the means (only m_k^2 enters).  Cases SU, ST: the same
%   pieces at g0 = 1 averaged over the start, log M_B gains -log(1 - 2 s0^2
%   q)/2 and the weights become kap_1' + kap_2' = kap_1 + kap_2 + st2 (mu_1 -
%   mu_2)/2, kap_1' kap_2' = kap_1 kap_2 + st2 (mu_1 kap_2 - mu_2 kap_1)/2,
%   st2 = s0^2/(1 - 2 s0^2 q), mu_k = m_k^2 at g0 = 1.  (Earlier versions
%   took q and m_k^2 from a Richardson extrapolation of the finite-n closed
%   forms, accurate to about 1e-7; the closed forms agree with it to that
%   accuracy and remove the limit.)
%
%   opts: c0; g0 (fixed-start cases); M (64; 32 in case DK with a zero start)
%   outer nodes per piece (pieces at most 2 units of log b, bisected where the
%   error estimate from the Legendre coefficients of the integrand exceeds otol
%   (1e-12), up to maxlev (12) times, and graded toward the step of the
%   integrand where a(b) crosses the conditional mean of R given B = b when its
%   width in log b is below stepw (0.1)); Nv (96)
%   nodes on the contour; oldDK, oldII (false: the old forms in cases DK and
%   DU, DT with a zero start); Nl (96) nodes of the
%   integral over l; region (testing: the sign conditions are also checked on
%   that many intermediate paths, info.nreg of info.ntot points fail).
%   info: time, the breakpoints, E B, nbad, the number of contour nodes at
%   which the weights (with the initial-value terms) left the half planes
%   above, and nunres, the number of path integrals whose nested-rule error
%   estimate was still above 1e-9 of the largest in its piece (and above
%   1e-11 divided by the range of b) after the node count was quadrupled,
%   both per point x; status per point, 0 when clean, otherwise
%   1 (nbad > 0) plus 2 (nunres > 0, or info.err > opts.errtol (1e-10), where
%   info.err is the estimated error of each value: the outer estimates plus,
%   with a zero start, the outer rule applied to the estimates of the inner
%   integrals); info.nrefine counts the bisected pieces.  A nonzero status raises the warning
%   tr_limdist:status.  Non-finite x: F = 0 at -Inf, 1 (cdf) or 0 (pdf) at
%   +Inf, NaN at NaN.
%
%   See also TR_CFLIM, TR_NSSTART, LTRATIO_CDF, LTRATIO_PDF.
if nargin < 4 || isempty(opts), opts = struct; end
if nargin < 5 || isempty(mode), mode = 'cdf'; end
if ~(isnumeric(c) && isscalar(c) && isreal(c) && isfinite(c)), error('tr_limdist:args','tr_limdist: c must be a finite real scalar.'); end
if ~(isnumeric(x) && isreal(x)), error('tr_limdist:args','tr_limdist: x must be real.'); end
if ~all(isfinite(x(:)))
    F = zeros(size(x)); F(isnan(x)) = NaN;
    if ~strcmp(mode,'pdf'), F(x == Inf) = 1; end
    fin = isfinite(x);
    if any(fin(:)), [F(fin),info] = tr_limdist(x(fin),cas,c,opts,mode); else, info = struct('time',[],'status',[]); end
    info.finite = fin;
    return
end
cas = upper(cas); pdfm = strcmp(mode,'pdf');
c0 = getf(opts,'c0',c); M = getf(opts,'M',64); Nv = getf(opts,'Nv',96); Nl = getf(opts,'Nl',96);
g0 = getf(opts,'g0',0);
stA = cas(1) == 'S';                                 % stationary start
if stA
    if ~(c < 0), error('tr_limdist: a stationary start requires c < 0.'); end
    s02 = -1/(2*c); cas = ['D' cas(2)]; g0 = 0;
end
st = stA && strcmp(cas,'DK');                        % case SK (closed form)
nz = st || (g0 ~= 0 && strcmp(cas,'DK'));            % DK: initial-value terms in closed form
nzR = ~strcmp(cas,'DK') && (stA || g0 ~= 0);         % DU, DT: initial-value terms (TR_NSSTART)
if (strcmp(cas,'DK') && ~stA && g0 == 0 && ~getf(opts,'oldDK',false)) || ...
        (any(strcmp(cas,{'DU','DT'})) && ~stA && g0 == 0 && ~getf(opts,'oldII',false))
    % zero start, cases DK (the closed form of E[e^{sB}; R <= a]) and DU, DT
    % (the closed form of E[e^{sB} delta(R - r)]): see the help
    if strcmp(cas,'DK'), [F,info] = dkzero(x,c,c0,opts,pdfm);
    else, [F,info] = duzero(x,c,c0,cas,opts,pdfm); end
    if any(info.status(:))
        iw = find(info.status,1);
        warning('tr_limdist:status',['tr_limdist: %d of %d points with nonzero status (first: x = %.6g, ' ...
            'nunres %d; see info.status); the result may be inaccurate there.'], ...
            nnz(info.status),numel(x),x(iw),info.nunres(iw));
    end
    return
end
% E B from the closed form
smax0 = domainend(cas,c);                            % real domain end of the zero-start M_B
hh = 1e-5; Lp = LB([hh -hh]); EB = (Lp(1)-Lp(2))/(2*hh);
if ~nz && ~nzR, Lp = tr_cflim(0,[hh -hh],cas,c,'ApB'); EB = imag(Lp(1)-Lp(2))/(2*hh); end
% right end of the real domain of M_B: the first zero of D(0, 2s - c^2) for
% s > 0, located on a grid in w = sign(z) |z|^(1/2), z = 2s - c^2 (D is
% oscillatory in w for z > 0), and refined by bisection
smax = domainend(cas,c);
if nz || nzR
    % the initial-value terms move the right end of the real domain of M_B
    % to the left (with a stationary start, to where 1 - 2 s0^2 q(s) = 0,
    % which tends to 0 as c -> 0); the crossing points of all paths must lie
    % to its left, so it is located here by bisection on the domain flag of LB
    hi = smax; if ~isfinite(hi), hi = 1e3; end
    [~,okh] = LB(hi*(1 - 1e-9));
    if ~okh
        lo = 0;
        for it = 1:80
            md = (lo + hi)/2; [~,okm] = LB(md);
            if okm, lo = md; else, hi = md; end
        end
        smax = lo;
    end
end
[zl,wl] = gl01(Nl);
[zm,wm] = gl01(M);
% density form (a double integral): the inner probability P(kap; a) is
% replaced by the density g of R under the tilt, after an integration by
% parts in b; available wherever g has a closed form, i.e. in all cases except
% DU, DT with a fixed nonzero start (noncentral difference of chi-squares)
% In case DK (zero or fixed start) the probability P is itself in closed form
% (erf), so (i) is already a double integral; it is used there, since the
% inverse-square-root edge of the density at a(b) = -g0^2/2 makes (ii) less
% accurate (to about 1e-8 instead of 1e-11).
useg = ~(nzR && ~stA) && ~(strcmp(cas,'DK') && ~st);
if useg && ~pdfm
    [~,k10,k20,mq0] = nodepar(0);                    % untilted weights (tilt 0)
end
nbad = 0; nreg = 0; ntot = 0;
F = zeros(size(x)); tt = F; brk = cell(size(x)); nbad = 0; nreg = 0; ntot = 0;
nbadv = zeros(size(x)); nunv = nbadv; nunres = 0; err_ = zeros(size(x)); nref_ = err_;
for i = 1:numel(x)
    t0 = clock; xx = x(i); nbad = 0; nunres = 0;
    % pieces in log b over [EB e^-8, EB e^6], breakpoints at the zeros of a(b)
    lo = log(EB)-getf(opts,'blo',8); hi = log(EB)+getf(opts,'bhi',6);
    blo = exp(lo); brng_ = exp(hi) - blo;            % (lo, hi are reused inside HB)
    q0 = 1/2; if strcmp(cas,'DK') && ~st, q0 = (1 + g0^2)/2; end   % edge (DK) or kink: a(b) + g0^2/2 = 0
    if c0 == 0
        sr = -q0/xx;
    else
        sr = roots([c0 xx q0]).'; sr = sr(imag(sr) == 0);
    end
    sr = sr(sr > 0); bk = sort(log(sr.^2)); bk = bk(bk > lo & bk < hi);
    br = [lo bk hi]; bb = br(1);
    for jj = 2:numel(br)
        np0 = max(1,ceil((br(jj)-br(jj-1))/getf(opts,'plen',2)));
        bb = [bb, br(jj-1)+(1:np0)*(br(jj)-br(jj-1))/np0]; %#ok<AGROW>
    end
    pa_ = bb(1:end-1); pb_ = bb(2:end);
    % the cosine map on every piece; where the density form has a logarithmic
    % (K0) or inverse-square-root (case DK) singularity at a zero of a(b), it is
    % applied twice on the pieces that end there, so that the distance to the end
    % behaves like z^4 and the rule converges fast; pieces whose error estimate
    % exceeds opts.otol are bisected (ADAPTB)
    pel_ = useg & any(abs(pa_(:) - bk(:).') < 1e-12,2).'; per_ = useg & any(abs(pb_(:) - bk(:).') < 1e-12,2).';
    if isempty(bk), pel_ = false(size(pa_)); per_ = pel_; end
    zs_ = (1-cos(pi*zm(:)))/2; ws_ = wm(:).*pi/2.*sin(pi*zm(:));
    ze_ = (1-cos(pi*zs_))/2; we_ = wm(:).*(pi/2)^2.*sin(pi*zs_).*sin(pi*zm(:));
    hbh_ = @Hb;
    [F(i),err_(i),~,~,nref_(i)] = adaptb(pa_,pb_ - pa_,pel_,per_,ones(size(pa_)),1,zs_,ws_,ze_,we_, ...
        @(bv,ix) hbwrap(hbh_,bv,xx),getf(opts,'otol',1e-12),getf(opts,'maxlev',12));
    if useg && ~pdfm
        % the boundary term of the integration by parts: P[R <= a(b_lo)]
        % without tilt (B < b_lo = E B e^-8 has negligible probability)
        alo = 1/2 + xx*sqrt(blo) + c0*blo;
        F(i) = F(i) + real(Pk(k10,k20,alo,mq0));
    end
    tt(i) = etime(clock,t0); brk{i} = exp(bk); nbadv(i) = nbad; nunv(i) = nunres;
end
stat = 1*(nbadv > 0) + 2*(nunv > 0 | err_ > getf(opts,'errtol',1e-10));
info = struct('time',tt,'breaks',{brk},'EB',EB,'smax',smax,'nbad',nbadv,'nunres',nunv, ...
    'err',err_,'status',stat,'nreg',nreg,'ntot',ntot,'nrefine',nref_);
if any(stat(:))
    iw = find(stat,1);
    warning('tr_limdist:status',['tr_limdist: %d of %d points with nonzero status (first: x = %.6g, ' ...
        'nbad %d, nunres %d; see info.status); the result may be inaccurate there.'], ...
        nnz(stat),numel(x),x(iw),nbadv(iw),nunv(iw));
end

    function H = Hb(b,xx)
        % H(b) = E[delta(B - b) 1{R <= a(b)}] (cdf) or sqrt(b) d/da of it (pdf)
        a = 1/2 + xx*sqrt(b) + c0*b;
        H = zeros(size(b));
        act = true(size(b)); if strcmp(cas,'DK') && ~st, act = a + g0^2/2 > 0; end
        ia = find(act); if isempty(ia), return, end
        bs = b(ia); as = a(ia);
        % saddle points s* of log M_B(s) - s b, all b together: log M_B is convex
        % on its real domain, so the derivative g1(s) = L'(s) - b increases; a
        % safeguarded Newton iteration keeps a bracket [slo, shi] (points outside
        % the domain count as g1 = +Inf), with central differences for L', L''.
        % Only an approximate s* is needed: any real s in the domain gives the
        % exact integral, s* and a1 just place and scale the contour.
        nb = numel(bs); s = zeros(1,nb); slo = -Inf(1,nb); shi = smax*ones(1,nb); sbd = smax*ones(1,nb);
        g2 = ones(1,nb); act = true(1,nb);
        for it = 1:200
            ia2 = find(act); if isempty(ia2), break, end
            sa = s(ia2); ba = bs(ia2);
            [L0,ok0] = LB(sa);
            bad = ~ok0 | sa >= smax;                     % outside the domain
            shi(ia2(bad)) = sa(bad); sbd(ia2(bad)) = min(sbd(ia2(bad)),sa(bad));
            e = min(1e-4*max(1,abs(sa)),0.1*(sbd(ia2) - sa)); okp = false(size(sa)); Lp = L0; Lm = L0;
            for k = 1:30
                tr = find(~okp & ~bad); if isempty(tr), break, end
                [Lpt,okt] = LB(sa(tr) + e(tr)); Lmt = LB(sa(tr) - e(tr)); okt = okt & sa(tr) + e(tr) < smax;
                Lp(tr) = Lpt; Lm(tr) = Lmt; okp(tr) = okt;
                e(tr(~okt)) = e(tr(~okt))/10; sbd(ia2(tr(~okt))) = min(sbd(ia2(tr(~okt))),sa(tr(~okt)) + 10*e(tr(~okt)));
            end
            g1 = (Lp - Lm)./(2*e) - ba; g2a = (Lp - 2*L0 + Lm)./e.^2;
            gd = ~bad & okp;
            lo = gd & g1 < 0; hi = gd & g1 >= 0;
            slo(ia2(lo)) = max(slo(ia2(lo)),sa(lo)); shi(ia2(hi)) = min(shi(ia2(hi)),sa(hi));
            g2(ia2(gd & g2a > 0)) = g2a(gd & g2a > 0);
            sn = sa - g1./g2a; sn(~gd | ~(g2a > 0)) = NaN;
            A0 = slo(ia2); B0 = shi(ia2);
            out = ~(sn > A0 & sn < B0);
            both = out & isfinite(A0) & isfinite(B0);
            sn(both) = (A0(both) + B0(both))/2;
            left = out & ~isfinite(A0); sn(left) = B0(left) - max(1,2*abs(B0(left)));
            right = out & ~isfinite(B0); sn(right) = A0(right) + max(1,abs(A0(right)));
            conv = gd & (abs(g1) < 1e-9*ba | (B0 - A0) < 1e-12*(1 + abs(sa)));
            s(ia2) = sn; s(ia2(conv)) = sa(conv);
            act(ia2(conv)) = false;
        end
        % With a fixed start in cases DU, DT the initial-value term has an
        % essential singularity at the end of the domain (M_B gains exp(g0^2
        % q(s)), q with a pole there), toward which the saddle point moves as b
        % grows; close to it the term is large and the path badly conditioned.
        % Any real point of the domain gives the same integral, so the point is kept
        % at a relative distance of 2% from the end (this affects only b far
        % in the upper tail of B).
        if nzR && ~stA && isfinite(smax0)
            s = min(s, smax0 - 0.02*max(1,abs(smax0)));
        end
        % L'' at the scale of the saddle-point width
        for pass = 1:2
            a1 = 1./sqrt(g2);
            e = min(0.05*a1,0.3*(sbd - s));
            [L0,~] = LB(s); [Lp,okp] = LB(s + e); [Lm,okm] = LB(s - e); okp = okp & s + e < smax;
            g2n = (Lp - 2*L0 + Lm)./e.^2;
            gd = okp & okm & g2n > 0 & isfinite(g2n); g2(gd) = g2n(gd);
        end
        a1 = 1./sqrt(g2);
        if useg && ~pdfm
            % the density form has the factor 1/zeta: keep the crossing point at
            % least one saddle-point width a1 from the pole at 0 (the integral
            % does not depend on the crossing point, up to the residue at 0)
            near = abs(s) < a1;
            sp = a1; sn0 = -a1;
            ok1 = near & (s >= 0) & (sp < 0.5*sbd); s(ok1) = sp(ok1);
            ok2 = near & ~ok1; s(ok2) = sn0(ok2);
        end
        % Talbot-type contour: the parabola s(th) = s* + mu th^2 - 2 i mu th, th >= 0,
        % through the saddle point, opening to the right around the singularities of
        % M_B on the real axis.  With mu = a1/2 the saddle-point Gaussian is
        % exp(-th^2/2), and e^(-s b) adds exp(-mu b th^2); mu >= 4.5 keeps
        % Re (-z)^(1/2) >= 3 far out, where the weights kap then keep the signs
        % of their real starting values.  Trapezoidal rule in tau, th = sth sinh(tau)
        % (the real part of the integrand is even in th).
        mu = max(a1/2,getf(opts,'mumin',4.5)); sth = a1./(2*mu);           % sth: width in th
        thm = sqrt(getf(opts,'tdec',41.5)./(bs.*mu));  % e^(-mu b th^2) < 1e-18
        % trapezoidal rule in tau with Nv nodes; the nodes with even indices
        % form the rule with half as many, which estimates the error.  Where
        % the estimate is not small (paths that pass close to a singularity of
        % M_B, as with a stationary start at small |c|, where the domain end
        % approaches 0) the column is recomputed with 2 Nv and then 4 Nv nodes
        % A column counts as unresolved when the estimate exceeds 1e-9 of the
        % largest column of the piece and, in absolute terms, 1e-11 divided by
        % the range of b: on pieces far in the upper tail of B, where every
        % column is negligible, the relative test alone flags columns whose
        % discrepancy cannot move the result by 1e-11 (with a fixed start at
        % c > 0 the inner integrals there converge from roundoff-sized values).
        [I,Ih] = pathint(1:numel(bs),Nv);
        sc = max(abs(I)) + realmin; tabs = 1e-11/brng_;
        for rf = 1:2
            bad = find(abs(I - Ih) > 1e-9*sc & abs(I - Ih) > tabs);
            if isempty(bad), break, end
            [I(bad),Ih(bad)] = pathint(bad,Nv*2^rf);
        end
        nunres = nunres + nnz(abs(I - Ih) > 1e-9*sc & abs(I - Ih) > tabs);
        if useg && ~pdfm
            lft = s < 0;
            if any(lft), I(lft) = I(lft) + real(gden(as(lft),k10,k20,mq0)); end
            I = (xx./(2*sqrt(bs)) + c0).*I;
        end
        H(ia) = I;

        function [I,Ih] = pathint(q,NN)
            % (1/2 pi i) times the path integral for the columns q, with NN
            % and NN/2 nodes on each half of the path
            bq = bs(q); sq = s(q); muq = mu(q); sthq = sth(q);
            ht = asinh(thm(q)./sthq)/NN;                 % th = sth sinh(tau)
            TAU = (0:NN)'*ht; TH = sthq.*sinh(TAU); W = sthq.*cosh(TAU).*ht; W(1,:) = W(1,:)/2;
            S = sq + muq.*TH.^2 - 2i*muq.*TH;
            W = 2*muq/pi.*(1 + 1i*TH).*W;                % ds/(2 pi i), both halves
            [L,k1,k2,mq] = nodepar(S);
            nbad = nbad + nnz(real(k1) <= 0) + nnz((~strcmp(cas,'DK') || st) & real(k2) >= 0);
            if isfield(opts,'region') && NN == Nv        % (testing) the swept region
                % (the weights with the initial-value terms, as on the path)
                for u = linspace(0,1,opts.region)
                    Su = sq + u*muq.*TH.^2 - 2i*muq.*TH; Su = Su(2:end,:);
                    [~,kq1,kq2] = nodepar(Su);
                    nreg = nreg + nnz(real(kq1) <= 0) + nnz((~strcmp(cas,'DK') || st) & real(kq2) >= 0);
                    ntot = ntot + numel(kq1);
                end
            end
            aa = repmat(as(q),size(S,1),1);
            if useg
                % density form: cdf (1/2 pi i) int M_B g e^(-zeta b)/zeta dzeta,
                % pdf sqrt(b) (1/2 pi i) int M_B g e^(-zeta b) dzeta
                g = gden(aa,k1,k2,mq);
                if pdfm, G = exp(L - S.*bq).*g.*sqrt(bq);
                else,    G = exp(L - S.*bq).*g./S; end
            else
                P = Pk(k1,k2,aa,mq);
                if pdfm, P = P.*repmat(sqrt(bq),size(S,1),1); end
                G = exp(L - S.*bq).*P;
            end
            I = real(sum(G.*W,1));
            W2 = 2*W(1:2:end,:);                          % the rule with step 2 ht
            Ih = real(sum(G(1:2:end,:).*W2,1));
        end
    end

    function [L,k1,k2,mq] = nodepar(S)
        % log M_B and the weights (and means) of R under the tilt at the nodes S
        [L,~,kap] = tr_cflim(0,-1i*S,cas,c,'BC');
        k1 = reshape(kap(1,:),size(S)); k2 = reshape(kap(2,:),size(S));
        mq = []; m2 = [];
        if nzR
            % initial-value terms of DU, DT (fixed start g0) and SU, ST
            % (stationary start) from the closed forms (TR_NSSTART):
            % given the B-tilt, R = (X1^2 - X2^2)/2 with (X1, X2) Gaussian,
            % variances s1 = 2 kap_1, s2 = -2 kap_2 and means (m1, m2) (fixed
            % start; only m1^2, m2^2 enter), or zero means and new variances
            % (stationary start: J(0) mixed into the same pair)
            s1 = 2*k1; s2 = -2*k2;
            if stA
                % stationary start: the fixed-start pieces at gamma0 = 1 (the
                % initial-value term is gamma0^2 Q(lambda, s)), averaged over
                % gamma0 ~ N(0, s02): M_B gains (1 - 2 s02 Q0)^(-1/2), and
                % (1 - 2 lam k1')(1 - 2 lam k2') = (1 - 2 lam k1)(1 - 2 lam k2)
                %   - st2 lam [mu1 (1 - 2 lam k2) - mu2 (1 - 2 lam k1)],
                % st2 = s02/(1 - 2 s02 Q0), mu_k the squared means at gamma0 = 1
                [Q0,mu1,mu2] = qpieces(S,1,s1,s2);
                d1 = 1 - 2*s02*Q0;
                ld = log(d1); ld = real(ld) + 1i*unwrap(imag(ld),[],1);   % continuous along each column
                L = L - ld/2;
                st2 = s02./d1;
                p1 = (k1 + k2) + st2.*(mu1 - mu2)/2;            % k1' + k2'
                p2 = k1.*k2 + st2.*(mu1.*k2 - mu2.*k1)/2;       % k1' k2'
                dsc = sqrt(p1.^2 - 4*p2);
                k1 = (p1 + dsc)/2; k2 = (p1 - dsc)/2;
                sw = real(k2) > real(k1); tq = k1(sw); k1(sw) = k2(sw); k2(sw) = tq;
            else
                [Q0,mq,m2] = qpieces(S,g0,s1,s2);              % m1^2, m2^2
                L = L + Q0;
            end
        end
        if nz
            [aq,mq] = tr_jbstart(-1i*S,c);
            if st
                d1 = 1 - 2*aq*s02;
                ld = log(d1); ld = real(ld) + 1i*unwrap(imag(ld),[],1);   % continuous along each column
                L = L - ld/2;
                t2 = s02./d1; v = 2*k1;                  % tilted Var J(0), Var J(1) | J(0)
                p1 = (mq.^2.*t2 + v - t2)/2; p2 = -t2.*v/4;
                dsc = sqrt(p1.^2 - 4*p2);
                k1 = (p1 + dsc)/2; k2 = (p1 - dsc)/2;
                sw = real(k2) > real(k1); tq = k1(sw); k1(sw) = k2(sw); k2(sw) = tq;
            else
                L = L + g0^2*aq;
            end
        end
        if nzR && ~stA, mq = {mq, m2}; end
    end

    function g = gden(aa,k1,k2,mq)
        % density of R at aa (real) under the tilt with weights k1, k2 (and means)
        if strcmp(cas,'DK') && ~st
            if g0 ~= 0
                % (X^2 - g0^2)/2, X ~ N(g0 mq, 2 k1)
                v = 2*k1; m = g0*mq; r2 = 2*aa + g0^2; r = sqrt(max(r2,0));
                g = (exp(-(r-m).^2./(2*v)) + exp(-(r+m).^2./(2*v)))./(sqrt(2*pi*v).*r);
                g(~(r2 > 0)) = 0;
            else
                g = exp(-aa./(2*k1))./sqrt(2*pi*k1.*aa); g(~(aa > 0)) = 0;
            end
            return
        end
        % k1 Z1^2 + k2 Z2^2: exp(-al a) K0(be |a|) sqrt(-s1 s2)/pi, s2 = 1/(2 k1),
        % s1 = 1/(2 k2), al = (s1+s2)/2, be = (s2-s1)/2 (exponentially scaled K0)
        s2 = 1./(2*k1); s1 = 1./(2*k2); al = (s1+s2)/2; be = (s2-s1)/2;
        z = be.*abs(aa);
        g = exp(-al.*aa - z).*besselk(0,z,1)./(2*pi*sqrt(k1).*sqrt(-k2));
    end

    function [L,ok] = LB(s)
        % log M_B(s) at real s (with the initial-value terms)
        sz0 = size(s);
        [L,ok] = tr_cflim(0*s(:).',-1i*s(:).',cas,c,'B');
        L = reshape(L,sz0); ok = reshape(ok,sz0);
        if nzR
            % the initial-value terms only inside the real domain of the
            % zero-start M_B (outside, the point is rejected anyway)
            iv = ok & isfinite(s) & s < smax0;
            Q0 = zeros(size(s));
            if any(iv(:))
                Qv = lpar(0,s(iv),1);
                Q0(iv) = real(Qv); iv(iv) = isfinite(Qv);
            end
            if stA
                d1 = 1 - 2*s02*Q0; ok = ok & iv & d1 > 0;
                L = L - log(abs(d1))/2;
            else
                ok = ok & iv; L = L + g0^2*Q0;
            end
        end
        if nz
            aq = tr_jbstart(-1i*s,c); ok = ok & isfinite(aq) & abs(imag(aq)) < 1e-8*max(1,abs(aq)); aq = real(aq);
            if st
                d1 = 1 - 2*aq*s02; ok = ok & d1 > 0;
                L = L - log(abs(d1))/2;
            else
                L = L + g0^2*aq;
            end
        end
    end

    function [Q0,m1s,m2s] = qpieces(S,gg,s1,s2)
        % initial-value term of the fixed start gamma0 = gg (cases DU, DT):
        % Q0 = log M_B - log M_B^(0), and the squared tilted means m1^2, m2^2 of
        % (X1, X2) from Q(lambda) - Q0 = (lambda/2) [m1^2/(1 - lambda s1)
        % - m2^2/(1 + lambda s2)] at lambda = 1/(4 s1) and -1/(4 s2)
        la = 0.25./s1; lb = -0.25./s2;
        Q0 = lpar(0,S,gg);
        Qa = lpar(la,S,gg) - Q0;
        Qb = lpar(lb,S,gg) - Q0;
        A11 = la/2./(1 - la.*s1); A12 = -la/2./(1 + la.*s2);
        A21 = lb/2./(1 - lb.*s1); A22 = -lb/2./(1 + lb.*s2);
        dt = A11.*A22 - A12.*A21;
        m1s = (Qa.*A22 - A12.*Qb)./dt; m2s = (A11.*Qb - A21.*Qa)./dt;
    end

    function Q = lpar(lam,s,par)
        % the initial-value part of log E exp(lam A' + s B) (fixed start par =
        % g0, cases DU, DT): par^2 times the closed form of TR_NSSTART (Nabeya
        % and Sorensen's determinants), a quadratic form in the start without
        % a branch
        Q = par^2*tr_nsstart(lam,s,cas,c);
    end

    function P = Pk(k1,k2,a,mq)
        % P(kap_1, kap_2; a) (cdf) or its derivative in a (pdf)
        P = zeros(size(a));
        if iscell(mq)
            P = Pnc(2*k1,-2*k2,a,sqrt(mq{1}),sqrt(mq{2})); return
        end
        if strcmp(cas,'DK') && ~st && g0 ~= 0
            % J(1) ~ N(m, 2 kap_1), {J(1)^2 <= r^2}, r^2 = 2a + g0^2 > 0
            m = g0*mq; r = sqrt(2*a + g0^2); sg = sqrt(4*k1);   % sqrt(2) times the s.d.
            if pdfm                                    % d/da = (d/dr) / r
                P = (exp(-((r - m)./sg).^2) + exp(-((r + m)./sg).^2))./(sqrt(pi)*sg/2)./r/2;
            else
                P = (erfz((r - m)./sg) + erfz((r + m)./sg))/2;
            end
            return
        end
        if strcmp(cas,'DK') && ~st
            if pdfm
                P = exp(-a./(2*k1))./sqrt(2*pi*k1.*a);
            else
                P = erfz(sqrt(a./(2*k1)));
            end
            return
        end
        s1 = 2*k1; s2 = -2*k2;
        % range of l: exp(-l^2 Re(1/(2 s2))) < 1e-18
        lm = sqrt(2*41.4./max(real(1./s2),1e-12));
        pos = a >= 0;
        P = complex(zeros(size(a)));
        if any(pos(:))
            % l = alpha sinh(tau), alpha = sqrt(2a) (or a floor), tau in [0, asinh(lm/alpha)]
            ap = a(pos); q1 = s1(pos); q2 = s2(pos); lmax = lm(pos);
            al = max(sqrt(2*ap),1e-8*lmax); tm = asinh(lmax./al);
            T = zl(:)*tm.'; WT = wl(:)*tm.';                   % Nl x K
            Lg = al.'.*sinh(T); dL = al.'.*cosh(T).*WT;
            ph = exp(-Lg.^2./(2*q2.'))./sqrt(2*pi*q2.');
            u2 = 2*ap.' + Lg.^2;
            if pdfm
                v = ph.*(2/sqrt(pi)).*exp(-u2./(2*q1.'))./sqrt(2*q1.')./sqrt(u2);
            else
                v = ph.*erfz(sqrt(u2./(2*q1.')));
            end
            P(pos) = 2*sum(v.*dL,1).';
        end
        if any(~pos(:))
            % u = sqrt(2a + l^2) in [0, umax], l = sqrt(u^2 + l0^2)
            an = a(~pos); q1 = s1(~pos); q2 = s2(~pos); lmax = lm(~pos);
            l0 = sqrt(-2*an); um = sqrt(max(lmax.^2 - l0.^2,0) + 1e-30);
            U = zl(:)*um.'; WU = wl(:)*um.';
            Lg = sqrt(U.^2 + l0.'.^2);
            ph = exp(-Lg.^2./(2*q2.'))./sqrt(2*pi*q2.');
            if pdfm
                v = ph.*(2/sqrt(pi)).*exp(-U.^2./(2*q1.'))./sqrt(2*q1.')./Lg;
            else
                v = ph.*erfz(U./sqrt(2*q1.')).*U./Lg;
            end
            P(~pos) = 2*sum(v.*WU,1).';
        end
    end

    function P = Pnc(s1,s2,a,m1,m2)
        % noncentral P[X1^2 - X2^2 <= 2a] (cdf) or its derivative in a (pdf),
        % X_k ~ N(m_k, s_k) (analytic continuation in s_k and m_k): the rule
        % of PK with the density of X2 and the distribution function of X1^2
        % symmetrized in the means (only m_k^2 enters)
        P = complex(zeros(size(a)));
        lm = sqrt(2*41.4./max(real(1./s2),1e-12)) + abs(m2);
        F1 = @(u,q1,mm) (erfz((u - mm)./sqrt(2*q1)) + erfz((u + mm)./sqrt(2*q1)))/2;
        f1 = @(u,q1,mm) (exp(-(u - mm).^2./(2*q1)) + exp(-(u + mm).^2./(2*q1)))./sqrt(2*pi*q1);
        f2 = @(l,q2,mm) (exp(-(l - mm).^2./(2*q2)) + exp(-(l + mm).^2./(2*q2)))./(2*sqrt(2*pi*q2));
        pos = a >= 0;
        if any(pos(:))
            ap = a(pos); q1 = s1(pos).'; q2 = s2(pos).'; lmax = lm(pos);
            mm1 = m1(pos).'; mm2 = m2(pos).';
            al = max(sqrt(2*ap),1e-8*lmax); tm = asinh(lmax./al);
            T = zl(:)*tm.'; WT = wl(:)*tm.';
            Lg = al.'.*sinh(T); dL = al.'.*cosh(T).*WT;
            ph = f2(Lg,q2,mm2); u = sqrt(2*ap.' + Lg.^2);
            if pdfm, v = ph.*f1(u,q1,mm1)./u; else, v = ph.*F1(u,q1,mm1); end
            P(pos) = 2*sum(v.*dL,1).';
        end
        if any(~pos(:))
            an = a(~pos); q1 = s1(~pos).'; q2 = s2(~pos).'; lmax = lm(~pos);
            mm1 = m1(~pos).'; mm2 = m2(~pos).';
            l0 = sqrt(-2*an); um = sqrt(max(lmax.^2 - l0.^2,0) + 1e-30);
            U = zl(:)*um.'; WU = wl(:)*um.';
            Lg = sqrt(U.^2 + l0.'.^2);
            ph = f2(Lg,q2,mm2);
            if pdfm, v = ph.*f1(U,q1,mm1)./Lg; else, v = ph.*F1(U,q1,mm1).*U./Lg; end
            P(~pos) = 2*sum(v.*WU,1).';
        end
    end
end

function smax = domainend(cas,c)
% first zero of D(0, 2s - c^2) in s > 0 (D(0, -c^2) = 1 at s = 0)
w = (-abs(c):0.01:4*pi+0.01); z = sign(w).*w.^2;
s = (z + c^2)/2; s = s(s > 0);
D = realD(s,cas,c);
k = find(D <= 0,1);
if isempty(k), smax = Inf; return, end
lo = 0; if k > 1, lo = s(k-1); end
hi = s(k);
for it = 1:100
    m = (lo + hi)/2;
    if realD(m,cas,c) > 0, lo = m; else, hi = m; end
    if hi - lo < 1e-14*max(1,hi), break, end
end
smax = lo;
end
function D = realD(s,cas,c)
% D(0, 2s - c^2) e^(-c) at real s, through tr_cflim (L = -log D/2 at real
% points with D > 0; the imaginary part pi marks D < 0)
[L,~] = tr_cflim(0*s,-1i*s,cas,c,'B');
D = exp(-2*real(L)).*cos(-2*imag(L));
end

function e = erfz(z)
% erf for complex arguments, through the Faddeeva function
e = 1 - exp(-z.^2).*faddeeva(1i*z);
r = imag(z) == 0; e(r) = erf(real(z(r)));
end
function [x,w] = gl01(m)
b = (1:m-1)./sqrt(4*(1:m-1).^2-1); [V,D] = eig(diag(b,1)+diag(b,-1));
[x,ix] = sort(diag(D)); w = 2*V(1,ix).^2; x = (x(:)'+1)/2; w = w(:)'/2;
end
function v = getf(o,f,d), if isfield(o,f), v = o.(f); else, v = d; end, end

function [H,E] = hbwrap(hb,bv,xx)
% the outer integrand of the general route (the nested HB) in the form ADAPTB expects
H = reshape(hb(bv(:).',xx),[],1); E = zeros(size(H));
end

function [F,info] = dkzero(x,c,c0,opts,pdfm)
% Case DK, zero start, by the closed form of E[e^{sB}; R <= a] (see the help):
% F(x) = int_0^Inf H(b) db, H(b) = E[delta(B - b) 1{R <= a(b)}], a(b) = 1/2 +
% x sqrt(b) + c0 b; pdf: f(x) = int_0^Inf sqrt(b) dH/da db.  Outer rule:
% Gauss-Legendre in log b on pieces of at most plen units over [min(3e-3, E B
% e^-8), E B e^6], breakpoints at the zeros of a(b) (the support edge), where
% the cosine map makes the square-root edge smooth; pieces whose error estimate
% exceeds opts.otol are bisected (ADAPTB).
t0 = tic;
M = getf(opts,'M',32); plen = getf(opts,'plen',2);
if abs(c) < 1e-8, EB = 1/2; else, EB = (exp(2*c) - 1 - 2*c)/(4*c^2); end
lo = min(log(3e-3),log(EB) - getf(opts,'blo',8)); hi = log(EB) + getf(opts,'bhi',6);
[z,w] = gl01(M); z = z(:); w = w(:);
zc = (1 - cos(pi*z))/2; wc = w.*pi.*sin(pi*z)/2;
SG = stepgrid(c,'DK',lo,hi);                         % for the sharp step of the outer integrand
nx = numel(x); brk = cell(size(x));
PA = []; PL = []; PX = []; PEL = false(1,0); PER = false(1,0);   % the pieces: start, length, point, zero at the ends
% x = -2 sqrt(c0/2) (c0 > 0): the zeros of a(b) coincide; there the density is
% infinite (a logarithmic singularity) and the distribution function is
% continuous
amin0 = false(1,nx);
if c0 > 0, amin0 = abs(x(:)' + sqrt(2*c0)) <= 8*eps*max(1,sqrt(2*c0)); end
for k = 1:nx
    xk = x(k);
    if c0 == 0, r = -1/(2*xk); else, r = roots([c0 xk 1/2]); end
    r = r(abs(imag(r)) < 1e-14 & real(r) > 0); lr = sort(2*log(real(r(:)))).';
    lr = lr(lr > lo & lr < hi); brk{k} = exp(lr);
    % near x = -2 sqrt(c0/2) (c0 > 0) the two zeros of a(b) merge at the minimum
    % of a(b)/sqrt(b): the panels are graded geometrically toward the zeros (or,
    % when a(b) > 0, toward the near-zero minimum) down to the scale sg of the gap
    sp = []; sg = Inf;
    if c0 > 0 && xk < 0
        r0 = -xk/(2*c0); amin = 1/2 - xk^2/(4*c0); l0 = 2*log(r0);
        if numel(lr) == 2 && lr(2) - lr(1) < 1e-10, lr = mean(lr); brk{k} = exp(lr);   % a double zero
        elseif numel(lr) == 2, sg = lr(2) - lr(1);
        elseif l0 > lo && l0 < hi
            if amin <= 1e-13, lr = l0; brk{k} = exp(lr);    % a double zero: a breakpoint, no grading
            else, sg = 2*sqrt(amin/c0)/r0; sp = l0; end
        end
    end
    pts = [lo hi lr]; gr = [lr sp];
    if sg < plen
        for g = gr
            d = sg*2.^(0:ceil(log2(plen/sg)));
            pts = [pts g-d g+d]; %#ok<AGROW>
        end
    end
    % the step of the outer integrand where a(b) crosses the conditional mean of
    % R given B = b (sharp at large x for c > c0): pieces graded toward it from
    % its width
    [ls,ws] = steproots(SG,xk,c0,getf(opts,'stepw',0.1));
    for g = 1:numel(ls)
        d = 2*ws(g)*2.^(0:ceil(log2(plen/(2*ws(g)))));
        pts = [pts ls(g) ls(g)-d ls(g)+d]; %#ok<AGROW>
    end
    pts = pts(pts >= lo & pts <= hi);
    for g = lr, pts(abs(pts - g) < 1e-9*min(max(sg,1e-3),plen) & pts ~= g) = []; end   % no slivers at a zero
    pts = unique(pts);
    if numel(lr) == 2, pts = pts(pts <= lr(1) | pts >= lr(2)); end
    for p = 1:numel(pts)-1
        if numel(lr) == 2 && pts(p) == lr(1), continue, end      % a(b) < 0 between the zeros
        np = max(1,ceil((pts(p+1) - pts(p))/plen)); ed = linspace(pts(p),pts(p+1),np+1);
        for q = 1:np
            PA(end+1) = ed(q); PL(end+1) = ed(q+1) - ed(q); PX(end+1) = k; %#ok<AGROW>
            PEL(end+1) = q == 1 && any(pts(p) == lr); PER(end+1) = q == np && any(pts(p+1) == lr); %#ok<AGROW>
        end
    end
end
evalH = @(B,IX) dkeval(B,IX,x,c,c0,opts,pdfm);
[F,err,nun,npairs,nref] = adaptb(PA,PL,PEL,PER,PX,nx,z,w,zc,wc,evalH,getf(opts,'otol',1e-12),getf(opts,'maxlev',12));
F = reshape(F,size(x)); err = reshape(err,size(x)); nun = reshape(nun,size(x)); nref = reshape(nref,size(x));
if pdfm && any(amin0), F(amin0) = Inf; end
% error estimate of F (or f): the outer error estimates of ADAPTB plus the outer
% rule applied to the estimates of the inner integrals; unresolved where it
% exceeds 1e-10 (the count of inner integrals with an estimate above 1e-9 of
% the largest |H| is info.nunres)
stat = 2*(err > getf(opts,'errtol',1e-10));
info = struct('time',toc(t0),'breaks',{brk},'EB',EB,'smax',dksmax(c), ...
    'nbad',zeros(size(x)),'nunres',nun,'err',err,'status',stat,'npairs',npairs,'nrefine',nref);
end

function [H,E] = dkeval(B,IX,x,c,c0,opts,pdfm)
% the outer integrand of DKZERO at the pairs (b, x(IX)): H(b) (cdf) or sqrt(b) dH/da (pdf)
X = x(IX); X = X(:); B = B(:);
A = 1/2 + X.*sqrt(B) + c0*B;
H = zeros(size(B)); E = H; ch = getf(opts,'chunk',4000);
for i = 1:ch:numel(B)
    j = (i:min(i+ch-1,numel(B)))';
    [H(j),E(j)] = dkpairs(B(j),A(j),A(j) - c*B(j),c,opts,pdfm);
end
if pdfm, H = H.*sqrt(B); E = E.*sqrt(B); end
end

function [H,Ek] = dkpairs(b,a,v,c,o,pdfm)
% H = E[delta(B - b) 1{R <= a}] (pdf: dH/da) for pairs (b, a), v = a - c b.
% log of the integrand: kind 0 (cdf) or 3 (pdf); the split path uses kinds 1
% (log M_B) and 2 (log of M_B erfc(sqrt(a eta))).  Three paths, each with a
% nested-rule error estimate (plus rounding from the size of the integrand),
% refined by doubling; a value is accepted when two paths agree (AGREEPATHS).
NN = getf(o,'Nv',96); kap = 2; tl = 45; tol = 1e-13;
b = b(:)'; a = a(:)'; v = v(:)';
H = zeros(size(a)); Ek = zeros(size(a)); ok = a > 0;
if ~any(ok), H = H(:); Ek = Ek(:); return, end
aa = a(ok); vv = v(ok); bb = b(ok); n = numel(aa);
k0 = 0; if pdfm, k0 = 3; end
s1 = (c^2 + pi^2)/2; smx = dksmax(c);
lk = @(S,q,k) dklam(S,repmat(bb(q),size(S,1),1),c,repmat(aa(q),size(S,1),1),repmat(vv(q),size(S,1),1),k);
ia = 1:n;
[s,a1,g2,L0] = rsad(@(S,q) lk(S,q,k0),bb,s1);
Lh = L0 - log(2*pi*max(g2,1e-6*bb.^2))/2;                       % log of the saddle-point approximation
if pdfm, lw = 1.5*log(bb); else, lw = log(bb); end
act = find(~(Lh + lw < -35));                   % the others are negligible (in d log b)
if ~pdfm && ~isempty(act)
    % the third route, H = f_B - H_c, H_c = E[delta(B - b) 1{R > a}]: saddle points
    % of log M_B - s b and of log M_B erfc - s b, left of s_max
    [sB,aB] = rsad(@(S,q) lk(S,act(q),1),bb(act),smx);
    [sC,aC] = rsad(@(S,q) lk(S,act(q),2),bb(act),smx);
end
np = 3; if pdfm, np = 2; end
[Hk,Ekk] = agreepaths(@evalp,np,act,n,NN,tol);
H(ok) = Hk; Ek(ok) = Ekk; H = H(:); Ek = Ek(:);

    function [I,E] = evalp(pth,q,NNq)
        % the value and the nested-rule estimate of path pth at the pairs q
        if pth == 1
            [I,E] = pthW(q,s(q),a1(q)/2,NNq,k0);
        elseif pth == 2
            [I,E] = pthT(q,s(q),a1(q),NNq,k0);
        else
            [~,iq] = ismember(q,act);
            [IB,EB] = pthW(q,sB(iq),aB(iq)/2,NNq,1); [IC,EC] = pthT(q,sC(iq),aC(iq),NNq,2);
            I = IB - IC; E = EB + EC;
        end
    end

    function [I,E] = pthW(q,sq,mu,NN,k)
        % parabola s^ + mu t^2 - 2 i mu t near the saddle, bending into the
        % wedge Re s - s^ = kap |Im s|; trapezoidal rule in u, t = sinh(u)
        bq = bb(q);
        tmax = max(10,(tl + 5)./(2*kap*bq.*mu));
        ht = asinh(tmax)/NN; TAU = (0:NN)'*ht; TH = sinh(TAU); Wt = cosh(TAU).*ht; Wt(1,:) = Wt(1,:)/2;
        rq = sqrt(1 + (TH/(2*kap)).^2);
        S = sq + mu.*(TH.^2./rq) - 2i*mu.*TH;
        dS = mu.*(2*TH./rq - TH.^3./(4*kap^2*rq.^3)) - 2i*mu;
        [I,E] = fin(exp(lk(S,q,k)).*(1i*dS/pi).*Wt);
    end
    function [I,E] = pthT(q,sq,a1q,NN,k)
        % vertical line th = thA + i y in the th-plane (Gaussian decay
        % e^(-y^2 b/2)); thA >= th^, and large enough that the oscillation of
        % a coth(th) cannot build up against the Gaussian
        bq = bb(q);
        qh = c^2 - 2*sq; th0 = zeros(size(qh)); th0(qh > 0) = sqrt(qh(qh > 0));
        tA = max(max(th0,asinh(aa(q)./sqrt(6*bq))/2),1); sA = (c^2 - tA.^2)/2;
        hq = 1e-4*a1q; L3 = real(lk([sA-hq; sA; sA+hq],q,k));
        Ls = (L3(3,:) - L3(1,:))./(2*hq); Lss = (L3(3,:) - 2*L3(2,:) + L3(1,:))./hq.^2;
        sg = 1./sqrt(max(Lss.*tA.^2 - Ls,bq)); bd = ~isfinite(sg); sg(bd) = 1./sqrt(bq(bd));
        ym = max(12*sg,sqrt(2*(tl + 10)./bq));
        um = asinh(ym./sg); U = (0:NN)'*(um/NN); Y = sg.*sinh(U);
        Wt = sg.*cosh(U).*(um/NN); Wt(1,:) = Wt(1,:)/2;
        T = tA + 1i*Y; S = (c^2 - T.^2)/2;
        [I,E] = fin(exp(lk(S,q,k)).*T.*Wt/pi);
    end
end
function [I,E] = fin(G)
nf = any(~isfinite(G),1); G(~isfinite(G)) = 0;
I = real(sum(G,1)); Ih = real(sum(2*G(1:2:end,:),1));       % the rule with step 2 h
E = abs(I - Ih) + 1e-14*sum(abs(G),1) + 1e3*abs(G(end,:)); E(nf) = Inf;   % (last term: the truncation)
end

function [Hk,Ekk,pk] = agreepaths(evalp,np,act,n,NN,tol)
% The inner inversion by np paths, for the pairs act of 1:n: evalp(pth,q,NN)
% gives the trapezoidal values and the nested-rule estimates of path pth with
% NN nodes at the pairs q.  Each path is refined on its own, the node count
% doubled up to three times until its estimate is below tol max(1,|I|); a pair
% is accepted as soon as two of its paths agree to 2 tol max(1,|I|), the value
% with the smaller estimate being kept and the discrepancy entering the error.
% The nested rule alone can be fooled: an oscillating tail that neither the
% rule nor its half resolves aliases into the same wrong value on both (case
% DK, c = 5, c0 = -5, b = 0.086, a = 35: the parabola with 96 and 192 nodes
% agrees with itself to 1e-15 and is off by 1e-7, which the vertical line
% exposes).  Where no two paths agree, the value with the smallest estimate is
% kept if every other path is consistent with it, the discrepancy within ten
% times the sum of the two estimates (a path that converges slowly, or not at
% all, is no evidence against a resolved one); otherwise the inconsistent paths and
% the best one are recomputed with the node count doubled, up to three more
% times, until two agree or all are consistent, and if they still disagree
% the best value is kept with the smallest inconsistent discrepancy as its
% error, which the outer rule then reports.  Pairs not in act are
% negligible: zero with a zero error.  pk is the path kept.
Hk = zeros(1,n); Ekk = zeros(1,n); pk = zeros(1,n);
IP = nan(np,n); EP = inf(np,n); NP = NN*ones(np,n);
done = true(1,n); done(act) = false;
agree = @(x,y) isfinite(x) & isfinite(y) & abs(x - y) <= 2*tol*max(1,max(abs(x),abs(y)));
for pth = 1:np
    q = find(~done); qr = q; NNq = NN;
    for rf = 0:3
        if isempty(qr), break, end
        [I,E] = evalp(pth,qr,NNq);
        better = E < EP(pth,qr); IP(pth,qr(better)) = I(better); EP(pth,qr(better)) = E(better); NP(pth,qr(better)) = NNq; %#ok<*AGROW>
        qr = qr(E > tol*max(1,abs(I))); NNq = 2*NNq;
    end
    for pp = 1:pth-1
        ag = q(~done(q) & agree(IP(pth,q),IP(pp,q)));
        if isempty(ag), continue, end
        [Hk(ag),Ekk(ag),pk(ag)] = accept(IP([pth pp],ag),EP([pth pp],ag),[pth pp]); done(ag) = true;
    end
end
for round = 0:3
    q = find(~done); if isempty(q), break, end
    % the best path of each pair, and the paths inconsistent with it
    [Eb,pb] = min(EP(:,q),[],1); ib = sub2ind(size(IP),pb,q); Ib = IP(ib);
    bad = false(np,numel(q));
    for pth = 1:np
        bad(pth,:) = pth ~= pb & isfinite(IP(pth,q)) & abs(Ib - IP(pth,q)) > 10*(Eb + EP(pth,q));
    end
    cons = ~any(bad,1) & isfinite(Eb);
    Hk(q(cons)) = Ib(cons); Ekk(q(cons)) = Eb(cons); pk(q(cons)) = pb(cons); done(q(cons)) = true;
    nf = ~isfinite(Eb); Hk(q(nf)) = 0; Ekk(q(nf)) = Inf; pk(q(nf)) = pb(nf); done(q(nf)) = true;
    q = q(~cons & ~nf); bad = bad(:,~cons & ~nf); pb = pb(~cons & ~nf);
    if isempty(q), break, end
    if round == 3
        % still disagreeing: the best value with the smallest inconsistent discrepancy as its error
        for k = 1:numel(q)
            d = abs(IP(pb(k),q(k)) - IP(bad(:,k),q(k)));
            Hk(q(k)) = IP(pb(k),q(k)); Ekk(q(k)) = max(EP(pb(k),q(k)),min(d)); pk(q(k)) = pb(k);
        end
        done(q) = true; break
    end
    for pth = 1:np
        qq = q(bad(pth,:) | pb == pth); if isempty(qq), continue, end
        NP(pth,qq) = 2*NP(pth,qq);
        for NNv = unique(NP(pth,qq))
            qn = qq(NP(pth,qq) == NNv); [I,E] = evalp(pth,qn,NNv);   % the finer rule replaces the value
            IP(pth,qn) = I; EP(pth,qn) = E;
        end
    end
    for p1 = 1:np
        for p2 = 1:p1-1
            m = ~done(q) & agree(IP(p1,q),IP(p2,q)); ag = q(m);
            if isempty(ag), continue, end
            [Hk(ag),Ekk(ag),pk(ag)] = accept(IP([p1 p2],ag),EP([p1 p2],ag),[p1 p2]); done(ag) = true;
        end
    end
end
end
function [H,E,p] = accept(V,Es,paths)
% of two agreeing paths (rows of V, Es) keep the value with the smaller estimate;
% the discrepancy enters the error
[Em,ib] = min(Es,[],1); m = size(V,2);
H = V(sub2ind(size(V),ib,1:m)); E = max(Em,abs(V(1,:) - V(2,:))); p = paths(ib);
end

function L = dklam(s,b,c,a,v,kind)
% log of the integrand times e^(-s b), cancellation-free in del = th - c and
% ep = th coth th - th = 2 th e^(-2 th)/(1 - e^(-2 th)) (eta = del + ep):
%   kind 0: e^(-c/2) (th/sinh th)^(1/2) sqrt(a) E(a eta)       (cdf)
%   kind 1: M_B = e^(-c/2) (th/sinh th)^(1/2) eta^(-1/2)
%   kind 2: M_B erfc(sqrt(a eta))
%   kind 3: e^(-c/2) (th/sinh th)^(1/2) e^(-a eta)/sqrt(pi a)  (pdf, d/da)
% with -s b = del (th + c) b/2 and -a eta - s b = -v eta - c b ep + del^2 b/2
q = c^2 - 2*s; th = sqrt(q);
if c > 0, del = -2*s./(th + c); else, del = th - c; end
e2 = exp(-2*th); sm = abs(th) < 1;
ep = zeros(size(th)); ls = ep;
ep(~sm) = 2*th(~sm).*e2(~sm)./(1 - e2(~sm));
ls(~sm) = (log(th(~sm)) - th(~sm) - log((1 - e2(~sm))/2))/2;      % log (th/sinh th)^(1/2)
if any(sm(:))
    qs = q(sm); r = ones(size(qs)); t = r; ch = r;
    for k = 1:14, t = t.*qs/((2*k)*(2*k+1)); r = r + t; end        % sinh(th)/th
    t = ones(size(qs)); for k = 1:14, t = t.*qs/((2*k-1)*(2*k)); ch = ch + t; end   % cosh(th)
    ls(sm) = -log(r)/2; ep(sm) = ch./r - th(sm);
end
eta = del + ep;
if kind == 3
    L = -c/2 + ls - log(pi*a)/2 - v.*eta - c*b.*ep + del.^2.*b/2; return
end
if kind > 0
    % sqrt(eta) continuous along each column (the nodes of a path, in order)
    r = sqrt(eta);
    if size(r,1) > 1
        fl = real(r(2:end,:).*conj(r(1:end-1,:))) < 0;
        r = r.*cumprod([ones(1,size(r,2)); 1 - 2*fl],1);
    end
    if kind == 1
        L = -c/2 + ls + del.*(th + c).*b/2 - log(r);
    else
        L = -c/2 + ls - log(r) - v.*eta - c*b.*ep + del.^2.*b/2 + log(faddeeva(1i*sqrt(a).*r));
    end
    return
end
t = a.*eta; zt = sqrt(t); L = zeros(size(s));
r1 = real(t) > 2;
if any(r1(:))
    % log M_B - s b + log erf(z), erf(z) = 1 - e^(-z^2) w(i z)
    L(r1) = -c/2 + ls(r1) + del(r1).*(th(r1) + c).*b(r1)/2 ...
        + log(1 - exp(-t(r1)).*faddeeva(1i*zt(r1))) - log(zt(r1)) + log(a(r1))/2;
end
r2 = ~r1;
if any(r2(:))
    % -a eta - s b + log g(t), g(t) = e^t E(t) = (e^t - w(i z))/z (series for small z)
    tt = t(r2); zz = zt(r2); lg = zeros(size(tt)); sz = abs(zz) < 0.2;
    if any(sz)
        ts = tt(sz); ser = zeros(size(ts)); term = ones(size(ts));
        for k = 0:20, ser = ser + term/(2*k+1); term = -term.*ts/(k+1); end
        lg(sz) = ts + log(2/sqrt(pi)*ser);
    end
    lg(~sz) = log((exp(tt(~sz)) - faddeeva(1i*zz(~sz)))./zz(~sz));
    L(r2) = -c/2 + ls(r2) + log(a(r2))/2 - v(r2).*eta(r2) - c*b(r2).*ep(r2) + del(r2).^2.*b(r2)/2 + lg;
end
end

function sm = dksmax(c)
% end of the real domain of M_B (case DK): eta(s) = 0, eta decreasing in s;
% bisection on the cancellation-free eta
if c == 1, sm = 1/2; return, end
lo = min(0,c^2/2) - 1; hi = (c^2 + pi^2)/2;
while real(dketa(lo,c)) <= 0, lo = 2*lo - 1; end
for k = 1:2000
    m = (lo + hi)/2; if m == lo || m == hi, break, end
    if real(dketa(m,c)) > 0, lo = m; else, hi = m; end
end
sm = lo;
end
function e = dketa(s,c)
th = sqrt(c^2 - 2*s);
if c > 0, del = -2*s./(th + c); else, del = th - c; end
e2 = exp(-2*th); ep = 2*th.*e2./(1 - e2); sm = abs(th) < 1e-4; ep(sm) = 1 - th(sm) + th(sm).^2/3;
e = del + ep;
end

function [F,info] = duzero(x,c,c0,cas,opts,pdfm)
% Cases DU, DT, zero start: form (ii) with the closed form of
% J(s; r) = E[e^(sB) delta(R - r)] (IIPAIRS), F(x) = P[R <= a(b_lo)] + int a'(b)
% G(a(b),b) db, f(x) = int sqrt(b) h(a(b),b) db; outer rule as for case DK over
% [3e-3, E B e^6] (P[B < 3e-3] < e^-40; E B of case DK, an upper bound), the
% cosine map applied twice on the pieces that end at a zero of a(b)
% (logarithmic singularity of g).
t0 = tic;
M = getf(opts,'M',64); plen = getf(opts,'plen',2);   % (32 leaves 1e-6 at c = -20, c0 = -40)
% upper end from E int J^2 >= E B (the demeaned or detrended B is smaller)
if abs(c) < 1e-8, EB = 1/2; else, EB = (exp(2*c) - 1 - 2*c)/(4*c^2); end
lo = log(getf(opts,'bmin',3e-3)); hi = log(EB) + getf(opts,'bhi',6);   % P[B < 3e-3] < e^-40
[z,w] = gl01(M); z = z(:); w = w(:);
t1 = (1 - cos(pi*z))/2; zc = (1 - cos(pi*t1))/2; wc = w.*(pi/2)^2.*sin(pi*t1).*sin(pi*z);   % cosine map twice (K0 edge)
SG = stepgrid(c,cas,lo,hi);                          % for the sharp step of the outer integrand
nx = numel(x); brk = cell(size(x));
PA = []; PL = []; PX = []; PEL = false(1,0); PER = false(1,0);   % the pieces: start, length, point, zero at the ends
% x = -2 sqrt(c0/2) (c0 > 0): the zeros of a(b) coincide; there the density is
% infinite (a logarithmic singularity) and the distribution function is
% continuous
amin0 = false(1,nx);
if c0 > 0, amin0 = abs(x(:)' + sqrt(2*c0)) <= 8*eps*max(1,sqrt(2*c0)); end
for k = 1:nx
    xk = x(k);
    if c0 == 0, r = -1/(2*xk); else, r = roots([c0 xk 1/2]); end
    r = r(abs(imag(r)) < 1e-14 & real(r) > 0); lr = sort(2*log(real(r(:)))).';
    lr = lr(lr > lo & lr < hi); brk{k} = exp(lr);
    % near x = -2 sqrt(c0/2) (c0 > 0) the two zeros of a(b) merge at the minimum
    % of a(b)/sqrt(b): the panels are graded geometrically toward the zeros (or,
    % when a(b) > 0, toward the near-zero minimum) down to the scale sg of the gap
    sp = []; sg = Inf;
    if c0 > 0 && xk < 0
        r0 = -xk/(2*c0); amin = 1/2 - xk^2/(4*c0); l0 = 2*log(r0);
        if numel(lr) == 2 && lr(2) - lr(1) < 1e-10, lr = mean(lr); brk{k} = exp(lr);   % a double zero
        elseif numel(lr) == 2, sg = lr(2) - lr(1);
        elseif l0 > lo && l0 < hi
            if amin <= 1e-13, lr = l0; brk{k} = exp(lr);    % a double zero: a breakpoint, no grading
            else, sg = 2*sqrt(amin/c0)/r0; sp = l0; end
        end
    end
    pts = [lo hi lr]; gr = [lr sp];
    if sg < plen
        for g = gr
            d = sg*2.^(0:ceil(log2(plen/sg)));
            pts = [pts g-d g+d]; %#ok<AGROW>
        end
    end
    % the step of the outer integrand where a(b) crosses the conditional mean of
    % R given B = b (sharp at large x for c > c0): pieces graded toward it from
    % its width
    [ls,ws] = steproots(SG,xk,c0,getf(opts,'stepw',0.1));
    for g = 1:numel(ls)
        d = 2*ws(g)*2.^(0:ceil(log2(plen/(2*ws(g)))));
        pts = [pts ls(g) ls(g)-d ls(g)+d]; %#ok<AGROW>
    end
    pts = pts(pts >= lo & pts <= hi);
    for g = lr, pts(abs(pts - g) < 1e-9*min(max(sg,1e-3),plen) & pts ~= g) = []; end   % no slivers at a zero
    pts = unique(pts);
    for p = 1:numel(pts)-1
        np = max(1,ceil((pts(p+1) - pts(p))/plen)); ed = linspace(pts(p),pts(p+1),np+1);
        for q = 1:np
            PA(end+1) = ed(q); PL(end+1) = ed(q+1) - ed(q); PX(end+1) = k; %#ok<AGROW>
            PEL(end+1) = q == 1 && any(pts(p) == lr); PER(end+1) = q == np && any(pts(p+1) == lr); %#ok<AGROW>
        end
    end
end
evalH = @(B,IX) dueval(B,IX,x,c,c0,cas,opts,pdfm);
[F,err,nun,npairs,nref] = adaptb(PA,PL,PEL,PER,PX,nx,z,w,zc,wc,evalH,getf(opts,'otol',1e-12),getf(opts,'maxlev',12));
F = reshape(F,size(x)); err = reshape(err,size(x)); nun = reshape(nun,size(x)); nref = reshape(nref,size(x));
if ~pdfm
    % boundary term of the integration by parts: P[R <= a(b_lo)] without tilt, with
    % the weights from the closed forms at s = 0 (those of TR_CFLIM lose kap_2 to
    % cancellation for c >> 0): the roots nu = 2 kap of A2 nu^2 - A1 nu + A0
    P0 = iipar(0,c,cas); S0 = sqrt(P0.A1^2 - 4*P0.A0*P0.A2);
    k10 = real((P0.A1 + S0)/(4*P0.A2)); k20 = real(P0.A0/(P0.A1 + S0)); blo = exp(lo);
    for k = 1:nx, F(k) = F(k) + Pk0(k10,k20,1/2 + x(k)*sqrt(blo) + c0*blo); end
end
% error estimate of F (or f): the outer error estimates of ADAPTB plus the outer
% rule applied to the estimates of the inner integrals; unresolved where it
% exceeds 1e-10 (the count of inner integrals with an estimate above 1e-9 of
% the largest |H| is info.nunres)
stat = 2*(err > getf(opts,'errtol',1e-10));
info = struct('time',toc(t0),'breaks',{brk},'EB',EB, ...
    'nbad',zeros(size(x)),'nunres',nun,'err',err,'status',stat,'npairs',npairs,'nrefine',nref);
end

function G = stepgrid(c,cas,lo,hi)
% The conditional mean and standard deviation of R given B = b (CONDMOM) on a
% coarse grid in log b over [lo, hi], for STEPROOTS; cached for repeated calls
% with the same c (quantiles, grids of x).
persistent CACHE
if isempty(CACHE) || CACHE.Count > 200, CACHE = containers.Map('KeyType','char','ValueType','any'); end
key = sprintf('%s|%.17g|%.17g|%.17g',cas,c,lo,hi);
if isKey(CACHE,key), G = CACHE(key); return, end
G = struct('c',c,'cas',cas);
if strcmp(cas,'DK'), G.smx = dksmax(c);
else, G.smx = domainend(cas,c); if ~isfinite(G.smx), G.smx = (c^2 + pi^2/4)/2; end, end
G.lb = linspace(lo,hi,61);
[G.m,G.sd] = condmom(G,G.lb);
CACHE(key) = G;
end

function [m,sd] = condmom(G,lb)
% At b = exp(lb): the saddle point s* of log M_B(s) - s b, and the mean m and the
% conditional standard deviation sd of R given B = b from the tilt at s* (double
% saddle point: Var[R | B] = K_uu - K_us^2/K_ss, with K_uu = 2 (kap_1^2 + kap_2^2),
% K_us = dm/ds and K_ss the curvature at s*, by central differences).
c = G.c; cas = G.cas; b = exp(lb(:).');
if strcmp(cas,'DK')
    Lf = @(S,q) real(dklam(S,repmat(b(q),size(S,1),1),c,ones(size(S)),ones(size(S)),1));
    mf = @(S) dkmom(S,c);
else
    Lf = @(S,q) mblog(S,b(q),c,cas);
    mf = @(S) iimom(S,c,cas);
end
[s,a1] = rsad(Lf,b,G.smx);
h = 1e-3*a1; h = min(h,(G.smx - s)/4);
[m,vt] = mf(s); mp = mf(s + h); mm = mf(s - h);
dm = (mp - mm)./(2*h);
L3 = Lf([s - h; s; s + h],1:numel(b)); kss = (L3(1,:) - 2*L3(2,:) + L3(3,:))./h.^2;
vc = vt - dm.^2./kss; vc = max(vc,1e-8*vt);
sd = sqrt(vc);
end

function L = mblog(S,bq,c,cas)
% log M_B(s) - s b (cases DU, DT, zero start)
P = iipar(S,c,cas);
L = real(P.logMB + P.del.*(P.th + c).*repmat(bq,size(S,1),1)/2);
end

function [m,v] = dkmom(s,c)
% mean and variance of R = kap_1 Z^2 under the tilt (case DK): kap_1 = 1/(2 eta)
k1 = 1./(2*real(dketa(s,c))); m = k1; v = 2*k1.^2;
end

function [m,v] = iimom(s,c,cas)
% mean and variance of R = kap_1 Z_1^2 + kap_2 Z_2^2 under the tilt (cases DU, DT):
% A2 nu^2 - A1 nu + A0 = 0, nu = 2 kap
P = iipar(s,c,cas); A2 = real(P.A2); A1 = real(P.A1); A0 = real(P.A0);
m = A1./(2*A2); v = ((A1./A2).^2 - 2*A0./A2)/2;
end

function [ls,ws] = steproots(G,x,c0,wmax)
% the points where a(b) = 1/2 + x sqrt(b) + c0 b crosses the conditional mean of R
% (sign changes on the grid of STEPGRID, refined by the Illinois variant of regula
% falsi), and the widths in log b, sd/|d(a - m)/d log b|; only those below wmax
phf = @(l,m) 1/2 + x*exp(l/2) + c0*exp(l) - m;
ph = phf(G.lb,G.m);
ok = isfinite(ph) & isfinite(G.sd);
k = find(ok(1:end-1) & ok(2:end) & sign(ph(1:end-1)) ~= sign(ph(2:end)));
ls = zeros(1,0); ws = ls;
for j = k
    la = G.lb(j); lc = G.lb(j+1); fa = ph(j); fc = ph(j+1); sa = G.sd(j); sdc = G.sd(j+1); side = 0;
    for it = 1:6
        l = la - fa*(lc - la)/(fc - fa);
        [mn,sn] = condmom(G,l); fn = phf(l,mn);
        if ~isfinite(fn), break, end
        if sign(fn) == sign(fa), la = l; fa = fn; sa = sn; if side == -1, fc = fc/2; end, side = -1;
        else, lc = l; fc = fn; sdc = sn; if side == 1, fa = fa/2; end, side = 1; end
    end
    t = fa/(fa - fc); l = la + t*(lc - la); sd = sa + t*(sdc - sa);
    [m1] = condmom(G,l - 1e-3); [m2] = condmom(G,l + 1e-3);
    sl = (phf(l + 1e-3,m2) - phf(l - 1e-3,m1))/2e-3;
    wd = sd/abs(sl);
    if wd < wmax, ls(end+1) = l; ws(end+1) = max(wd,1e-6); end %#ok<AGROW>
end
end

function [H,E] = dueval(B,IX,x,c,c0,cas,opts,pdfm)
% the outer integrand of DUZERO at the pairs (b, x(IX)): sqrt(b) h (pdf) or a'(b) G (cdf)
X = x(IX); X = X(:); B = B(:);
A = 1/2 + X.*sqrt(B) + c0*B;
H = zeros(size(B)); E = H; ch = getf(opts,'chunk',4000);
for i = 1:ch:numel(B)
    j = (i:min(i+ch-1,numel(B)))';
    [H(j),E(j)] = iipairs(B(j),A(j),c,cas,opts,pdfm);
end
if pdfm, fac = sqrt(B); else, fac = X./(2*sqrt(B)) + c0; end
H = H.*fac; E = E.*abs(fac);
end

function [F,err,nun,npairs,nref] = adaptb(PA,PL,PEL,PER,PX,nx,zs,ws,ze,we,evalH,otol,maxlev)
% Adaptive outer rule in log b.  Each piece [PA, PA + PL] of point PX carries
% the M-point rule (zs, ws) on [0, 1], or (ze, we), the cosine-mapped rule,
% when it ends at a zero of a(b) (PEL, PER).  The Legendre coefficients C_n of
% the integrand in the rule's variable at its Gauss-Legendre nodes estimate the
% error: the largest of the last two, extrapolated to n = 3M/2 by their decay
% from the middle (est = C_hi^3/C_mid^2 when C_hi < C_mid, otherwise C_hi).
% Pieces with est > max(otol, 10 times the noise floor of the inner error
% estimates) are bisected, up to maxlev times; the estimates of the
% accepted pieces, plus the rule applied to the error estimates of the inner
% integrals, make up err.  At large x and c > 0 the integrand has a sharp step
% in log b (near x sqrt(b) = c b, where R crosses a(b)), which the fixed rule
% does not resolve.  evalH(B, IX) returns the outer integrand and its error.
M = numel(zs); [zg,wg] = gl01(M); zg = zg(:); wg = wg(:); xg = 2*zg - 1;
Pn = zeros(M); Pn(:,1) = 1; if M > 1, Pn(:,2) = xg; end
for n = 1:M-2, Pn(:,n+2) = ((2*n + 1)*xg.*Pn(:,n+1) - n*Pn(:,n))/(n + 1); end
LM = ((2*(0:M-1)' + 1)/2).*(Pn.'.*(2*wg.'));        % C = LM * (values at the nodes)
mh = floor(M/2);
F = zeros(nx,1); err = F; nref = F; npairs = 0;
PA = PA(:).'; PL = PL(:).'; PEL = logical(PEL(:).'); PER = logical(PER(:).'); PX = PX(:).';
Ha = cell(1,maxlev+1); Ea = Ha; Wa = Ha; Xa = Ha;
for lev = 0:maxlev
    np = numel(PA); if np == 0, break, end
    ed = PEL | PER;
    Z = repmat(zs(:),1,np); Wq = repmat(ws(:),1,np);
    if any(ed), Z(:,ed) = repmat(ze(:),1,nnz(ed)); Wq(:,ed) = repmat(we(:),1,nnz(ed)); end
    B = exp(PA + PL.*Z); W = PL.*Wq.*B;
    IXm = repmat(PX,M,1);
    [H,E] = evalH(B(:),IXm(:)); H = reshape(H,M,np); E = reshape(E,M,np);
    npairs = npairs + M*np;
    G = W.*H; G(W == 0) = 0; I = sum(G,1);
    C = abs(LM*(G./wg));
    if M >= 4
        chi = max(C(M,:),C(M-1,:)); cmid = max(C(mh+1,:),C(mh,:));
        est = chi; dec = chi < cmid; est(dec) = chi(dec).^3./cmid(dec).^2;
    else
        est = zeros(1,np);
    end
    est(~isfinite(est)) = 0;
    % noise floor: the inner error estimates and rounding (bisection cannot
    % reduce the coefficients below it)
    WE = abs(W).*E; WE(W == 0) = 0;
    nse = sum(WE,1) + 1e3*eps*sum(abs(G),1);
    rf = est > max(otol,10*nse) & lev < maxlev; ok = ~rf;
    if any(ok)
        F = F + accumarray(PX(ok).',I(ok).',[nx 1]);
        err = err + accumarray(PX(ok).',(est(ok) + sum(WE(:,ok),1)).',[nx 1]);
        Ha{lev+1} = reshape(H(:,ok),[],1); Ea{lev+1} = reshape(E(:,ok),[],1);
        Wa{lev+1} = reshape(W(:,ok),[],1); Xa{lev+1} = reshape(IXm(:,ok),[],1);
    end
    if ~any(rf), break, end
    nref = nref + accumarray(PX(rf).',1,[nx 1]);
    h = PL(rf)/2;
    PA = [PA(rf), PA(rf) + h]; PL = [h, h]; PX = [PX(rf), PX(rf)];
    PEL = [PEL(rf), false(1,nnz(rf))]; PER = [false(1,nnz(rf)), PER(rf)];
end
H = vertcat(Ha{:}); E = vertcat(Ea{:}); W = vertcat(Wa{:}); IX = vertcat(Xa{:});
if isempty(IX)
    nun = zeros(nx,1);
else
    sc = accumarray(IX,abs(H),[nx 1],@max);
    nun = accumarray(IX,double(E > 1e-9*sc(IX) & abs(W).*E > 1e-14),[nx 1]);
end
end


function P = Pk0(k1,k2,a)
% P[k1 Z1^2 + k2 Z2^2 <= a], k1 > 0 > k2: 2 int_{l0}^Inf phi(l; s2) erf(sqrt((2a + l^2)/(2 s1))) dl
s1 = 2*k1; s2 = -2*k2; l0 = sqrt(max(0,-2*a)); sd = sqrt(s2);
f = @(l) 2*exp(-l.^2/(2*s2))/sqrt(2*pi*s2).*erf(sqrt(max(2*a + l.^2,0)/(2*s1)));
P = integral(f,l0,l0+40*sd,'AbsTol',1e-15,'RelTol',1e-13);
end

function [H,Ek,info] = iipairs(b,r,c,cas,o,pdfm)
% Cases DU, DT, zero start: for pairs (b, r),
%   cdf (pdfm false): G(r,b) = (1/2 pi i) int J(s;r) e^{-s b}/s ds   (Re s > 0),
%   pdf: h(r,b) = (1/2 pi i) int J(s;r) e^{-s b} ds,
% J(s;r) = E[e^{sB} delta(R - r)] = M_B(s) g_s(r) in closed form (IIPAR):
%   log J = -c/2 - lcd/2 - log pi - log(-A0)/2 - 2 A2 r/(A1+S) + log K0s(S r/(-2 A0)), r > 0
%   (r < 0: (A1+S)|r|/(2 A0) in place of the exponential term), S = sqrt(A1^2 - 4 A0 A2),
% analytic through the end of the domain of M_B.  Saddle point of log J - s b on the
% real axis; paths W (parabola/wedge) and T (vertical line in th), as for case DK,
% accepted when they agree (AGREEPATHS).
NN=getf(o,'Nv',96); kap=2; tl=45; tol=1e-13;
b=b(:)'; r=r(:)'; n=numel(b); r(abs(r)<1e-200)=1e-200;     % (log singularity of g at r = 0)
bb=b; rr=r;
lk=@(S,q) iilog(S,repmat(bb(q),size(S,1),1),c,cas,repmat(rr(q),size(S,1),1));
sab=iiabs(c,cas);                         % abscissa of J on the real axis
[s,a1,g2,L0]=rsad(lk,bb,sab);
% a saddle point pinned at the abscissa (a branch point beyond which J is not
% analytic, as at cosh th = 0): cross the axis 1/b to its left, with width 1/b
pin=s>sab-1./bb; s(pin)=sab-1./bb(pin); a1(pin)=1./bb(pin); g2(pin)=bb(pin).^2;
L0(pin)=real(lk(s(pin),find(pin)));
Lh=L0-log(2*pi*max(g2,1e-6*bb.^2))/2;
% negligible in the outer measure (pdf: sqrt(b) b d log b; cdf: a'(b) b d log b)
if pdfm, lw=1.5*log(bb); else, lw=log(bb)+log(1+abs(c)+abs(rr)./bb)-log(max(abs(s),a1)); end   % (cdf: the factor 1/s)
act=find(~(Lh+lw<-35));
[Hk,Ekk,pk]=agreepaths(@evalp,2,act,n,NN,tol);
sk=true(1,n); sk(act)=false;
if ~pdfm
  % negligible path integral, but crossing left of the pole at 0 leaves its residue
  % J(0;r), the density of R (the cdf integrand is the density on {B > b})
  q=find(sk & s<0); if ~isempty(q), Hk(q)=real(exp(lk(0*s(q),q))); end
end
H=Hk(:); Ek=Ekk(:); info=struct('s',s,'path',pk,'Lh',Lh,'sab',sab);

  function [I,E]=evalp(pth,q,NNq)
    % the value and the nested-rule estimate of path pth at the pairs q
    if pth==1, [I,E]=pthW(q,NNq); else, [I,E]=pthT(q,NNq); end
  end
  function [I,E]=pthW(q,NN)
    bq=bb(q); mu=a1(q)/2; sq=s(q);
    if ~pdfm
      % keep the crossing point at least one saddle-point width from the pole at 0
      near=abs(sq)<a1(q); sq(near & sq>=0)=a1(q(near & sq>=0)); sq(near & sq<0)=-a1(q(near & sq<0));
      sq=min(sq,sab-0.5*a1(q));
    end
    tmax=max(10,(tl+5)./(2*kap*bq.*mu));
    ht=asinh(tmax)/NN; TAU=(0:NN)'*ht; TH=sinh(TAU); Wt=cosh(TAU).*ht; Wt(1,:)=Wt(1,:)/2;
    rq=sqrt(1+(TH/(2*kap)).^2);
    S=sq+mu.*(TH.^2./rq)-2i*mu.*TH; dS=mu.*(2*TH./rq-TH.^3./(4*kap^2*rq.^3))-2i*mu;
    G=exp(lk(S,q)).*(1i*dS/pi).*Wt;
    if ~pdfm, G=G./S; end
    [I,E]=fin(G);
    if ~pdfm, lf=sq<0; if any(lf), I(lf)=I(lf)+real(exp(lk(0*sq(lf),q(lf)))); end, end
  end
  function [I,E]=pthT(q,NN)
    bq=bb(q); sq=s(q); a1q=a1(q);
    qh=c^2-2*sq; th0=zeros(size(qh)); th0(qh>0)=sqrt(qh(qh>0));
    tA=max(max(th0,asinh(abs(rr(q))./sqrt(6*bq))/2),1);
    if ~pdfm
      % crossing point s_A = (c^2 - tA^2)/2 at least a1 from the pole at 0
      sA=(c^2-tA.^2)/2; near=abs(sA)<a1q;
      if any(near), tA(near)=sqrt(max(c^2+2*a1q(near),1)); end
    end
    sA=(c^2-tA.^2)/2;
    hq=1e-4*a1q; L3=real(lk([sA-hq;sA;sA+hq],q));
    Ls=(L3(3,:)-L3(1,:))./(2*hq); Lss=(L3(3,:)-2*L3(2,:)+L3(1,:))./hq.^2;
    sg=1./sqrt(max(Lss.*tA.^2-Ls,bq)); bd=~isfinite(sg); sg(bd)=1./sqrt(bq(bd));
    ym=max(12*sg,sqrt(2*(tl+10)./bq));
    um=asinh(ym./sg); U=(0:NN)'*(um/NN); Y=sg.*sinh(U); Wt=sg.*cosh(U).*(um/NN); Wt(1,:)=Wt(1,:)/2;
    T=tA+1i*Y; S=(c^2-T.^2)/2;
    G=exp(lk(S,q)).*T.*Wt/pi;
    if ~pdfm, G=G./S; end
    [I,E]=fin(G);
    if ~pdfm, lf=sA<0; if any(lf), I(lf)=I(lf)+real(exp(lk(0*sA(lf),q(lf)))); end, end
  end
end

function L = iilog(S,b,c,cas,r)
% log J(s;r) - s b, continuous along each column (the nodes of a path, in order)
P=iipar(S,c,cas);
lcd=P.lcd; A0=P.A0; A1=P.A1; A2=P.A2;
if size(S,1)>1, lcd=real(lcd)+1i*uwc(imag(lcd)); end
la0=log(-A0); if size(S,1)>1, la0=real(la0)+1i*uwc(imag(la0)); end
Sq=sqrt(A1.^2-4*A0.*A2);                       % principal: Re S >= 0 (g decays in |r|)
z=Sq.*abs(r)./(-2*A0);
ex=-2*A2.*r./(A1+Sq); ng=r<0; ex(ng)=(A1(ng)+Sq(ng)).*abs(r(ng))./(2*A0(ng));
% -s b = del (th + c) b/2
L=-c/2-lcd/2-log(pi)-la0/2+ex+log(besselk(0,z,1))+P.del.*(P.th+c).*b/2;
end

function sab = iiabs(c,cas)
% abscissa on the real axis: the first s > 0 where cosh th det Sig, A1^2 - 4 A0 A2 or
% -A0 leaves (0, Inf)
okf=@(s) okp(iipar(s,c,cas));
% cosh th = 0 at th = i pi/2 (a branch point of (cosh th det Sig)^(-1/2)) bounds it
sp=(c^2+pi^2/4)/2;
lo=0; hi=sp*(1-1e-12);
if okf(hi), sab=hi; return, end
for k=1:200, m=(lo+hi)/2; if m==lo||m==hi, break, end; if okf(m), lo=m; else, hi=m; end, end
sab=lo;
end
function t = okp(P)
v=exp(P.lcd); t=real(v)>0 && abs(imag(v))<1e-9*abs(v) && real(P.A1^2-4*P.A0*P.A2)>0 && real(-P.A0)>0;
end

function P = iipar(s,c,cas)
% Cases DU, DT, zero start, by Girsanov: under the tilt exp(-th^2/2 int W^2)
% of Brownian motion (th = sqrt(c^2 - 2s)), X = (W(1), int W[, int r W]) is
% Gaussian with covariance Sig(th); e^(sB) gains exp(X'QX/2), Q = diag(c, -2s Gm),
% and R = X'KX/2.  Returns, elementwise in s:
%   lcd = log(cosh th det Sig), D = det(I - Q Sig) (cancellation-free),
%   A2 = det P' = D/det Sig, A1, A0: det(K - nu P') = 0  <=>  A2 nu^2 - A1 nu + A0 = 0
%   (P' = Sig^-1 - Q; nu = 2 kap the weights of R under the tilt), logMB.
T=[1.0 -0.33333333333333333 0.13333333333333333 -0.053968253968253968 0.021869488536155203 -0.0088632355299021966 0.003592128036572481 -0.0014558343870513183 0.00059002744094558598 -0.00023912911424355248 9.6915379569294503e-5 -3.9278323883316834e-5 1.5918905069328965e-5 -6.4516892156554308e-6 2.6147711512907546e-6 -1.0597268320104654e-6 4.2949110782738059e-7 -1.7406618963571648e-7 7.0546369464009683e-8 -2.8591366623052539e-8 1.1587644432798852e-8 -4.6962953982309016e-9 1.9033368339312759e-9 -7.7139336353590623e-10 3.126339545892087e-10 -1.2670576930305401e-10 5.1351914080393677e-11 -2.0812146867700474e-11 8.4348454190943383e-12 -3.4185140868111558e-12 1.3854715742948469e-12 -5.6151047924146801e-13 2.2757162553728748e-13 -9.2231305851395317e-14 3.7379940310967389e-14 -1.5149519187148605e-14 6.1398688626168138e-15 -2.4883951222762789e-15 1.0085085566354096e-15 -4.0873312268690138e-16 1.6565329513786284e-16 -6.7136751750487117e-17 2.7209500613044588e-17 -1.1027595233722371e-17 4.4693160072374125e-18];
E=[1.0 -0.5 0.20833333333333333 -0.084722222222222222 0.034350198412698413 -0.013922233245149912 0.0056424968100315323 -0.0022868190951648293 0.00092681292737742184 -0.0003756231338525945 0.00015223432221797662 -6.1698246877700525e-5 2.5005357609459247e-5 -1.0134289721572026e-5 4.1072729198567004e-6 -1.664615015128028e-6 6.7464305456632013e-7 -2.7342253129896734e-7 1.1081397802278207e-7 -4.4911213669537136e-8 1.820182931124577e-8 -7.3769235610848765e-9 2.9897545073926764e-9 -1.2117018619561619e-9 4.9108426750009155e-10 -1.9902895700495881e-10 8.066339801136952e-11 -3.2691643852499818e-11 1.3249424201396147e-11 -5.3697893706595739e-12 2.1762936597810882e-12 -8.8201859824934e-13 3.5746867347671487e-13 -1.4487659644686842e-13 5.871627293628006e-14 -2.3796809091881837e-14 9.6444834564008509e-15 -3.9087619176859166e-15 1.5841615363041244e-15 -6.4203648775599251e-16 2.6020758752402585e-16 -1.0545816304260601e-16 4.2740583616893926e-17 -1.7322106086512011e-17 7.0203851674541611e-18 -2.8452549391662691e-18];
q=c^2-2*s; th=sqrt(q);
if c>0, del=-2*s./(th+c); else, del=th-c; end
sm=abs(th)<1; e2=exp(-2*th);
% g1 = tanh/th, g2 = (1-sech)/th^2, g3 = (th-tanh)/th^3, g4 = (th^2-2+2sech)/(2th^4),
% g5 = (th^3/3-th+tanh)/th^5, t1 = 1 - tanh, lch = log cosh
tn=(1-e2)./(1+e2); sc=2*exp(-th)./(1+e2);
g1=tn./th; g2=(1-sc)./th.^2; g3=(th-tn)./th.^3; g4=(th.^2-2+2*sc)./(2*th.^4); g5=(th.^3/3-th+tn)./th.^5;
t1=2*e2./(1+e2); lch=th+log((1+e2)/2);
if any(sm(:))
  Q=q(sm); h=@(cf) polyval(fliplr(cf),Q);
  g1(sm)=h(T); g2(sm)=-h(E(2:end)); g3(sm)=-h(T(2:end)); g4(sm)=h(E(3:end)); g5(sm)=h(T(3:end));
  t1(sm)=1-th(sm).*g1(sm); lch(sm)=log(cosh(th(sm)));
end
% N11 = 1 - c g1 = (th - c tanh)/th = (del + c t1)/th (cancellation-free); small th: direct
N11=(del+c*t1)./th; N11(sm)=1-c*g1(sm);
if cas(2)=='U'
  s11=g1; s12=g2; s22=g3;
  dS=s11.*s22-s12.^2;
  D=N11.*(1+2*s.*s22)+2*c*s.*s12.^2;           % det(I - Q Sig), Q = diag(c,-2s)
  A2=D./dS;
  A1=(s11-2*s12)./dS+2*s;                       % tr(adj K P'), K = [1 -1; -1 0]
  A0=-ones(size(s));
else
  % X = (W1, m0, m1); Gm = [4 -6; -6 12]
  S=cell(3); S{1,1}=g1; S{1,2}=g2; S{1,3}=g3; S{2,2}=g3; S{2,3}=g4; S{3,3}=g5;
  S{2,1}=S{1,2}; S{3,1}=S{1,3}; S{3,2}=S{2,3};
  Qd={c,0,0;0,-8*s,12*s;0,12*s,-24*s};          % Q = blkdiag(c, -2s Gm)
  N=cell(3); for i=1:3, for j=1:3, N{i,j}=(i==j)-(Qd{i,1}.*S{1,j}+Qd{i,2}.*S{2,j}+Qd{i,3}.*S{3,j}); end, end
  N{1,1}=N11;                                   % (row 1 of Q is (c,0,0))
  D=det3(N); dS=det3(S);
  A2=D./dS;
  Si=adj3(S); Pp=cell(3); for i=1:3, for j=1:3, Pp{i,j}=Si{i,j}./dS-Qd{i,j}; end, end
  % P'11 = (Sig^-1)11 - c: from P' = N Sig^-1 (row 1 of N is cancellation-free)
  Pp{1,1}=(N{1,1}.*Si{1,1}+N{1,2}.*Si{2,1}+N{1,3}.*Si{3,1})./dS;
  K=[1 2 -6; 2 -12 12; -6 12 0]; aK=adj3n(K);
  aP=adj3(Pp);
  A1=0; A0=0; for i=1:3, for j=1:3, A1=A1+K(i,j)*aP{j,i}; A0=A0+aK(i,j)*Pp{j,i}; end, end
end
P=struct('th',th,'del',del,'lcd',lch+log(dS),'D',D,'A2',A2,'A1',A1,'A0',A0,'dS',dS,'lch',lch);
P.logMB=-c/2-lch/2-log(D)/2;
end
function d=det3(M)
d=M{1,1}.*(M{2,2}.*M{3,3}-M{2,3}.*M{3,2})-M{1,2}.*(M{2,1}.*M{3,3}-M{2,3}.*M{3,1})+M{1,3}.*(M{2,1}.*M{3,2}-M{2,2}.*M{3,1});
end
function A=adj3(M)
A=cell(3);
A{1,1}=M{2,2}.*M{3,3}-M{2,3}.*M{3,2}; A{1,2}=-(M{1,2}.*M{3,3}-M{1,3}.*M{3,2}); A{1,3}=M{1,2}.*M{2,3}-M{1,3}.*M{2,2};
A{2,1}=-(M{2,1}.*M{3,3}-M{2,3}.*M{3,1}); A{2,2}=M{1,1}.*M{3,3}-M{1,3}.*M{3,1}; A{2,3}=-(M{1,1}.*M{2,3}-M{1,3}.*M{2,1});
A{3,1}=M{2,1}.*M{3,2}-M{2,2}.*M{3,1}; A{3,2}=-(M{1,1}.*M{3,2}-M{1,2}.*M{3,1}); A{3,3}=M{1,1}.*M{2,2}-M{1,2}.*M{2,1};
end
function A=adj3n(M)
A=[M(2,2)*M(3,3)-M(2,3)*M(3,2), -(M(1,2)*M(3,3)-M(1,3)*M(3,2)), M(1,2)*M(2,3)-M(1,3)*M(2,2);
  -(M(2,1)*M(3,3)-M(2,3)*M(3,1)), M(1,1)*M(3,3)-M(1,3)*M(3,1), -(M(1,1)*M(2,3)-M(1,3)*M(2,1));
   M(2,1)*M(3,2)-M(2,2)*M(3,1), -(M(1,1)*M(3,2)-M(1,2)*M(3,1)), M(1,1)*M(2,2)-M(1,2)*M(2,1)];
end

function [s,a1,g2,L0] = rsad(Lf,bb,smx)
% minimum of a real convex function f(s) (one per column; Lf(S,q) evaluates the
% columns q at the rows of S), s < smx: bracketing, then the Illinois variant of
% regula falsi on the derivative, then the curvature at the minimum.  The steps
% of the central differences are set from the width at the starting point (a
% scan of second differences over steps sc 10^k), above the rounding noise.
n = numel(bb); sc = 1./bb; ia = 1:n;
s0 = min(-sc, smx - sc/2); t0 = -(sc/2).^2/2 - abs(smx); s0(bb < 0.05 & t0 < s0) = t0(bb < 0.05 & t0 < s0);
w0 = wscan(Lf,s0,sc,smx,ia);
hd = @(s,w) max(1e-4*w,1e-12*abs(s));
g = @(s,h,q) dfd(Lf,s,h,q);
sL = s0; gL = g(sL,hd(sL,w0),ia);
sR = smx - 1e-12*max(1,abs(smx))*ones(1,n); 
for k = 1:200                                            % move left until f' < 0
    m = ~(gL < 0); if ~any(m), break, end
    sR(m) = min(sR(m),sL(m)); sL(m) = sL(m) - max(sc(m),w0(m)).*4.^k; q = find(m);
    gL(m) = g(sL(m),hd(sL(m),w0(m)),q);
end
gR = g(sR,hd(sR,w0),ia); gR(~isfinite(gR) | gR <= 0) = Inf;
side = zeros(1,n);
for it = 1:300
    w = sR - sL; act = w > 1e-9*w0; if ~any(act), break, end
    sn = (sL + sR)/2; fin = isfinite(gR);
    sn(fin) = sL(fin) - gL(fin).*w(fin)./(gR(fin) - gL(fin));
    bad = ~(sn > sL & sn < sR); sn(bad) = (sL(bad) + sR(bad))/2;
    gn = g(sn,hd(sn,w0),ia);
    up = act & gn < 0; dw = act & ~(gn < 0);
    sL(up) = sn(up); gL(up) = gn(up); gR(up & side == -1) = gR(up & side == -1)/2; side(up) = -1;
    sR(dw) = sn(dw); gR(dw) = gn(dw); gL(dw & side == 1) = gL(dw & side == 1)/2; side(dw) = 1;
    gR(~isfinite(gR)) = Inf;
end
s = (sL + sR)/2;
[a1,L0] = wscan(Lf,s,sc,smx,ia); g2 = 1./a1.^2;
end
function [a1,L0] = wscan(Lf,s,sc,smx,ia)
% width: the step d = sc 10^k at which (f(s+d) + f(s-d))/2 - f(s) is closest to 1/2
n = numel(s); K = -6:12; nK = numel(K);
D = sc(:)'.*10.^K(:); D = min(D,repmat(max((smx - s)/2,eps*abs(s)),nK,1));
Lm = real(Lf([s; s - D; s + D],ia)); L0 = Lm(1,:);
Dl = (Lm(2:nK+1,:) + Lm(nK+2:end,:))/2 - L0;
[~,kb] = min(abs(log(max(Dl,realmin)/0.5)),[],1);
id = sub2ind(size(Dl),kb,1:n); dd = D(id); g2 = 2*Dl(id)./dd.^2;
bad = ~(g2 > 0) | ~isfinite(g2); g2(bad) = 1./sc(bad).^2;
a1 = 1./sqrt(g2);
end
function d = dfd(Lf,s,h,q)
L = real(Lf([s-h; s+h],q)); d = (L(2,:) - L(1,:))./(2*h);
d(~isfinite(L(2,:))) = Inf;
end
function p = uwc(p)
% phase unwrapped down each column (as unwrap(p,[],1), vectorized over the columns)
d = diff(p,1,1); p(2:end,:) = p(2:end,:) - 2*pi*cumsum(round(d/(2*pi)),1);
end
