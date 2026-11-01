function [L,ok,kap] = tr_cflim(tA,tB,cas,c,form)
%TR_CFLIM  Limiting joint characteristic functions of the Dickey-Fuller functionals.
%
%   [L,ok] = TR_CFLIM(tA,tB,cas,c,form) returns the logarithm of a joint
%   characteristic function of the limiting functionals under phi = 1 + c/n,
%   fixed start, no drift.  W is a standard Brownian motion, J the
%   Ornstein-Uhlenbeck process dJ = cJ dr + dW, J(0) = 0, and Jt the process
%   used by the regression: J itself (cas 'DK'), J demeaned ('DU'), or J
%   demeaned and detrended ('DT').  With
%      A = int_0^1 Jt dW,   B = int_0^1 Jt^2 dr,   A' = A + c B,
%   n(phihat - phi) => A/B and the t-ratio => A/sqrt(B).  form:
%      'AB'  (default)  L = log E exp(i tA A  + i tB B)
%      'ApB'            L = log E exp(i tA A' + i tB B)
%      'JB'  (case DK)  L = log E exp(i tA J(1) + i tB B)
%      'B'              L = log E exp(i tB B) (tA ignored), and kap (2 x N): the
%                       weights of R = A' + 1/2 under the B-tilt, i.e.
%                       E[exp(i tB B + i u R)] = exp(L) prod_k (1 - 2 i u kap_k)^(-1/2),
%                       kap(1) the one with the larger real part (kap(2) = 0 in case DK)
%      'F'              the coefficients F0, F1, F2 of D at z = 2 i tB - c^2
%                       (kap, scaled by exp(-L); used by TR_LIMMOM)
%      'BC'             as 'B', with tB a matrix whose columns are paths, each
%                       starting at a real point of the domain (i tB real, D > 0);
%                       the branch of log D is continued along each column
%   tA, tB: arrays of a common size (or scalars), real or complex, with the real
%   parts of (i tA, i tB) in the domain of the moment generating function.
%
%   Closed form (Nabeya and Tanaka, 1990, Theorem 4, Models 1, 2 and 4):
%      E exp(i tA A' + i tB B) = exp(-i tA/2) D(lambda, z)^(-1/2),
%      lambda = -i tA,  z = 2 i tB - c^2,
%   with D = e^c [F0(z) + F1(z) lambda + F2(z) lambda^2] a polynomial in lambda
%   (degree 1 in case DK, 2 in cases DU and DT) whose coefficients are entire
%   functions of z, evaluated by their Taylor series (NT_SERIES) for |z| < 4 and
%   otherwise from the closed forms written in theta = (-z)^(1/2) (principal
%   root) and exp(-2 theta), with exp(theta) factored out.  In case DK,
%   E exp(i tA J(1) + i tB B) = [e^c F0(z)]^(-1/2) exp(-(tA^2/2) F1(z)/F0(z)).
%
%   Branch.  On a vertical line of the moment generating function (fixed real
%   parts), D does not vanish, and the continuous logarithm is obtained along
%   the two-leg path from the real point: first in tB with lambda fixed at its
%   real part (a function of one variable, unwrapped on a refined grid), then in
%   lambda with z fixed, along which the polynomial D sweeps the angle
%   sum_k Arg[(lambda - rho_k)/(Re lambda - rho_k)] exactly (rho_k its roots).
%   ok (meaningful at real moment generating function arguments): D > 0 on the
%   segment from the origin.
%
%   Explosive side.  For c > 0 the characteristic function varies on the scale
%   tA = O(e^-c), tB = O(e^-2c), where D e^(-c-theta) is O(e^-2c) while the
%   terms of the polynomial in lambda are O(e^-c) and cancel (at the origin
%   F0 = e^-2c is a difference of O(1) numbers).  There D is evaluated from
%   cancellation-free forms (subfunction DSTAB), and the angle swept along
%   lambda from the ratio of such values; the result is accurate to a few units
%   of rounding for c up to 300 or so.
%
%   See also TR_LIMDIST, TR_LIMMOM, TR_NSSTART, NT_SERIES.
if nargin < 5 || isempty(form), form = 'AB'; end
cas = upper(cas); form = upper(form);
switch cas
    case 'DK', j = 1;
    case 'DU', j = 2;
    case 'DT', j = 4;
    otherwise, error('tr_cflim: case must be DK, DU or DT.');
end
if strcmp(form,'BC')
    [L,ok,kap] = pathB(tB,j,-c); return
end
if strcmp(form,'F')
    % the coefficients of D(lambda, z) e^(-c) = F0 + F1 lambda + F2 lambda^2 at
    % z = 2 i tB - c^2, scaled by exp(-ls): kap = F (3 x N), L = ls (no branch
    % continuation; used by TR_LIMMOM near the real axis)
    [kap,L] = coefs(2i*tB(:).' - c^2,j,-c); ok = true(size(L)); return
end
sz = size(tA); if isscalar(tA), sz = size(tB); end
tA = tA(:).' + zeros(1,prod(sz)); tB = tB(:).' + zeros(1,prod(sz));
% tau0 = t1^2 - 2 i t2 in the coefficients (t1, t2) of (A, B), formed before
% the shift tB - c tA, which cancels on the explosive side (see DSTAB)
tau0 = [];
if strcmp(form,'AB'), tau0 = tA.^2 - 2i*tB; tB = tB - c*tA; end   % A = A' - c B
if strcmp(form,'APB'), tau0 = tA.^2 - 2i*(tB + c*tA); end
b = -c;                                              % Nabeya-Tanaka's c
sB = 1i*tB;                                          % mgf argument of B
if strcmp(form,'JB') || strcmp(form,'B')
    if j ~= 1 && strcmp(form,'JB'), error('tr_cflim: form JB is for case DK.'); end
    lam = zeros(size(tA));
else
    lam = -1i*tA;
end
% ---- log D at the points, with the continuous branch
[lD,ok] = logD(lam,sB,j,b,tau0);
kap = [];
if strcmp(form,'JB')
    [F,~] = coefs(2*sB - b^2,j,b);
    L = -lD/2 - (tA.^2/2).*F(2,:)./F(1,:);
elseif strcmp(form,'B')
    L = -lD/2;
    [F,~] = coefs(2*sB - b^2,j,b);
    p1 = F(2,:)./(2*F(1,:)); p2 = F(3,:)./(4*F(1,:));    % kap1 + kap2, kap1 kap2
    dsc = sqrt(p1.^2 - 4*p2);
    k1 = (p1 + dsc)/2; k2 = (p1 - dsc)/2;
    sw = real(k2) > real(k1); t = k1(sw); k1(sw) = k2(sw); k2(sw) = t;
    if j == 1, k1 = p1; k2 = zeros(size(p1)); end     % rank one
    kap = reshape([k1; k2],[2 sz]);
else
    L = -1i*tA/2 - lD/2;
end
real0 = imag(lam) == 0 & imag(sB) == 0 & ok;          % real mgf arguments: L real
L(real0) = real(L(real0));
L = reshape(L,sz); ok = reshape(ok,sz);
end

% =========================================================================
function [L,ok,kap] = pathB(tB,j,b)
% log E exp(i tB B) continued along the columns of tB, and the weights kap
sz = size(tB); sB = 1i*tB; L = complex(zeros(sz)); ok = true(1,sz(2));
for q = 1:sz(2)
    sq = sB(:,q).';
    [lq,okq] = lq_path(sq,j,b);
    L(:,q) = -lq.'/2; ok(q) = okq;
end
[F,~] = coefs(2*sB(:).' - b^2,j,b);
p1 = F(2,:)./(2*F(1,:)); p2 = F(3,:)./(4*F(1,:));
dsc = sqrt(p1.^2 - 4*p2);
k1 = (p1 + dsc)/2; k2 = (p1 - dsc)/2;
sw = real(k2) > real(k1); t = k1(sw); k1(sw) = k2(sw); k2(sw) = t;
if j == 1, k1 = p1; k2 = zeros(size(p1)); end         % rank one
kap = reshape([k1; k2],[2 sz]);
end

function [lD,ok] = lq_path(s,j,b)
% log D(0, 2 s - b^2) continuous along the path s (a row), from its real start:
% log D = c + theta + log q, theta = (-z)^(1/2), q = D e^(-c - theta) slowly
% varying, its logarithm unwrapped on a refined path
n = numel(s); t = (0:n-1); tt = t;
[lv,th] = qeval(s,j,b);
for it = 1:40
    d = angle(exp(1i*(imag(lv(2:end)) - imag(lv(1:end-1)))));
    bad = find(abs(d) > pi/4);
    if isempty(bad), break, end
    tm = (tt(bad) + tt(bad+1))/2;
    fl = min(floor(tm),n-2); fr = tm - fl;
    sm = s(fl+1).*(1-fr) + s(fl+2).*fr;
    [lm,thm] = qeval(sm,j,b);
    tt = [tt tm]; lv = [lv lm]; th = [th thm];
    [tt,ix] = sort(tt); lv = lv(ix); th = th(ix);
end
d = angle(exp(1i*(imag(lv(2:end)) - imag(lv(1:end-1)))));
ph = [0 cumsum(d)] + imag(lv(1));
lt = real(lv) + 1i*ph + th;
% at the real start D > 0: log D must be real there
ok = abs(imag(s(1))) == 0;
k0 = lt(1); m = round(imag(k0)/(2*pi)); lt = lt - 2i*pi*m;
ok = ok && abs(imag(lt(1))) < 1e-8;
[~,loc] = ismember(t,tt);
lD = -b + lt(loc);
end

function [lv,th] = qeval(s,j,b)
% log q = log(D e^(-c)) - theta, theta the principal root of -z (continuous on
% paths in the lower half plane of s, where imag(-z) > 0)
z = 2*s - b^2; th = sqrt(-z);
[F,ls] = coefs(z,j,b);
Dt = F(1,:);                                       % lambda = 0
% coefs scales by exp(-ls), ls = principal root or 0; rescale to exp(-th)
lv = log(Dt) + (ls - th);
end

% =========================================================================
function [lD,ok] = logD(lam,sB,j,b,tau0)
% continuous log D(lambda, z), z = 2 sB - b^2, on the vertical planes through
% the real parts
N = numel(lam); lD = complex(zeros(1,N)); ok = true(1,N);
if isempty(tau0), tau0 = NaN(1,N); end
lr = real(lam); sr = real(sB);
rp = imag(lam) == 0 & imag(sB) == 0;
if any(rp)
    % real points (real mgf arguments): D must be positive, also on the
    % segment from the origin; log D is then real
    ir = find(rp);
    z = 2*sr(ir) - b^2;
    l = lr(ir); [Dt,ls] = dval(l,z,tau0(ir),j,b);
    Dr = real(exp(ls).*Dt);
    okr = Dr > 0;
    if any(okr)
        fr = (1:15)'/16; K = numel(ir);
        zz = fr*(2*sr(ir)) - b^2; ll = fr*l;
        [Ds,lss] = dval(ll(:).',zz(:).',[],j,b);
        Ds = real(exp(lss).*Ds);
        okr = okr & all(reshape(Ds,15,K) > 0,1);
    end
    lD(ir) = -b + real(ls) + log(abs(Dt)) + 1i*pi*(Dr <= 0);
    ok(ir) = okr;
end
cp = find(~rp);
if isempty(cp), return, end
[~,~,ig] = unique([lr(cp)' sr(cp)'],'rows');
for q = 1:max(ig)
    ii = cp(ig == q); l0 = lr(ii(1)); s0 = sr(ii(1));
    % B-leg: log D(l0, s0 + i y) continuous in y from y = 0
    y = imag(sB(ii));
    [phB,ok0] = bleg(l0,s0,y,j,b);
    % A-leg: from l0 to lam at fixed sB
    z = 2*sB(ii) - b^2;
    [F,~] = coefs(z,j,b);
    lm = lam(ii);
    [Dt,ls] = dval(lm,z,tau0(ii),j,b);
    D0 = dval(l0+0*lm,z,[],j,b);
    % the angle swept by D along the segment from l0 to lm (fixed z), as the
    % sum of the angles swept by its linear factors lambda - r_k; each is the
    % principal argument of a ratio.  The factor of the root that may lie
    % close to the segment is obtained from the ratio Dt/D0 of the stable
    % values (DVAL), so that no difference of nearly equal numbers is formed
    dA = zeros(1,numel(ii));
    a2 = F(3,:); a1 = F(2,:); a0 = F(1,:);
    lin = abs(a2) <= 1e-14*(abs(a1) + abs(a0));        % degree one (case DK)
    if any(lin)
        dA(lin) = angle(Dt(lin)./D0(lin));             % = angle((lm - rho)/(l0 - rho))
    end
    qd = ~lin;
    if any(qd)
        dsc = sqrt(a1(qd).^2 - 4*a2(qd).*a0(qd));
        sg = sign(real(conj(a1(qd)).*dsc)); sg(sg == 0) = 1;
        rb = (-a1(qd) - sg.*dsc)./(2*a2(qd));          % the root of larger modulus
        dA(qd) = angle((Dt(qd)./D0(qd)).*(l0 - rb)./(lm(qd) - rb)) + angle((lm(qd) - rb)./(l0 - rb));
    end
    dA(lm == l0) = 0;
    lD(ii) = -b + real(ls) + log(abs(Dt)) + 1i*(phB + dA);
    ok(ii) = ok0;
end
end

% =========================================================================
function [ph,ok] = bleg(l0,s0,y,j,b)
% imaginary part of the continuous log D(l0, s0 + i y), starting from the real
% point y = 0 (where D > 0 in the domain), by unwrapping on a refined grid
f = @(yy) evalD(l0,s0+1i*yy,j,b);
yt = unique([0; y(:)]);
% start grid: the targets plus a sinh-spaced grid over their range
ym = max(abs(yt));
if ym > 0
    u = (-asinh(ym/1e-3):0.05:asinh(ym/1e-3))'; yg = 1e-3*sinh(u);
    yg = yg(yg >= min(yt) & yg <= max(yt));
    yt = unique([yt; yg]);
end
lv = f(yt);
for it = 1:40
    d = angle(exp(1i*(imag(lv(2:end)) - imag(lv(1:end-1)))));
    bad = find(abs(d) > pi/4);
    if isempty(bad), break, end
    ym2 = (yt(bad) + yt(bad+1))/2;
    yt = [yt; ym2]; lv = [lv; f(ym2)];
    [yt,ix] = sort(yt); lv = lv(ix);
end
d = angle(exp(1i*(imag(lv(2:end)) - imag(lv(1:end-1)))));
k0 = find(yt == 0,1);
ph0 = zeros(size(yt));
cs = [0; cumsum(d)];
ph0 = cs - cs(k0) + imag(lv(k0));                    % imag(lv(k0)) is 0 or pi
ok = real(exp(lv(k0))) > 0 && abs(imag(lv(k0))) < 1e-8;
[~,loc] = ismember(y(:),yt);
ph = ph0(loc).';
end

function lv = evalD(l0,s,j,b)
z = 2*s(:).' - b^2;
[Dt,ls] = dval(l0+0*z,z,[],j,b);
lv = (ls + log(Dt)).';
end

% =========================================================================
function [Ds,ls] = dval(lam,z,tau0,j,b)
% D(lambda, z) e^{-c}, scaled by exp(-ls) as in COEFS, at the points (lam, z)
% (rows of a common size).  tau0 = t1^2 - 2 i t2 (NaN or [] where it is to
% be formed from lambda and z).  On the explosive side (c > 0, |z| >= 4) the
% stable forms of DSTAB are used; elsewhere the polynomial in lambda.
z = z(:).'; lam = lam(:).' + zeros(1,numel(z));
[F,ls] = coefs(z,j,b);
Ds = F(1,:) + F(2,:).*lam + F(3,:).*lam.^2;
if b < 0
    c = -b; th = sqrt(-z);
    st = abs(z) >= 4 & real(th + c - lam) >= c/2;
    if any(st)
        if isempty(tau0), t0 = NaN(1,numel(z)); else, t0 = tau0(:).' + zeros(1,numel(z)); end
        Ds(st) = dstab(lam(st),z(st),t0(st),j,c);
    end
end
end

function Ds = dstab(lam,z,tau0,j,c)
% D(lambda, z) e^{-c - theta}, theta = (-z)^(1/2), for c > 0, without
% cancellation.  On the explosive side the characteristic function varies on
% the scale t1 = O(e^{-c}), t2 = O(e^{-2c}), where D e^{-c-theta} is
% O(e^{-2c}) but the terms of the polynomial in lambda are O(e^{-c}) and
% cancel.  With omega = c - lambda (= c + i t1), X = (z + c^2)/2 = i(t2 - c t1)
% and tau0 = theta^2 - omega^2 = t1^2 - 2 i t2, the differences theta - c =
% -2X/(theta + c) and theta - omega = tau0/(theta + omega) are formed as
% quotients, and the polynomials are rearranged so that every term is a
% product of small factors (the same rearrangement as the limits of the AR(1)
% programs, Online Supplement of Bao and Kan, Section S7.5.3, for case DT).
% With E = e^{-theta}:
%   DK:  [(theta - omega) + E^2 (theta + omega)]/(2 theta);
%   DU:  [P0 + P1 E + P2 E^2]/theta^4, with lambda2 = lambda^2 - c^2 lambda,
%        P0 = (nu/2)[(theta - 2c)(theta + c) + c^2 theta - lambda theta] + lambda^2,
%        P1 = -4cX - 2 lambda2,
%        P2 = theta(2X + c^3)/2 + c^2 theta^2/2 + 2cX + lambda2 (1 + theta/2),
%        nu = theta - omega;
%   DT:  D3/(2 theta^8), D3 the cancellation-free form of the AR(1) programs
%        (Bao and Kan, Online Supplement, Section S7.5.3).
X = (z + c^2)/2; th = sqrt(-z); E1 = exp(-th); E2 = E1.^2;
nt = isnan(tau0); tau0(nt) = -lam(nt).^2 - 2*X(nt) + 2*c*lam(nt);
om = c - lam; nu = tau0./(th + om);
switch j
    case 1
        Ds = (nu + E2.*(th + om))./(2*th);
    case 2
        L2 = lam.^2 - c^2*lam;
        P0 = nu/2.*((th - 2*c).*(th + c) + c^2*th - lam.*th) + lam.^2;
        P1 = -4*c*X - 2*L2;
        P2 = th.*(2*X + c^3)/2 + c^2*th.^2/2 + 2*c*X + L2.*(1 + th/2);
        Ds = (P0 + P1.*E1 + P2.*E2)./th.^4;
    case 4
        zc = 2*X; vmc = -zc./(th + c); vpc = th + c; v2 = th.^2;
        Bm = bpoly(-vmc,c,-1);
        B1 = (c-3)*(c^2-4*c+6)*v2 - 6*(c^3-5*c^2+12*c-12);
        h1 = (th - 2) + E1.*(th + 2); h2 = -(v2 - 6*th + 12) + E1.*(v2 + 6*th + 12);
        K1 = c*(c-2)*(c^2-6*c+12);
        Pp = th.^3 + 8*v2 + 24*th + 24; Qc = (c-3)*th - 3*(c-2);
        D3 = 8*(vmc.*Qc).^2 + K1*vmc.*(8*(v2-6).*E1 + Pp.*E2) ...
             - tau0.*h1.*h2.*(K1./(th + om) + 4*(c^2-3*c+3)) + 8*zc.*B1.*E1 - vpc.*Bm.*E2;
        Ds = D3./(2*th.^8);
end
end

function B = bpoly(u,c,sg)
% B0(theta) expanded about theta = c (sg = 1, u = theta - c) or theta = -c
% (sg = -1, u = theta + c)
c4 = c^4-8*c^3+32*c^2-72*c+72;
if sg > 0
    b0 = c*(c-2)^2*(c^2-6*c+12)^2;
    b1 = 3*c^6-40*c^5+232*c^4-744*c^3+1344*c^2-1152*c+288;
    b2 = 3*c^5-32*c^4+152*c^3-408*c^2+576*c-288;
else
    b0 = -c^7;
    b1 = 3*c^6-8*c^5+8*c^4+24*c^3-96*c^2+288;
    b2 = -3*c^5+16*c^4-40*c^3+24*c^2+144*c-288;
end
B = ((c4*u + b2).*u + b1).*u + b0;
end

% =========================================================================
function [F,ls] = coefs(z,j,b)
% F (3 x numel(z)): F0, F1, F2 of D e^{-c} = F0 + F1 lambda + F2 lambda^2, scaled
% by exp(-ls); ls = 0 (series, |z| < 4) or theta = (-z)^(1/2)
z = z(:).'; F = complex(zeros(3,numel(z))); ls = complex(zeros(1,numel(z)));
sm = abs(z) < 4;
if any(sm)
    A = ntcoef(j,b); zz = z(sm);
    for m = 1:min(3,size(A,2))
        p = zeros(size(zz));
        for k = size(A,1):-1:1, p = p.*zz + A(k,m); end
        F(m,sm) = p;
    end
end
bg = ~sm;
if any(bg)
    zz = z(bg); X = (zz + b^2)/2; th = sqrt(-zz); E = exp(-2*th);
    C = (1+E)/2; S = (1-E)./(2*th); one = exp(-th);   % exp(-theta) cos, sin/theta, 1
    ls(bg) = th;
    switch j
        case 1
            F(1,bg) = C + b*S; F(2,bg) = S;
        case 2
            F(3,bg) = S./zz + 2*(C-one)./zz.^2;
            F(2,bg) = -b^2*S./zz - 2*b^2*(C-one)./zz.^2;
            F(1,bg) = (2*X - b^3).*S./zz - b^2*C./zz - 4*b*X.*(C-one)./zz.^2;
        case 4
            T = (S + (C-one)./zz)./zz.^3;
            F(3,bg) = -4*(b^2+3*b+27)*S./zz.^2 + (192*X + 288*(b+1)).*T ...
                      - 32*(b^2+3*b+6)*C./zz.^3 - 16*(b^2+3*b-3)*one./zz.^3;
            F(2,bg) = b^4*S./zz.^2 - 24*b^4*T + 8*b^4*C./zz.^3 + 4*b^4*one./zz.^3;
            F(1,bg) = (b^5 - 8*(b^2-3*b-3)*X).*S./zz.^2 - 96*(b+1)*X.^2.*T ...
                      + (b^4./zz.^2 + 16*b^3*X./zz.^3).*C + 8*b^2*(b+3)*X.*one./zz.^3;
    end
    if b < 0
        % c > 0: F0 = D(0, z) e^{-c-theta} cancels to O(e^{-2c}) near z = -c^2
        % (at the origin it is e^{-2c}); the stable form of DSTAB
        F(1,bg) = dstab(zeros(size(zz)),zz,NaN(size(zz)),j,-b);
    end
end
end

function A = ntcoef(j,b)
% coefficients of z^k lambda^m (m = 0, 1, 2) at the given c, from NT_SERIES
persistent T
if isempty(T), T = {nt_series(1),nt_series(2),[],nt_series(4)}; end
B = T{j}; Q = size(B,3);
cp = reshape(b.^(0:Q-1),1,1,Q);
A = sum(B.*cp,3);
if size(A,2) < 3, A(:,end+1:3) = 0; end
end
