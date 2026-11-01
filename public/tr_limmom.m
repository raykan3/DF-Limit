function [m,info] = tr_limmom(p,cas,c,opts)
%TR_LIMMOM  Moments of the limiting distribution of the least squares t-ratio.
%
%   [m,info] = TR_LIMMOM(p,cas,c,opts): m(j) = E[t^j], j = 1..p, for the
%   local-to-unity limit of the t-ratio under phi = 1 + c/n,
%      t => A0/sqrt(B),   A0 = A + (c - c0) B,
%   with A, B the functionals of TR_CFLIM, c0 the local null value (opts.c0,
%   default c: the t-ratio centered at the true phi; c0 = 0: the Dickey-Fuller
%   test of phi = 1).  Cases 'DK', 'DU', 'DT' with a fixed start
%   J(0) = g0 = y0/(sigma sqrt(n)) (opts.g0, default 0), and 'SK', 'SU', 'ST'
%   with a stationary start J(0) ~ N(0, -1/(2c)) (c < 0).
%
%   Method.  By the gamma integral B^(-j/2) = Gamma(j/2)^(-1) int_0^Inf
%   s^(j/2-1) exp(-s B) ds,
%      E[(A0/sqrt(B))^j] = (1/Gamma(j/2)) int_0^Inf s^(j/2-1) M_j(s) ds,
%      M_j(s) = d^j/du^j E[exp(u A0 - s B)] at u = 0,
%   where the moment generating function is in closed form (the Fredholm
%   determinant of Nabeya and Tanaka, TR_CFLIM, form 'AB'; with a start, times
%   exp(g0^2 q(u,s)), or (1 - 2 s0^2 q)^(-1/2) for the stationary start, q
%   in closed form, TR_NSSTART).  It is
%   used only near the real axis, where it is smooth and positive.  The
%   derivatives in u are Cauchy integrals on a circle of radius r(s) inside
%   the domain of analyticity (trapezoidal rule, Nu nodes), with log D
%   continued along the circle from the real point u = r (a circle on which
%   it does not close encloses a zero of D, and its radius is halved); the integral over
%   s is the trapezoidal rule in log s, with an error estimate from the rule
%   with half the step.
%   With a zero start the moment generating function is evaluated from the
%   closed forms of TR_LIMDIST (Girsanov's theorem): log E exp(u R - u/2 + v B)
%   = -u/2 - c/2 - [log(sinh th/th) + log(th coth th - c - u)]/2 in case DK,
%   -u/2 - c/2 - [log cosh th + log det(I - (Q + u K) Sig)]/2 in cases DU, DT,
%   th = (c^2 - 2v)^(1/2), free of the cancellation of terms of order e^c in
%   the general form for c >> 0.  For c > 0 the radius r(s) is also at least
%   0.4 times the distance to the nearest real zero of the argument of the
%   logarithm, which for c0 = c >> 0 lies far beyond 1/(2 kap_1) (the terms of
%   order u of R and c0 B cancel), where a circle of radius 0.4/(2 kap_1)
%   leaves the derivatives below the rounding level.  Checked against graded
%   Gauss-Legendre integrals of the TR_LIMDIST densities: to 1.2e-10 (relative)
%   at c = c0 = 10, 2e-7 at c = c0 = 20.
%
%   opts: c0 (c), g0 (0), Nu (64) nodes on the circle, h (0.05) step in
%   log s, tlo (min(-60, -2c-40)) and thi (8) the range of log s, rfac (0.4).
%   info: time, err (estimated quadrature error of each moment).
%
%   See also TR_LIMDIST, TR_CFLIM, TR_NSSTART, LTRATIO_MOM.
if nargin < 4 || isempty(opts), opts = struct; end
cas = upper(cas); t0 = tic;
c0 = getf(opts,'c0',c); g0 = getf(opts,'g0',0);
Nu = getf(opts,'Nu',64); h = getf(opts,'h',0.05);
stA = cas(1) == 'S';
if stA
    if ~(c < 0), error('tr_limmom: a stationary start requires c < 0.'); end
    s02 = -1/(2*c); casD = ['D' cas(2)];
else
    casD = cas;
end
% radius of the circle in u at each s: the moment generating function of
% R = A' + 1/2 under the tilt exp(-s B) exists for u < 1/(2 kap_1(-s)); with
% c0 ~= 0 the argument of B moves by -c0 u, which must stay in the domain
smax = bdomain(casD,c);
% for c > 0 B is of order e^(2c) and the range of log s extends below -60 by 2c
tt = getf(opts,'tlo',min(-60,-2*c-40)):h:getf(opts,'thi',8); s = exp(tt);
zs = ~stA && g0 == 0;                               % zero start: closed forms (ZMGF)
if zs
    kmax = zkap(-s,c,casD);
else
    [~,~,kap] = tr_cflim(0,1i*s,casD,c,'B');       % weights at the real tilt -s
    kap = reshape(kap,2,[]); kmax = max(abs(kap),[],1);
end
r = 0.4./(2*kmax);
if c0 ~= 0, r = min(r,0.4*(s + smax)/abs(c0)); end
if zs && c > 0
    % with the closed forms the radius is set by the nearest zero of the
    % argument of the logarithm along the real u axis (for c0 = c >> 0 the
    % terms of order u of R and c0 B cancel and it lies far beyond 1/(2 kap_1),
    % where a circle of radius 0.4/(2 kap_1) leaves the derivatives below rounding)
    r = max(r,getf(opts,'rfac',0.4)*zrad(s,r,c,c0,casD));
end
if stA || g0 ~= 0, r = min(r,0.25); end             % the start term must stay analytic
% Cauchy integral: d^j f(0)/du^j = j!/(Nu r^j) sum_k f(r w_k) w_k^(-j).  log D
% (and the start factor) is continued along each circle from the real point
% u = r, where D > 0; a circle on which it does not return to its starting
% value encloses a zero of D, and its radius is halved
w = exp(2i*pi*(0:Nu-1)'/Nu);
S = repmat(s,Nu,1);
L = complex(zeros(Nu,numel(s))); todo = true(1,numel(s));
for tries = 1:8
    [Lq,okq] = logmgf(r(todo).*w,S(:,todo));
    iq = find(todo); L(:,iq(okq)) = Lq(:,okq);
    todo(iq(okq)) = false;
    if ~any(todo), break, end
    r(todo) = r(todo)/2;
end
if any(todo), error('tr_limmom: no circle in u free of zeros of D was found.'); end
f = exp(L);
m = zeros(1,p); err = m;
for j = 1:p
    Mj = factorial(j)./(Nu*r.^j).*real(sum(f.*w.^(-j),1));
    g = s.^(j/2).*Mj;                                % ds = s dt
    I = h*sum(g); I2 = 2*h*sum(g(1:2:end));
    m(j) = I/gamma(j/2); err(j) = abs(I - I2)/gamma(j/2);
end
info = struct('time',toc(t0),'err',err,'r',r);

    function [L,ok] = logmgf(U,S)
        % log E exp(u A0 - s B) = log E exp(u A' + v B), v = -s - c0 u
        %   = -u/2 - log D(-u, 2 v - c^2)/2   (zero start),
        % U, S: Nu x m, each column a circle starting at the real point u = r;
        % ok(j): the logarithms are continuous along column j
        V = -S - c0*U;
        if zs, [L,ok] = zmgf(U,V,c,casD); return, end
        [ls,~,F] = tr_cflim(0,-1i*V(:).',casD,c,'F');
        lam = -U(:).';
        Dt = F(1,:) + F(2,:).*lam + F(3,:).*lam.^2;
        [lDt,ok] = circlog(reshape(Dt,size(U)));
        L = reshape(-U(:).'/2 - (c + ls)/2,size(U)) - lDt/2;
        if stA || g0 ~= 0
            % start term: g0^2 q(u,s), q quadratic in the start (TR_NSSTART)
            q = tr_nsstart(U,V,casD,c);
            if stA
                [ld,ok2] = circlog(1 - 2*s02*q); ok = ok & ok2;
                L = L - ld/2;
            else
                L = L + g0^2*q;
            end
        end
    end
end

function [lz,ok] = circlog(Z)
% logarithm of Z continued down each column (a closed circle starting at a
% point where Z > 0); ok: the steps are resolved and the column closes
lz = log(Z);
d = angle(exp(1i*diff(imag(lz),1,1)));
ok = abs(imag(lz(1,:))) < 1e-8 & all(abs(d) < pi/3,1);
ph = imag(lz(1,:)) + [zeros(1,size(Z,2)); cumsum(d,1)];
lz = real(lz) + 1i*ph;
ok = ok & abs(ph(end,:) + angle(Z(1,:)./Z(end,:)) - ph(1,:)) < 1e-6;
end

function smax = bdomain(cas,c)
% right end of the real domain of E exp(s B) (zero start): first zero of D(0, 2s - c^2)
lo = 0; hi = 0.5;
while true
    [~,ok] = tr_cflim(0,-1i*hi,cas,c,'B');
    if ~ok || hi > 1e4, break, end
    lo = hi; hi = 2*hi;
end
for it = 1:60
    md = (lo + hi)/2; [~,ok] = tr_cflim(0,-1i*md,cas,c,'B');
    if ok, lo = md; else, hi = md; end
end
smax = lo;
end

function v = getf(o,f,d), if isfield(o,f), v = o.(f); else, v = d; end, end

function [L,ok] = zmgf(U,V,c,cas)
% zero start: log E exp(u R - u/2 + v B) by Girsanov's theorem (as in
% TR_LIMDIST), free of the cancellation that D(0, -c^2) = e^(-c)/cosh c
% suffers in the general form for c >> 0.  th = (c^2 - 2v)^(1/2).
%   DK: -u/2 - c/2 - [log(sinh th/th) + log(eta - u)]/2, eta = th coth th - c
%   DU, DT: -u/2 - c/2 - [log cosh th + log det(I - (Q + u K) Sig)]/2
[P,Sg,N,K] = zsig(V,c,cas);
if strcmp(cas,'DK')
    [lz,ok] = circlog(P.eta - U);
    L = -U/2 - c/2 - (P.lsh + lz)/2;
else
    m = size(K,1); M = cell(m);
    for i = 1:m, for j = 1:m
        ks = 0; for k = 1:m, ks = ks + K(i,k)*Sg{k,j}; end
        M{i,j} = N{i,j} - U.*ks;
    end, end
    if m == 2, d = M{1,1}.*M{2,2} - M{1,2}.*M{2,1}; else, d = det3c(M); end
    [lz,ok] = circlog(d);
    L = -U/2 - c/2 - (P.lch + lz)/2;
end
end

function km = zkap(v,c,cas)
% largest |weight| of R under the real tilt exp(v B) (zero start)
P = zsig(v,c,cas);
if strcmp(cas,'DK')
    km = 1./(2*abs(P.eta));
else
    S = sqrt(P.A1.^2 - 4*P.A0.*P.A2);
    km = max(abs((P.A1 + S)./(4*P.A2)),abs(2*P.A0./(2*(P.A1 + S))));   % nu/2, nu the roots
end
end

function [P,Sg,N,K] = zsig(v,c,cas)
% Brownian motion tilted by exp(-th^2/2 int W^2), th = (c^2 - 2v)^(1/2):
% Sig the covariance of (W(1), int W[, int r W]), N = I - Q Sig with Q =
% blkdiag(c, -2 v Gm) (N11 = (th - c + c(1 - tanh th))/th), K: R = X'KX/2;
% DK: eta = th coth th - c = (th - c) + 2 th e^(-2 th)/(1 - e^(-2 th)).
T = [1.0 -0.33333333333333333 0.13333333333333333 -0.053968253968253968 0.021869488536155203 -0.0088632355299021966 0.003592128036572481 -0.0014558343870513183 0.00059002744094558598 -0.00023912911424355248 9.6915379569294503e-5 -3.9278323883316834e-5 1.5918905069328965e-5 -6.4516892156554308e-6 2.6147711512907546e-6 -1.0597268320104654e-6 4.2949110782738059e-7 -1.7406618963571648e-7 7.0546369464009683e-8 -2.8591366623052539e-8 1.1587644432798852e-8 -4.6962953982309016e-9 1.9033368339312759e-9 -7.7139336353590623e-10 3.126339545892087e-10 -1.2670576930305401e-10 5.1351914080393677e-11 -2.0812146867700474e-11 8.4348454190943383e-12 -3.4185140868111558e-12 1.3854715742948469e-12 -5.6151047924146801e-13 2.2757162553728748e-13 -9.2231305851395317e-14 3.7379940310967389e-14 -1.5149519187148605e-14 6.1398688626168138e-15 -2.4883951222762789e-15 1.0085085566354096e-15 -4.0873312268690138e-16 1.6565329513786284e-16 -6.7136751750487117e-17 2.7209500613044588e-17 -1.1027595233722371e-17 4.4693160072374125e-18];
E = [1.0 -0.5 0.20833333333333333 -0.084722222222222222 0.034350198412698413 -0.013922233245149912 0.0056424968100315323 -0.0022868190951648293 0.00092681292737742184 -0.0003756231338525945 0.00015223432221797662 -6.1698246877700525e-5 2.5005357609459247e-5 -1.0134289721572026e-5 4.1072729198567004e-6 -1.664615015128028e-6 6.7464305456632013e-7 -2.7342253129896734e-7 1.1081397802278207e-7 -4.4911213669537136e-8 1.820182931124577e-8 -7.3769235610848765e-9 2.9897545073926764e-9 -1.2117018619561619e-9 4.9108426750009155e-10 -1.9902895700495881e-10 8.066339801136952e-11 -3.2691643852499818e-11 1.3249424201396147e-11 -5.3697893706595739e-12 2.1762936597810882e-12 -8.8201859824934e-13 3.5746867347671487e-13 -1.4487659644686842e-13 5.871627293628006e-14 -2.3796809091881837e-14 9.6444834564008509e-15 -3.9087619176859166e-15 1.5841615363041244e-15 -6.4203648775599251e-16 2.6020758752402585e-16 -1.0545816304260601e-16 4.2740583616893926e-17 -1.7322106086512011e-17 7.0203851674541611e-18 -2.8452549391662691e-18];
q = c^2 - 2*v; th = sqrt(q);
if c > 0, del = -2*v./(th + c); else, del = th - c; end
sm = abs(th) < 1; e2 = exp(-2*th);
tn = (1 - e2)./(1 + e2); sc = 2*exp(-th)./(1 + e2);
g1 = tn./th; g2 = (1 - sc)./th.^2; g3 = (th - tn)./th.^3; g4 = (th.^2 - 2 + 2*sc)./(2*th.^4); g5 = (th.^3/3 - th + tn)./th.^5;
t1 = 2*e2./(1 + e2); lch = th + log((1 + e2)/2);
ep = 2*th.*e2./(1 - e2); lsh = th + log((1 - e2)/2) - log(th);      % th coth th - th, log(sinh th/th)
if any(sm(:))
    Q = q(sm); hp = @(cf) polyval(fliplr(cf),Q);
    g1(sm) = hp(T); g2(sm) = -hp(E(2:end)); g3(sm) = -hp(T(2:end)); g4(sm) = hp(E(3:end)); g5(sm) = hp(T(3:end));
    t1(sm) = 1 - th(sm).*g1(sm); lch(sm) = log(cosh(th(sm)));
    r = ones(size(Q)); t = r; ch = r;
    for k = 1:14, t = t.*Q/((2*k)*(2*k+1)); r = r + t; end
    t = ones(size(Q)); for k = 1:14, t = t.*Q/((2*k-1)*(2*k)); ch = ch + t; end
    lsh(sm) = log(r); ep(sm) = ch./r - th(sm);
end
N11 = (del + c*t1)./th; N11(sm) = 1 - c*g1(sm);
P = struct('th',th,'eta',del + ep,'lsh',lsh,'lch',lch);
Sg = {}; N = {}; K = [];
if strcmp(cas,'DK'), return, end
if cas(2) == 'U'
    Sg = {g1, g2; g2, g3}; Qd = {c, 0; 0, -2*v}; K = [1 -1; -1 0];
else
    Sg = {g1, g2, g3; g2, g3, g4; g3, g4, g5}; Qd = {c,0,0; 0,-8*v,12*v; 0,12*v,-24*v};
    K = [1 2 -6; 2 -12 12; -6 12 0];
end
m = size(K,1); N = cell(m);
for i = 1:m, for j = 1:m
    acc = 0; for k = 1:m, acc = acc + Qd{i,k}.*Sg{k,j}; end
    N{i,j} = (i == j) - acc;
end, end
N{1,1} = N11;
if true
    % the equation A2 nu^2 - A1 nu + A0 = 0 of the weights (for ZKAP)
    if m == 2
        dS = g1.*g3 - g2.^2; D = N{1,1}.*N{2,2} - N{1,2}.*N{2,1};
        P.A2 = D./dS; P.A1 = (g1 - 2*g2)./dS + 2*v; P.A0 = -ones(size(v));
    else
        D = det3c(N); dS = det3c(Sg); Si = adj3c(Sg);
        Pp = cell(3); for i = 1:3, for j = 1:3, Pp{i,j} = Si{i,j}./dS - Qd{i,j}; end, end
        Pp{1,1} = (N{1,1}.*Si{1,1} + N{1,2}.*Si{2,1} + N{1,3}.*Si{3,1})./dS;
        aP = adj3c(Pp); aK = [-144 -72 -48; -72 -36 -24; -48 -24 -16];   % adj(K)
        A1 = 0; A0 = 0;
        for i = 1:3, for j = 1:3, A1 = A1 + K(i,j)*aP{j,i}; A0 = A0 + aK(i,j)*Pp{j,i}; end, end
        P.A2 = D./dS; P.A1 = A1; P.A0 = A0;
    end
end
end
function d = det3c(M)
d = M{1,1}.*(M{2,2}.*M{3,3}-M{2,3}.*M{3,2}) - M{1,2}.*(M{2,1}.*M{3,3}-M{2,3}.*M{3,1}) + M{1,3}.*(M{2,1}.*M{3,2}-M{2,2}.*M{3,1});
end
function A = adj3c(M)
A = cell(3);
A{1,1} = M{2,2}.*M{3,3}-M{2,3}.*M{3,2}; A{1,2} = -(M{1,2}.*M{3,3}-M{1,3}.*M{3,2}); A{1,3} = M{1,2}.*M{2,3}-M{1,3}.*M{2,2};
A{2,1} = -(M{2,1}.*M{3,3}-M{2,3}.*M{3,1}); A{2,2} = M{1,1}.*M{3,3}-M{1,3}.*M{3,1}; A{2,3} = -(M{1,1}.*M{2,3}-M{1,3}.*M{2,1});
A{3,1} = M{2,1}.*M{3,2}-M{2,2}.*M{3,1}; A{3,2} = -(M{1,1}.*M{3,2}-M{1,2}.*M{3,1}); A{3,3} = M{1,1}.*M{2,2}-M{1,2}.*M{2,1};
end

function R = zrad(s,r0,c,c0,cas)
% smallest |u| > 0 on the real axis at which the argument of the logarithm in
% ZMGF changes sign (or stops being finite), from u = +-r0 2^k, k = 0..60
R = inf(size(s)); ia = 1:numel(s);
for sg = [1 -1]
    lo = zeros(size(s)); hi = inf(size(s)); act = true(size(s)); u = sg*r0;
    for k = 0:80
        q = ia(act); if isempty(q), break, end
        d = zarg(u(q),-s(q) - c0*u(q),c,cas);
        bad = ~(d > 0) | ~isfinite(d);
        hi(q(bad)) = abs(u(q(bad))); act(q(bad)) = false;
        lo(q(~bad)) = abs(u(q(~bad))); u(q(~bad)) = 2*u(q(~bad));
    end
    for it = 1:30                                    % bisection between lo and hi
        q = ia(isfinite(hi)); if isempty(q), break, end
        m = (lo(q) + hi(q))/2; d = zarg(sg*m,-s(q) - c0*sg*m,c,cas);
        ok = d > 0 & isfinite(d); lo(q(ok)) = m(ok); hi(q(~ok)) = m(~ok);
    end
    R = min(R,lo);
end
R(~(R > 0)) = r0(~(R > 0))/0.4;
end
function d = zarg(u,v,c,cas)
% the argument of the logarithm in ZMGF at real u, v (real there)
[P,Sg,N,K] = zsig(v,c,cas);
if strcmp(cas,'DK'), d = real(P.eta - u); return, end
m = size(K,1); M = cell(m);
for i = 1:m, for j = 1:m
    ks = 0; for k = 1:m, ks = ks + K(i,k)*Sg{k,j}; end
    M{i,j} = N{i,j} - u.*ks;
end, end
if m == 2, d = M{1,1}.*M{2,2} - M{1,2}.*M{2,1}; else, d = det3c(M); end
d = real(d).*sign(real(exp(P.lch)));
end
