function nfail = ltratio_test(quick,strict)
%LTRATIO_TEST  Validation suite for the limiting t-ratio programs.
%
%   nfail = LTRATIO_TEST runs the checks below, prints each comparison with
%   its tolerance and a PASS/FAIL mark, and returns the number of failures.
%   LTRATIO_TEST(true) runs the quick version (fewer points; about a minute
%   in MATLAB), LTRATIO_TEST(quick,true) raises an error at the end when a
%   check has failed, for use in scripts.
%
%    1. the two representations against each other: the single integral
%       (DFX_CORE) and the double integral (TR_LIMDIST), distribution function
%       and density, zero and stationary starts, c on both sides of the unit
%       root and local null values c0 ~= 0 (the two share nothing but the
%       Fredholm determinant);
%    2. exact values: P[tau <= 0] = erf((2v)^(-1/2)) in case DK (c0 = 0),
%       P[tau_mu <= 0] from the distribution of the coefficient statistic
%       (Bao and Kan, 2026), and P[A' - c0 B <= 0] at c = 10, c0 = 5 from a
%       one-dimensional inversion; the 5% points against nine-digit values and
%       the asymptotic critical values of MacKinnon (2010) to their accuracy;
%       the quantile function LTRATIO_ICDF;
%    3. the initial observation: continuity at a zero start, the stationary
%       start as the Gauss-Hermite average of the fixed start (an identity the
%       programs do not use), and the fixed start of case DK against the
%       closed form of the tilted mean (TR_JBSTART);
%    4. moments: Nabeya (1999) at the unit root (eight digits), LTRATIO_MOM
%       against DFX_MOMENTS (independent implementations), and the mean
%       against a Gauss-Legendre integral of the density;
%    5. the density against central differences of the distribution function;
%    6. argument validation and non-finite x.
%
%   See also LTRATIO_CDF, LTRATIO_PDF, LTRATIO_ICDF, LTRATIO_MOM.
if nargin < 1 || isempty(quick), quick = false; end
if nargin < 2 || isempty(strict), strict = false; end
nfail = 0; ntest = 0; t00 = tic;
S = struct('method','single'); D = struct('method','double');

fprintf('\n=== 1. single integral (DFX) against double integral (TR_LIMDIST) ===\n');
if quick
    P = {'DK',0,0,[-2.5 -0.5 1.5]; 'DU',0,0,[-3.4 -1.6 0.2]; 'DT',-5,0,[-3.5 -2.5]; 'SU',-5,0,[-2.8 -1.2]; 'DK',2,2,[-1.5 0 1.5]};
else
    P = {'DK',0,0,[-3 -2.5 -1.94 -0.5 0.9 2]; 'DU',0,0,[-3.43 -2.86 -1.57 -0.08 0.6]; 'DT',0,0,[-3.96 -3.41 -2.18 -0.94];
         'DK',-5,0,[-3 -1.7 -0.5]; 'DU',-5,0,[-3 -2.1 -1.2]; 'DT',-10,0,[-3.5 -2.86 -2.2];
         'SK',-5,0,[-2.5 -1.8 -1]; 'SU',-5,0,[-3 -2.1 -1.2]; 'ST',-10,0,[-3.5 -2.9 -2.2];
         'DK',2,2,[-1.5 0 1.5 3]; 'DU',2,1,[-3 -0.5 1.3]; 'DT',-2.5,-5,[-3.6 -1.1 0.3]; 'DU',5,-5,[57.11 126.85]; 'DK',0,-5,[-1.1 2.7 6.5]};
end
for k = 1:size(P,1)
    cas = P{k,1}; c = P{k,2}; c0 = P{k,3}; xs = P{k,4};
    o = struct('c0',c0);
    [F1,i1] = ltratio_cdf(xs,cas,c,setfield(o,'method','single'));
    [F2,i2] = ltratio_cdf(xs,cas,c,setfield(o,'method','double'));
    f1 = ltratio_pdf(xs,cas,c,setfield(o,'method','single'));
    f2 = ltratio_pdf(xs,cas,c,setfield(o,'method','double'));
    check(sprintf('%s c=%g c0=%g: F single vs double (%d points; %.1f/%.1f s)',cas,c,c0,numel(xs),i1.time,i2.time),max(abs(F1-F2)),1e-11);
    check(sprintf('%s c=%g c0=%g: f single vs double (relative)',cas,c,c0),max(abs(f1-f2)./max(abs(f2),1e-3)),1e-10);
end

fprintf('\n=== 2. exact values, critical values and quantiles ===\n');
for cc = [-10 0 2 5 20]
    v = (exp(2*cc) - 1)/(2*cc); if cc == 0, v = 1; end
    F = ltratio_cdf(0,'DK',cc,S); G = ltratio_cdf(0,'DK',cc,D);
    check(sprintf('DK c=%g: F(0) = erf((2v)^(-1/2)), single',cc),abs(F-erf(1/sqrt(2*v))),1e-12);
    check(sprintf('DK c=%g: F(0) = erf((2v)^(-1/2)), double',cc),abs(G-erf(1/sqrt(2*v))),1e-12);
end
check('DU c=0: F(0) = 0.95756559 (Bao and Kan, 2026, eight digits)',abs(ltratio_cdf(0,'DU',0)-0.95756559),5e-9);
ref = [0.000302655553129 0.000414190811913]; cs = {'DK','DU','DT'};
for j = 2:3
    F = ltratio_cdf(0,cs{j},10,struct('c0',5));
    check(sprintf('%s c=10 c0=5: F(0) vs a one-dimensional inversion',cs{j}),abs(F-ref(j-1)),1e-11);
end
q5 = [-1.94087325 -2.86156886 -3.41045363];           % nine-digit values (independent route)
mck = [-2.56574 -1.94100 -1.61682; -3.43035 -2.86154 -2.56677; -3.95877 -3.41049 -3.12705];   % MacKinnon (2010), 1%, 5%, 10%
for j = 1:3
    F = ltratio_cdf(q5(j),cs{j});
    check(sprintf('%s: F at the 5%% point',cs{j}),abs(F-0.05),2e-9);
    Fm = ltratio_cdf(mck(j,:),cs{j});
    check(sprintf('%s: F at MacKinnon''s critical values vs 0.01, 0.05, 0.10',cs{j}),max(abs(Fm-[0.01 0.05 0.10])),1.5e-4);
    [q,inf1] = ltratio_icdf([0.05 0.5],cs{j});
    check(sprintf('%s: LTRATIO_ICDF(0.05) vs the nine-digit value',cs{j}),abs(q(1)-q5(j)),6e-9);
    check(sprintf('%s: F(LTRATIO_ICDF(p)) - p, convergence',cs{j}),max(abs(inf1.F-[0.05 0.5])) + ~all(inf1.conv),1e-9);
end
[q,inf1] = ltratio_icdf([0.01 0.05 0.5],'SU',-5);
check('SU c=-5: quantiles increasing and converged',double(~all(diff(q) > 0)) + ~all(inf1.conv),0.5);

fprintf('\n=== 3. the initial observation ===\n');
cc = -5; x0 = -2;
for cas = cs
    F0 = ltratio_cdf(x0,cas{1},cc); F1 = ltratio_cdf(x0,cas{1},cc,struct('g0',1e-6));
    check(sprintf('%s c=%g: start 1e-6 vs zero start',cas{1},cc),abs(F1-F0),1e-9);
end
[gz,gw] = ghnodes(24); s0 = sqrt(-1/(2*cc)); ip = find(gz > 0);   % (16 nodes leave 7e-11)
sts = {'SK','DK'; 'SU','DU'}; if ~quick, sts = [sts; {'ST','DT'}]; end
for j = 1:size(sts,1)
    Fs = ltratio_cdf(x0,sts{j,1},cc);
    Fm = 0;
    for k = ip(:).', Fm = Fm + 2*gw(k)*ltratio_cdf(x0,sts{j,2},cc,struct('g0',s0*gz(k))); end
    check(sprintf('%s c=%g: stationary start vs the Gauss-Hermite average of %s over J(0)',sts{j,1},cc,sts{j,2}),abs(Fs-Fm),2e-11);
end
% fixed start in case DK: U = (J(1)^2 - J(0)^2)/2 - 1/2, so P[tau <= 0] = P[J(1)^2 <= 1 + g0^2]
% with J(1) ~ N(g0 e^c, (e^(2c)-1)/(2c))
g0 = 1.5; v = (exp(2*cc) - 1)/(2*cc); m = g0*exp(cc); r = sqrt(1 + g0^2);
F = ltratio_cdf(0,'DK',cc,struct('g0',g0));
Pex = (erf((r - m)/sqrt(2*v)) + erf((r + m)/sqrt(2*v)))/2;
check(sprintf('DK c=%g g0=%g: F(0) = P[J(1)^2 <= 1 + g0^2]',cc,g0),abs(F-Pex),1e-11);
% fixed start in cases DU, DT: the closed-form initial-value term Q(lam,s) of TR_NSSTART
% is, as a function of lam, Q(0,s) + (lam/2) [m1^2/(1 - 2 lam kap_1) - m2^2/(1 - 2 lam kap_2)]
% with m_k^2 >= 0 at real s (a noncentral form of rank two): fit m1^2, m2^2 at two values
% of lam and predict a third
for cas = {'DU','DT'}
    s = -1; [~,~,kap] = tr_cflim(0,-1i*s,cas{1},cc,'B'); k1 = kap(1); k2 = kap(2);
    la = [0.1 0.2 0.3]; Q = tr_nsstart(la,s,cas{1},cc) - tr_nsstart(0,s,cas{1},cc);
    A = [la(1)/2/(1-2*la(1)*k1), -la(1)/2/(1-2*la(1)*k2); la(2)/2/(1-2*la(2)*k1), -la(2)/2/(1-2*la(2)*k2)];
    mm = A\Q(1:2).';
    pred = la(3)/2*(mm(1)/(1-2*la(3)*k1) - mm(2)/(1-2*la(3)*k2));
    check(sprintf('%s c=%g: TR_NSSTART is a rank-two noncentral form in lam (prediction)',cas{1},cc),abs(pred-Q(3)) + any(mm < 0) + max(abs(imag(mm))),1e-12);
end

fprintf('\n=== 4. moments ===\n');
nab = [-0.42309565 1.14159508 -1.06171218 3.52380298;
       -1.53296244 3.05599535 -6.71991749 16.34553684;
       -2.18135582 5.32067592 -14.03580010 39.55443050];   % Nabeya (1999, Table 4)
for j = 1:3
    m = ltratio_mom(4,cs{j});
    check(sprintf('%s: moments 1-4 vs Nabeya (1999)',cs{j}),max(abs(m-nab(j,:))),1e-7);
end
for cfg = {{'DU',0},{'DU',-5},{'SU',-5},{'DT',-10},{'SK',-2.5}}
    cas = cfg{1}{1}; cc = cfg{1}{2};
    m1 = ltratio_mom(4,cas,cc);
    model = find('KUT' == cas(2)); o = struct(); if cas(1) == 'S', o.start = 'stationary'; end
    m2 = dfx_moments(model,cc,o);
    check(sprintf('%s c=%g: LTRATIO_MOM vs DFX_MOMENTS (relative)',cas,cc),max(abs(m1(:)-m2(:))./abs(m2(:))),1e-9);
end
% the mean against a Gauss-Legendre integral of the density (DU, c = -5, c0 = 0; 192
% nodes: with 96 the rule itself is off by 8e-7, the single and the double integral
% agreeing on both)
m = ltratio_mom(1,'DU',-5,struct('c0',0));
[xg,wg] = glnodes(192,-12,6);
f = ltratio_pdf(xg,'DU',-5,struct('c0',0,'method','double'));
check('DU c=-5 c0=0: mean vs integral of x f(x) over [-12, 6]',abs(sum(wg.*xg.*f)-m),1e-9);
check('DU c=-5 c0=0: mass of the density over [-12, 6]',abs(sum(wg.*f)-1),1e-9);

fprintf('\n=== 5. density against the distribution function ===\n');
h = 1e-3;
for cfg = {{'DK',0,0,-1.5},{'DU',-5,0,-2.2},{'ST',-10,0,-3},{'DT',2,1,-2}}
    cas = cfg{1}{1}; cc = cfg{1}{2}; c0 = cfg{1}{3}; x = cfg{1}{4}; o = struct('c0',c0);
    f = ltratio_pdf(x,cas,cc,o);
    Fp = ltratio_cdf(x + [-2 -1 1 2]*h,cas,cc,o);
    fd = (Fp(1) - 8*Fp(2) + 8*Fp(3) - Fp(4))/(12*h);
    check(sprintf('%s c=%g c0=%g: f vs fifth-order difference of F at x=%g',cas,cc,c0,x),abs(f-fd),1e-9);
end

fprintf('\n=== 6. argument validation and non-finite x ===\n');
bad = {{-2,'XX',0,struct()},{-2,'SU',0,struct()},{-2,'DU','a',struct()},{-2,'DU',0,struct('c0',[1 2])}, ...
       {-2,'SU',-5,struct('g0',1)},{-2,'DU',0,struct('method','triple')},{[1 2]*1i,'DU',0,struct()}};
nerr = 0;
for k = 1:numel(bad)
    try, ltratio_cdf(bad{k}{:}); catch, nerr = nerr + 1; end
end
check('invalid arguments rejected',double(nerr ~= numel(bad)),0);
F = ltratio_cdf([-Inf Inf NaN],'DK',0);
check('F(-Inf), F(Inf), F(NaN) = 0, 1, NaN',abs(F(1)) + abs(F(2)-1) + ~isnan(F(3)),0);
f = ltratio_pdf([-Inf Inf NaN],'DU',0,D);
check('f(-Inf), f(Inf), f(NaN) = 0, 0, NaN (double)',abs(f(1)) + abs(f(2)) + ~isnan(f(3)),0);
f = ltratio_pdf(-sqrt(4),'DK',0,struct('c0',2));
check('DK c0=2: f(-sqrt(2 c0)) = Inf',double(~isinf(f)),0);

fprintf('\n=== 7. tails with c0 ~= 0 ===\n');
% a far root of X(v) = c0 v^2 + q v + 1/2 was dropped by BREAKPTS, which ran the near-side form
% past it (an error, 'Re alpha <= 0 on a ray', or a wrong value); the single integral's absolute
% error floor (1e-14 to 1e-21) is handled by the double integral below 1e-3 (LTRATIO_DIST)
nerr = 0; f = zeros(1,4);
try, f(1) = ltratio_pdf(9,'SU',-1.2,struct('c0',-1.2)); catch, nerr = nerr + 1; end
try, f(2) = ltratio_pdf(-9,'SU',-1.2,struct('c0',1.2)); catch, nerr = nerr + 1; end
try, f(3) = ltratio_pdf(3,'SU',-1.2,struct('c0',-0.05)); catch, nerr = nerr + 1; end
try, f(4) = ltratio_pdf(3,'SU',-1.2,struct('c0',-0.05,'method','single')); catch, nerr = nerr + 1; end
check('no error at far roots: SU c=-1.2, c0=-1.2 (x=9), c0=1.2 (x=-9), c0=-0.05 (x=3)',nerr,0);
g = ltratio_pdf([3 -9],'SU',-1.2,struct('c0',-0.05,'method','double'));
check('SU c=-1.2 c0=-0.05, x=3: single (far root kept) vs double (relative)',abs(f(4)-g(1))/g(1),1e-10);
g = ltratio_pdf(-9,'SU',-1.2,struct('c0',1.2,'method','double'));
check('SU c=-1.2 c0=1.2, x=-9 (f = 6.2e-16): default (tail by the double integral) vs double (rel)',abs(f(2)-g)/g,1e-10);
xs = [-9 -8 -6 3 4 5]; fd = ltratio_pdf(xs,'DT',0,struct('c0',-2,'method','double')); fn = ltratio_pdf(xs,'DT',0,struct('c0',-2));
check('DT c=0 c0=-2: default vs double in both tails, f from 4e-11 to 2.5e-6 (relative)',max(abs(fn-fd)./abs(fd)),1e-10);
fs = ltratio_pdf([8.5 9 10 12],'SU',-1.2,struct('c0',-1.2,'method','single'));    % noise below the floor
fs2 = ltratio_pdf([8.5 9 10 12],'SU',-1.2,struct('c0',-1.2,'method','double'));
check('no negative density (single and double, SU c=c0=-1.2, x = 8.5 to 12)',nnz([fs fs2] < 0),0);

fprintf('\n%d of %d checks failed (%.0f s).\n',nfail,ntest,toc(t00));
if strict && nfail > 0, error('ltratio_test:fail','%d check(s) failed.',nfail); end

    function check(name,err,tol)
        ntest = ntest + 1; pass = isfinite(err) && err <= tol;
        if ~pass, nfail = nfail + 1; end
        mk = {'FAIL','pass'};
        fprintf('  %-72s %9.2e  (tol %7.1e)  %s\n',name,err,tol,mk{pass+1});
    end
end

% ---------------------------------------------------------------------------
function [x,w] = ghnodes(m)
% Gauss-Hermite rule for the standard normal: E f(Z) ~ sum w f(x)
b = sqrt((1:m-1)); [V,D] = eig(diag(b,1)+diag(b,-1));
[x,ix] = sort(diag(D)); w = V(1,ix).^2; w = w(:);
end

function [x,w] = glnodes(n,a,b)
% Gauss-Legendre rule on [a, b]
k = (1:n-1)'; be = k./sqrt(4*k.^2 - 1);
[V,E] = eig(diag(be,1) + diag(be,-1));
[x,ix] = sort(diag(E)); w = 2*V(1,ix)'.^2;
x = a + (b - a)*(x + 1)/2; w = w*(b - a)/2;
end
