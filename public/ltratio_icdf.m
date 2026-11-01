function [q,info] = ltratio_icdf(p,cas,c,opts)
%LTRATIO_ICDF  Quantiles of the limiting Dickey-Fuller t-ratio.
%
%   q = LTRATIO_ICDF(p,cas,c,opts): the limit of the p-quantile of the
%   t-ratio, p in (0,1) (an array), by Newton's method on LTRATIO_CDF with
%   LTRATIO_PDF as the derivative, safeguarded by bisection, started at the
%   normal approximation from the first two moments (LTRATIO_MOM).  Arguments
%   and options as in LTRATIO_CDF; opts.tol (1e-10) is the tolerance on q.
%   info: F, the distribution function at q; iter; conv (logical).
%
%   Example: ltratio_icdf([0.01 0.05 0.10], 'DU')   % the asymptotic critical values of tau_mu
%
%   See also LTRATIO_CDF, LTRATIO_PDF, LTRATIO_MOM.
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
ltratio_args(cas,c,opts);
if ~(isnumeric(p) && isreal(p) && all(p(:) > 0 & p(:) < 1)), error('ltratio:args','ltratio_icdf: p must lie in (0,1).'); end
tol = 1e-10; if isfield(opts,'tol') && ~isempty(opts.tol), tol = opts.tol; o = rmfield(opts,'tol'); else, o = opts; end
m = ltratio_mom(2,cas,c,o); mu = m(1); sd = sqrt(max(m(2) - m(1)^2,1e-6));
q = zeros(size(p)); Fq = q; it = q; conv = false(size(p));
for k = 1:numel(p)
    pk = p(k);
    x = mu + sd*sqrt(2)*erfinv(2*pk - 1);
    lo = x - sd; hi = x + sd;
    while ltratio_cdf(lo,cas,c,o) > pk, lo = lo - 2*(x - lo); end
    while ltratio_cdf(hi,cas,c,o) < pk, hi = hi + 2*(hi - x); end
    for j = 1:60
        F = ltratio_cdf(x,cas,c,o); f = ltratio_pdf(x,cas,c,o);
        if F > pk, hi = x; else, lo = x; end
        xn = x - (F - pk)/f;
        if ~(lo < xn && xn < hi) || ~isfinite(xn), xn = (lo + hi)/2; end
        if abs(xn - x) < tol, x = xn; conv(k) = true; break; end
        x = xn;
    end
    q(k) = x; Fq(k) = ltratio_cdf(x,cas,c,o); it(k) = j;
end
info = struct('F',Fq,'iter',it,'conv',conv);
if ~all(conv(:)), warning('ltratio:icdf','ltratio_icdf: Newton''s method did not converge at %d of %d points.',nnz(~conv),numel(p)); end
end
