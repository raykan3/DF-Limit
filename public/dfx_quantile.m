function q = dfx_quantile(prob, model, c, opts)
%DFX_QUANTILE  Quantile of the limiting Dickey-Fuller t-ratio (port of DFExact.quantile):
%   Newton's method on DFX_CDF with DFX_PDF as derivative, safeguarded by bisection.
%   q = DFX_QUANTILE(prob, model, c, opts); opts as in DFX_CDF, plus q0 (start) and tol (1e-9).
if nargin < 3, c = 0; end
if nargin < 4, opts = struct(); end
q = zeros(size(prob));
for k = 1:numel(prob), q(k) = one(prob(k), model, c, opts); end
end

function q = one(prob, model, c, opts)
q0 = []; if isfield(opts, 'q0'), q0 = opts.q0; end
tol = 1e-9; if isfield(opts, 'tol'), tol = opts.tol; end
if isempty(q0), q0 = [-0.5, -1.6, -2.2]; q0 = q0(model); end
q = q0;
lo = q - 1; hi = q + 1;
while dfx_cdf(lo, model, c, opts) > prob, lo = lo - 2 * (q - lo); end
while dfx_cdf(hi, model, c, opts) < prob, hi = hi + 2 * (hi - q); end
for it = 1:60
    F = dfx_cdf(q, model, c, opts); f = dfx_pdf(q, model, c, opts);
    if F > prob, hi = q; else, lo = q; end
    qn = q - (F - prob) / f;
    if ~(lo < qn && qn < hi), qn = (lo + hi) / 2; end
    if abs(qn - q) < tol, q = qn; return; end
    q = qn;
end
error('dfx_quantile: no convergence');
end
