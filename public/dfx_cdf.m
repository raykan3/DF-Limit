function F = dfx_cdf(q, model, c, opts)
%DFX_CDF  Limiting distribution function of the Dickey-Fuller t-ratio (port of DFExact.cdf).
%   F = DFX_CDF(q, model, c, opts): P(tau <= q) for the limit of the t-ratio under phi = 1 + c/n,
%   model 1, 2, 3 (no deterministic terms, intercept, intercept and trend), zero drift.
%   opts.start = 'zero' (default; cases DK, DU, DT) or 'stationary' (cases SK, SU, ST, c < 0);
%   opts.c0 = 0 (default): the local null value, the t-ratio for phi0 = 1 + c0/n, whose limit is
%   (U - c0 V)/sqrt(V) (the kernel of the single integral then runs along a parabola).
%   opts.tau: the line Re t = tau (default 1 for c <= 0 with a zero start, min(1, t_*/2) otherwise).
%   Example: dfx_cdf(-2.86, 2)    % P(tau_mu <= -2.86) at the unit root
%   See also DFX_PDF, DFX_QUANTILE, DFX_MOMENTS, DFX_TSTAR.
if nargin < 3, c = 0; end
if nargin < 4, opts = struct(); end
F = dfx_core('cdf', q, model, c, opts);
end
