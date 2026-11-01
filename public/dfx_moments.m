function m = dfx_moments(model, c, opts)
%DFX_MOMENTS  Raw moments of the limiting Dickey-Fuller t-ratio (port of DFExact.moments).
%   m = DFX_MOMENTS(model, c, opts): E tau^p, p = 1..opts.pmax (default 4), from
%   E tau^p = Gamma(p/2)^(-1) int_0^inf s^(p/2-1) d^p/du^p M(u,-s)|_{u=0} ds (Proposition 2).
%   opts.start as in DFX_CDF; opts.h (0.05), opts.Nc (64), opts.rfrac (0.4).
if nargin < 2, c = 0; end
if nargin < 3, opts = struct(); end
m = dfx_core('moments', model, c, opts);
end
