function [m,info] = ltratio_mom(p,cas,c,opts)
%LTRATIO_MOM  Moments of the limiting Dickey-Fuller t-ratio.
%
%   m = LTRATIO_MOM(p,cas,c,opts): m(j) = E[tau^j], j = 1..p, for the limit
%   tau = (U - c0 V)/sqrt(V) of the t-ratio under phi = 1 + c/n (arguments
%   and options c0, g0 as in LTRATIO_CDF).  By the gamma integral
%   V^(-j/2) = Gamma(j/2)^(-1) int_0^Inf s^(j/2-1) exp(-s V) ds the moments
%   are one-dimensional integrals of the joint moment generating function of
%   (U, V) at real arguments (Proposition 3 of the paper), where it is
%   smooth and positive; the derivatives in the argument of U are Cauchy
%   integrals.  Engine: TR_LIMMOM (the zero-start moment generating function
%   from the Girsanov closed forms, free of cancellation for c > 0; the
%   initial-value terms of a fixed or a stationary start in closed form,
%   TR_NSSTART).  DFX_MOMENTS is an independent implementation for c0 = 0
%   with a zero or a stationary start, used by LTRATIO_TEST as a check.
%   info: the error estimate of the integral over s per moment, and the time.
%
%   Example: m = ltratio_mom(4,'DU');  m(2)   % 3.0559953450 (Nabeya, 1999)
%
%   See also LTRATIO_CDF, LTRATIO_ICDF, TR_LIMMOM, DFX_MOMENTS.
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
[cas,c,c0,g0] = ltratio_args(cas,c,opts);
if strcmp(cas,'DG'), error('ltratio:args','ltratio_mom: the moments are not available for case DG.'); end
if ~(isnumeric(p) && isscalar(p) && p == round(p) && p >= 1), error('ltratio:args','ltratio_mom: p must be a positive integer.'); end
o = opts; if isfield(o,'method'), o = rmfield(o,'method'); end
o.c0 = c0; o.g0 = g0;
[m,info] = tr_limmom(p,cas,c,o);
end
