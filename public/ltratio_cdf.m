function [F,info] = ltratio_cdf(x,cas,c,opts)
%LTRATIO_CDF  Limiting distribution function of the Dickey-Fuller t-ratio.
%
%   F = LTRATIO_CDF(x,cas,c,opts) is the limit as n -> Inf of P[t <= x] for
%   the least squares t-ratio t of the hypothesis phi = phi0 in the Gaussian
%   first-order autoregression y_t = alpha + delta t + phi y_{t-1} + sigma e_t
%   under the local alternatives phi = 1 + c/n, phi0 = 1 + c0/n (opts.c0,
%   default 0: the Dickey-Fuller tests of phi = 1), no drift (alpha = 0,
%   delta = 0).  cas is the case: the initial observation, Deterministic
%   (fixed) or Stationary, and the deterministic terms of the regression,
%   intercept Known to be zero, Unknown intercept, or intercept and Trend:
%      'DK'  y_0 fixed, regression of y_t on y_{t-1}         (tau)
%      'DU'  y_0 fixed, regression on 1, y_{t-1}               (tau_mu)
%      'DT'  y_0 fixed, regression on 1, t, y_{t-1}            (tau_tau)
%      'SK', 'SU', 'ST'  the same regressions with y_0 drawn from the
%            stationary distribution (c < 0)
%   With a fixed start y_0 = g0 sigma sqrt(n) (opts.g0, default 0), the start
%   enters the limit through the Ornstein-Uhlenbeck process started at g0.
%   The case 'DG' is the GLS-detrended trend case of Elliott, Rothenberg and
%   Stock (1996), detrended at cbar = -13.5 (the DF-GLS tests; Section 7 of
%   the paper): zero start, the line integral representation only, no
%   moments.  c defaults to 0 (the unit root).  x may be an array.
%
%   The limit is tau = (U - c0 V)/sqrt(V), with U = int_0^1 J~ dJ and
%   V = int_0^1 J~^2 dr, J the Ornstein-Uhlenbeck process dJ = cJ dr + dW
%   and J~ its residual on the deterministic terms of the regression (the
%   paper, Section 2).  Two exact representations are implemented:
%      'single'  the line integral of Theorem 2 (or 'line'): the inversion in
%                the argument of U is explicit (a scaled chi-square(1) density
%                in case DK, a Bessel K_0 density otherwise) and one integral
%                along Re t = tau remains (engine DFX_CORE);
%      'double'  the conditional integral of Theorem S1 of the
%                Supplementary Material (or 'conditional'): the
%                inner inversion in the argument of V along a path through a
%                saddle point, the outer integral over V = b (engine TR_LIMDIST).
%   opts.method selects it: 'auto' (the default) takes, point by point, the
%   faster of the two as measured in these programs (LTRATIO_METHOD: the
%   line integral for c0 = 0; with c0 ~= 0 the conditional integral except for
%   c0 > 0 at x >= 0 in cases DK, DU, DT, SK, and the line integral except
%   for c0 < 0 at x >= 0 in cases SU, ST), and recomputes the tails of the
%   line integral by the conditional integral (LTRATIO_DIST); with a fixed
%   nonzero start (g0 ~= 0) only the conditional integral is available in these
%   programs (the C and Julia implementations also carry the noncentral
%   kernel of the line integral), and it is used whatever opts.method
%   says.  The two agree to about 1e-12
%   (LTRATIO_TEST).
%
%   Further options are passed to the engine: for 'single' those of DFX_CDF
%   (tau, hu, N, ...), for 'double' those of TR_LIMDIST (M, Nv, otol, ...).
%   info: method, time, and the engine's diagnostics (for 'double' the
%   status and error estimates of TR_LIMDIST, where a nonzero status raises
%   a warning; with 'auto' a point whose status is nonzero gets the line
%   integral's value instead, info.nfallback counting them).
%
%   Examples
%      ltratio_cdf(-2.86, 'DU')                       % P[tau_mu <= -2.86] at the unit root
%      ltratio_cdf(-1.94087, 'DK', -5)                % power of the 5% test of tau at c = -5
%      ltratio_cdf(-3.41, 'ST', -10)                  % trend case, stationary start
%      ltratio_cdf(0, 'DK', 2, struct('c0',2))        % t-ratio centered at the true value
%      ltratio_cdf(-2, 'DU', -5, struct('g0',1.5))    % fixed start of 1.5 sigma sqrt(n)
%
%   See also LTRATIO_PDF, LTRATIO_ICDF, LTRATIO_MOM, LTRATIO_TEST, DFX_CDF, TR_LIMDIST.
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
[F,info] = ltratio_dist(x,cas,c,opts,'cdf');
end
