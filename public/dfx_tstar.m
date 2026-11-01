function ts = dfx_tstar(model, c, start)
%DFX_TSTAR  t_* = sup{t : E exp(tV) < inf}, the first zero of D(0,t) (port of DFExact.tstar).
%   ts = DFX_TSTAR(model, c, start), start 'zero' (default) or 'stationary'.
if nargin < 2, c = 0; end
if nargin < 3, start = 'zero'; end
ts = dfx_core('tstar', model, c, strcmpi(start, 'stationary'));
end
