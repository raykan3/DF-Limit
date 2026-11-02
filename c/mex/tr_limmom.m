function [m,info] = tr_limmom(p,cas,c,opts)
%TR_LIMMOM  Moments of the limiting t-ratio (C engine, MEX file LTRATIO_MEX).
%
%   [m,info] = TR_LIMMOM(p,cas,c,opts): a drop-in replacement for the MATLAB engine of the same name: m(j) = E[t^j],
%   j = 1..p, for the limit (U - c0 V)/sqrt(V) under phi = 1 + c/n; cases 'DK', 'DU', 'DT' (fixed start opts.g0,
%   default 0) and 'SK', 'SU', 'ST' (stationary start, c < 0); opts.c0 defaults to c, as in the MATLAB engine
%   (LTRATIO_MOM passes its own c0).  info: time, err (the estimated quadrature error of each moment).  The quadrature
%   options of the MATLAB engine (Nu, h, tlo, thi, rfac) are fixed in the C engine.
%
%   See also LTRATIO_MEX, TR_LIMDIST, DFX_CORE, LTRATIO_MOM.
t0 = tic;
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
if ~(isnumeric(p) && isscalar(p) && p == round(p) && p >= 1), error('tr_limmom:args','tr_limmom: p must be a positive integer.'); end
cas = upper(char(cas));
if ~(numel(cas) == 2 && any(cas(1) == 'DS') && any(cas(2) == 'KUT')), error('tr_limmom:args','tr_limmom: the case must be DK, DU, DT, SK, SU or ST.'); end
model = find('KUT' == cas(2)); stat = double(cas(1) == 'S');
c0 = getf(opts,'c0',c); g0 = getf(opts,'g0',0);
[m,err] = ltratio_mex('mom', p, model, double(c), double(c0), double(g0), stat);
info = struct('time',toc(t0),'err',err);
end
function v = getf(o, name, default)
if isstruct(o) && isfield(o,name) && ~isempty(o.(name)), v = o.(name); if isnumeric(v) || islogical(v), v = double(v); end, else, v = default; end
end
