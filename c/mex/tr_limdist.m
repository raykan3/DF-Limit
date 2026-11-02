function [F,info] = tr_limdist(x,cas,c,opts,mode)
%TR_LIMDIST  Limiting distribution and density of the t-ratio by the double integral (C engine, MEX file LTRATIO_MEX).
%
%   [F,info] = TR_LIMDIST(x,cas,c,opts,mode): a drop-in replacement for the MATLAB engine of the same name, with the
%   same arguments: mode 'cdf' (default) or 'pdf'; cases 'DK', 'DU', 'DT' (fixed start opts.g0, default 0) and 'SK',
%   'SU', 'ST' (stationary start, c < 0); opts.c0 the local null value (default c, as in the MATLAB engine: the t-ratio
%   centered at the true value; LTRATIO_CDF passes its own c0); the options M, Nv, Nl, maxlev, plen, otol, stepw,
%   errtol, blo, bhi, bmin, mumin, tdec of the double integral.  info: time, status, err, nbad, nunres, nrefine
%   (per point; a nonzero status raises a warning), and finite when x has non-finite entries (0 at -Inf, 1 at Inf for
%   the cdf, NaN at NaN).
%
%   See also LTRATIO_MEX, DFX_CORE, TR_LIMMOM, LTRATIO_CDF, LTRATIO_PDF.
t0 = tic;
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
if nargin < 5 || isempty(mode), mode = 'cdf'; end
if ~(isnumeric(c) && isscalar(c) && isreal(c) && isfinite(c)), error('tr_limdist:args','tr_limdist: c must be a finite real scalar.'); end
if ~(isnumeric(x) && isreal(x)), error('tr_limdist:args','tr_limdist: x must be real.'); end
cas = upper(char(cas));
if ~(numel(cas) == 2 && any(cas(1) == 'DS') && any(cas(2) == 'KUT')), error('tr_limdist:args','tr_limdist: the case must be DK, DU, DT, SK, SU or ST.'); end
kind = 2; if strcmpi(mode,'pdf'), kind = 1; end
model = find('KUT' == cas(2)); stat = double(cas(1) == 'S');
c0 = getf(opts,'c0',c); g0 = getf(opts,'g0',0);
dopt = [getf(opts,'M',NaN) getf(opts,'Nv',NaN) getf(opts,'Nl',NaN) getf(opts,'maxlev',NaN) getf(opts,'plen',NaN) getf(opts,'otol',NaN) ...
        getf(opts,'stepw',NaN) getf(opts,'errtol',NaN) getf(opts,'blo',NaN) getf(opts,'bhi',NaN) getf(opts,'bmin',NaN) getf(opts,'mumin',NaN) getf(opts,'tdec',NaN)];
F = zeros(size(x)); fin = isfinite(x);
F(isnan(x)) = NaN; if kind == 2, F(x == Inf) = 1; end
info = struct('time',0,'status',zeros(size(x)),'err',zeros(size(x)),'nbad',zeros(size(x)),'nunres',zeros(size(x)),'nrefine',zeros(size(x)));
if any(fin(:))
    [Fv,err,status,nbad,nunres,nrefine] = ltratio_mex('double', kind, double(x(fin)), model, double(c), double(c0), double(g0), stat, dopt);
    F(fin) = Fv; info.err(fin) = err; info.status(fin) = status; info.nbad(fin) = nbad; info.nunres(fin) = nunres; info.nrefine(fin) = nrefine;
end
if ~all(fin(:)), info.finite = fin; end
info.time = toc(t0);
if any(info.status(:))
    iw = find(info.status,1);
    warning('tr_limdist:status',['tr_limdist: %d of %d points with nonzero status (first: x = %.6g, ' ...
        'nbad %d, nunres %d; see info.status); the result may be inaccurate there.'], ...
        nnz(info.status),numel(x),x(iw),info.nbad(iw),info.nunres(iw));
end
end
function v = getf(o, name, default)
if isstruct(o) && isfield(o,name) && ~isempty(o.(name)), v = o.(name); if isnumeric(v) || islogical(v), v = double(v); end, else, v = default; end
end
