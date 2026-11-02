function varargout = dfx_core(action, varargin)
%DFX_CORE  Engine of the DFX routines, on the C engine of the repository df_limiting (MEX file LTRATIO_MEX).
%
%   A drop-in replacement for the MATLAB engine of the same name: the single integral of Theorem 2 (the general kernel
%   when c0 ~= 0), the moments and t_*, computed in C on several threads.  Not called directly; use DFX_CDF, DFX_PDF,
%   DFX_MOMENTS and DFX_TSTAR, or LTRATIO_CDF, LTRATIO_PDF, LTRATIO_ICDF and LTRATIO_MOM.
%
%   Actions:
%     dfx_core('cdf', q, model, c, opts), dfx_core('pdf', q, model, c, opts)
%     dfx_core('tstar', model, c, stat), dfx_core('moments', model, c, opts)
%     dfx_core('nodefac', k)      multiplies every node count by k (k = 2: convergence check)
%     dfx_core('threads', n)      the number of threads (sets it with n)
%   opts: start ('zero' or 'stationary'), c0 (0), gen (false; the general kernel also at c0 = 0), tau, hu, N, pfac,
%   edge_hw, uniform (hw > 0: the trapezoidal rule in w for checking); for 'moments' start, c0, g0, pmax (4).
%   The testing actions of the MATLAB engine (k0x, count, tune, D) are not available.
%
%   See also LTRATIO_MEX, TR_LIMDIST, TR_LIMMOM, LTRATIO.
persistent nodefac
if isempty(nodefac), nodefac = 1; end
switch lower(action)
    case {'cdf','pdf'}
        q = varargin{1}; model = varargin{2};
        if ~(isnumeric(model) && isscalar(model) && any(model == [1 2 3]))
            error('dfx: model must be 1, 2 or 3 (no deterministic terms, intercept, intercept and trend)');
        end
        c = 0; if numel(varargin) >= 3 && ~isempty(varargin{3}), c = varargin{3}; end
        o = struct(); if numel(varargin) >= 4 && ~isempty(varargin{4}), o = varargin{4}; end
        stat = startflag(getf(o,'start','zero'));
        c0 = getf(o,'c0',0);
        sopt = [getf(o,'tau',NaN) getf(o,'hu',NaN) getf(o,'N',NaN) nodefac getf(o,'pfac',NaN) getf(o,'edge_hw',NaN) getf(o,'uniform',NaN) getf(o,'gen',NaN)];
        kind = 2; if strcmpi(action,'pdf'), kind = 1; end
        varargout{1} = ltratio_mex('single', kind, double(q), model, double(c), double(c0), stat, double(sopt));
    case 'tstar'
        stat = false; if numel(varargin) >= 3, stat = startflag(varargin{3}); end
        varargout{1} = ltratio_mex('tstar', varargin{1}, double(varargin{2}), stat);
    case 'moments'
        model = varargin{1}; c = 0; if numel(varargin) >= 2 && ~isempty(varargin{2}), c = varargin{2}; end
        o = struct(); if numel(varargin) >= 3 && ~isempty(varargin{3}), o = varargin{3}; end
        stat = startflag(getf(o,'start','zero'));
        varargout{1} = ltratio_mex('mom', getf(o,'pmax',4), model, double(c), double(getf(o,'c0',0)), double(getf(o,'g0',0)), stat);
    case 'nodefac'
        nodefac = varargin{1};
    case 'threads'
        if isempty(varargin), varargout{1} = ltratio_mex('threads'); else, varargout{1} = ltratio_mex('threads', varargin{1}); end
    otherwise
        error('dfx_core: the action %s is not available in the C engine', action);
end
end
function v = getf(o, name, default)
if isstruct(o) && isfield(o,name) && ~isempty(o.(name)), v = o.(name); if isnumeric(v) || islogical(v), v = double(v); end, else, v = default; end
end
function stat = startflag(s)
if ischar(s) || isstring(s), stat = double(strncmpi(char(s),'stat',4)); else, stat = double(logical(s)); end
end
