function varargout = ltratio(what,varargin)
%LTRATIO  The limiting Dickey-Fuller t-ratio distributions from the C engine (MEX file LTRATIO_MEX), self-contained.
%
%   F = LTRATIO('cdf',x,cas,c,opts)    P[t <= x]
%   f = LTRATIO('pdf',x,cas,c,opts)    density
%   q = LTRATIO('icdf',p,cas,c,opts)   quantiles
%   m = LTRATIO('mom',p,cas,c,opts)    moments 1..p
%   n = LTRATIO('threads'[,n])         the number of threads (sets it with n)
%
%   The arguments are those of LTRATIO_CDF, LTRATIO_PDF, LTRATIO_ICDF and LTRATIO_MOM of the MATLAB programs: cas is
%   'DK', 'DU', 'DT', 'SK', 'SU' or 'ST', c defaults to 0, opts is a structure with the fields c0 (0), g0 (0), method
%   ('auto', the default: the faster representation at each x, the single integral at c0 = 0 and the double integral
%   otherwise, and the single integral with a fixed start; 'single'; 'double') and, for 'icdf', tol (1e-10).  Nothing but
%   this file and the MEX file is needed; with the MATLAB programs on the path too, DFX_CORE, TR_LIMDIST and
%   TR_LIMMOM next to this file replace their engines, so that LTRATIO_CDF and the others run on the C engine.
%
%   Examples
%      ltratio('icdf', [0.01 0.05 0.10], 'DU')              % asymptotic critical values of tau_mu
%      ltratio('cdf', -2.86157, 'DU', -5)                   % limiting power of the 5% test at c = -5
%      ltratio('cdf', -2, 'DU', -5, struct('g0', 1.5))      % fixed start
%      ltratio('pdf', [-1 0 1], 'DK', 2, struct('c0', 2))   % t-ratio centered at the true value
%      ltratio('mom', 4, 'SU', -5)                          % moments
if strcmpi(what,'threads'), varargout{1} = ltratio_mex('threads', varargin{:}); return, end
if nargin < 3, error('ltratio:args','usage: ltratio(what, x, cas, c, opts)'); end
x = varargin{1}; cas = upper(char(varargin{2}));
if ~(numel(cas) == 2 && any(cas(1) == 'DS') && any(cas(2) == 'KUT')), error('ltratio:args','the case must be DK, DU, DT, SK, SU or ST.'); end
model = find('KUT' == cas(2)); stat = double(cas(1) == 'S');
c = 0; if nargin > 3 && ~isempty(varargin{3}), c = varargin{3}; end
opts = struct; if nargin > 4 && ~isempty(varargin{4}), opts = varargin{4}; end
c0 = 0; if isfield(opts,'c0') && ~isempty(opts.c0), c0 = opts.c0; end
g0 = 0; if isfield(opts,'g0') && ~isempty(opts.g0), g0 = opts.g0; end
method = 0;
if isfield(opts,'method') && ~isempty(opts.method)
    switch lower(char(opts.method))
        case 'auto', method = 0;
        case 'single', method = 1;
        case 'double', method = 2;
        otherwise, error('ltratio:args','opts.method must be ''auto'', ''single'' or ''double''.');
    end
end
switch lower(what)
    case 'cdf'
        varargout{1} = ltratio_mex('dist', 2, double(x), model, double(c), double(c0), double(g0), stat, method);
    case 'pdf'
        varargout{1} = ltratio_mex('dist', 1, double(x), model, double(c), double(c0), double(g0), stat, method);
    case 'icdf'
        tol = 1e-10; if isfield(opts,'tol') && ~isempty(opts.tol), tol = opts.tol; end
        [q,Fq,it,conv] = ltratio_mex('icdf', double(x), model, double(c), double(c0), double(g0), stat, method, double(tol));
        varargout{1} = q; if nargout > 1, varargout{2} = struct('F',Fq,'iter',it,'conv',logical(conv)); end
        if ~all(conv(:)), warning('ltratio:icdf','ltratio: Newton''s method did not converge at %d of %d points.',nnz(~conv),numel(conv)); end
    case 'mom'
        varargout{1} = ltratio_mex('mom', double(x), model, double(c), double(c0), double(g0), stat);
    otherwise
        error('ltratio:args','what must be cdf, pdf, icdf, mom or threads.');
end
end
