function [cas,c,c0,g0,method] = ltratio_args(cas,c,opts)
%LTRATIO_ARGS  Validation of the arguments of the LTRATIO functions.
%
%   [cas,c,c0,g0,method] = LTRATIO_ARGS(cas,c,opts) checks the case ('DK',
%   'DU', 'DT', 'SK', 'SU', 'ST'; upper case returned), the local-to-unity
%   parameter c (a finite real scalar; a stationary start requires c < 0),
%   and the options c0 (the local null value, default 0), g0 (the fixed start
%   y_0 = g0 sigma sqrt(n), default 0; fixed-start cases only) and method
%   ('auto' by default: the faster representation point by point, see
%   LTRATIO_METHOD; 'single'; 'double'; a fixed start g0 ~= 0 forces 'double').
%
%   See also LTRATIO_CDF, LTRATIO_PDF, LTRATIO_ICDF, LTRATIO_MOM.
if ~(ischar(cas) || isstring(cas)), error('ltratio:args','ltratio: the case must be one of DK, DU, DT, SK, SU, ST.'); end
cas = upper(char(cas));
if ~any(strcmp(cas,{'DK','DU','DT','SK','SU','ST'}))
    error('ltratio:args','ltratio: the case must be one of DK, DU, DT, SK, SU, ST.');
end
if ~(isnumeric(c) && isscalar(c) && isreal(c) && isfinite(c)), error('ltratio:args','ltratio: c must be a finite real scalar.'); end
if cas(1) == 'S' && ~(c < 0), error('ltratio:args','ltratio: a stationary start (cases SK, SU, ST) requires c < 0.'); end
if ~isstruct(opts), error('ltratio:args','ltratio: opts must be a structure.'); end
c0 = 0; if isfield(opts,'c0') && ~isempty(opts.c0), c0 = opts.c0; end
if ~(isnumeric(c0) && isscalar(c0) && isreal(c0) && isfinite(c0)), error('ltratio:args','ltratio: opts.c0 must be a finite real scalar.'); end
g0 = 0; if isfield(opts,'g0') && ~isempty(opts.g0), g0 = opts.g0; end
if ~(isnumeric(g0) && isscalar(g0) && isreal(g0) && isfinite(g0)), error('ltratio:args','ltratio: opts.g0 must be a finite real scalar.'); end
if g0 ~= 0 && cas(1) == 'S', error('ltratio:args','ltratio: opts.g0 (a fixed start) applies to cases DK, DU, DT only.'); end
method = 'auto'; if isfield(opts,'method') && ~isempty(opts.method), method = lower(char(opts.method)); end
if ~any(strcmp(method,{'auto','single','double'})), error('ltratio:args','ltratio: opts.method must be ''auto'', ''single'' or ''double''.'); end
if g0 ~= 0, method = 'double'; end                  % the single integral needs a central form of U + 1/2
end
