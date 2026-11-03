function [F,info] = ltratio_dist(x,cas,c,opts,mode)
%LTRATIO_DIST  Engine dispatch of LTRATIO_CDF and LTRATIO_PDF (mode 'cdf' or 'pdf').
%
%   Validates the arguments, chooses the representation (opts.method:
%   'single', the default, or 'double'; a fixed nonzero start forces
%   'double'), maps the case to the engine's conventions and calls DFX_CORE
%   (single integral) or TR_LIMDIST (double integral).
%
%   Tails.  The single integral has an absolute accuracy, not a relative one:
%   its terms are of order one, and the computed density or distribution
%   function carries an error of order 1e-14 to 1e-21 whatever its size,
%   which finer rules do not reduce (2e-14 in the lower tail of cases DT
%   and ST at c0 ~= 0; densities below the floor are noise and can be
%   negative: -6e-21 at x = 8.5 in case SU, c = c0 = -1.2, where the density
%   is 3e-22).  When the method is not given, the values of the single
%   integral with a density below 1e-3 in absolute value, or a distribution
%   function below 1e-3, are therefore recomputed by the double integral,
%   which keeps its relative accuracy in the tails and returns an error
%   estimate (info.ntail of them; info.err holds the estimates there and NaN
%   elsewhere), so that the relative error stays near 1e-10 or below.  With
%   opts.method = 'single' given explicitly, the single integral is used
%   throughout.  Far out, below the error of either method, a computed
%   density can be a tiny negative number; it is returned as zero.
%   Non-finite x: F = 0 at -Inf, 1 (cdf) or 0 (pdf) at +Inf, NaN at NaN.
%
%   See also LTRATIO_CDF, LTRATIO_PDF, DFX_CDF, TR_LIMDIST.
t0 = tic;
[cas,c,c0,g0,method] = ltratio_args(cas,c,opts);
if ~(isnumeric(x) && isreal(x)), error('ltratio:args','ltratio_%s: x must be real.',mode); end
F = zeros(size(x)); fin = isfinite(x);
F(isnan(x)) = NaN; if strcmp(mode,'cdf'), F(x == Inf) = 1; end
info = struct('method',method,'time',0);
if any(fin(:))
    switch method
        case 'single'
            model = find('KUT' == cas(2));
            o = rmfield_safe(opts,{'method','g0'});
            o.c0 = c0;
            if cas(1) == 'S', o.start = 'stationary'; else, o.start = 'zero'; end
            F(fin) = dfx_core(mode,x(fin),model,c,o);
            explicit = isstruct(opts) && isfield(opts,'method') && ~isempty(opts.method);
            if ~explicit
                if strcmp(mode,'pdf'), tl = fin & abs(F) < 1e-3; else, tl = fin & F < 1e-3; end
                info.ntail = nnz(tl);
                if any(tl(:))
                    o2 = rmfield_safe(opts,{'method'}); o2.c0 = c0; o2.g0 = g0;
                    [Fv,inf2] = tr_limdist(x(tl),cas,c,o2,mode);
                    F(tl) = Fv;
                    info.err = NaN(size(x)); if isfield(inf2,'err'), info.err(tl) = inf2.err; end
                end
            end
        case 'double'
            o = rmfield_safe(opts,{'method'});
            o.c0 = c0; o.g0 = g0;
            [Fv,inf2] = tr_limdist(x(fin),cas,c,o,mode);
            F(fin) = Fv;
            fn = fieldnames(inf2);
            for k = 1:numel(fn), info.(fn{k}) = inf2.(fn{k}); end
    end
end
if strcmp(mode,'pdf'), F(F < 0) = 0; end            % a density below the error of the method
info.time = toc(t0);
end

function o = rmfield_safe(o,names)
for k = 1:numel(names)
    if isfield(o,names{k}), o = rmfield(o,names{k}); end
end
end
