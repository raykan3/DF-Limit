function [F,info] = ltratio_dist(x,cas,c,opts,mode)
%LTRATIO_DIST  Engine dispatch of LTRATIO_CDF and LTRATIO_PDF (mode 'cdf' or 'pdf').
%
%   Validates the arguments, chooses the representation (opts.method:
%   'single', the default, or 'double'; a fixed nonzero start forces
%   'double'), maps the case to the engine's conventions and calls DFX_CORE
%   (single integral) or TR_LIMDIST (double integral).  Non-finite x: F = 0 at
%   -Inf, 1 (cdf) or 0 (pdf) at +Inf, NaN at NaN.
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
        case 'double'
            o = rmfield_safe(opts,{'method'});
            o.c0 = c0; o.g0 = g0;
            [Fv,inf2] = tr_limdist(x(fin),cas,c,o,mode);
            F(fin) = Fv;
            fn = fieldnames(inf2);
            for k = 1:numel(fn), info.(fn{k}) = inf2.(fn{k}); end
    end
end
info.time = toc(t0);
end

function o = rmfield_safe(o,names)
for k = 1:numel(names)
    if isfield(o,names{k}), o = rmfield(o,names{k}); end
end
end
