function [F,info] = ltratio_dist(x,cas,c,opts,mode)
%LTRATIO_DIST  Engine dispatch of LTRATIO_CDF and LTRATIO_PDF (mode 'cdf' or 'pdf').
%
%   Validates the arguments, chooses the representation (opts.method:
%   'auto', the default, 'single' or 'line', 'double' or 'conditional'; a
%   fixed nonzero start forces 'double'), maps the case to the engine's conventions and calls DFX_CORE
%   (line integral) or TR_LIMDIST (conditional integral).  With 'auto' the
%   choice is made point by point by LTRATIO_METHOD, from the timings of the
%   two representations in the MATLAB programs (single for c0 = 0; with
%   c0 ~= 0 double except for c0 > 0 at x >= 0 in cases DK, DU, DT, SK, and
%   single except for c0 < 0 at x >= 0 in cases SU, ST); info.ndouble is
%   the number of points sent to the conditional integral by the rule.  A point
%   at which the conditional integral reports that its error estimate is not
%   met (info.status, info.err) gets the line integral's value instead, when a
%   zero or stationary start makes it available; info.nfallback counts them.
%
%   Tails.  The line integral has an absolute accuracy, not a relative one:
%   its terms are of order one, and the computed density or distribution
%   function carries an error of order 1e-14 to 1e-21 whatever its size,
%   which finer rules do not reduce (2e-14 in the lower tail of cases DT
%   and ST at c0 ~= 0; densities below the floor are noise and can be
%   negative: -6e-21 at x = 8.5 in case SU, c = c0 = -1.2, where the density
%   is 3e-22).  With 'auto', the values of the line integral with a
%   density below 1e-3 in absolute value, or a distribution function below
%   1e-3, are therefore recomputed by the conditional integral, which keeps its
%   relative accuracy in the tails and returns an error estimate (info.ntail
%   of them; info.err holds the estimates at every point computed by the
%   conditional integral and NaN elsewhere), so that the relative error stays
%   near 1e-10 or below.  With opts.method = 'single' given explicitly, the
%   line integral is used throughout.  Far out, below the error of either
%   method, a computed density can be a tiny negative number; it is returned
%   as zero.  Non-finite x: F = 0 at -Inf, 1 (cdf) or 0 (pdf) at +Inf, NaN
%   at NaN.
%
%   See also LTRATIO_METHOD, LTRATIO_CDF, LTRATIO_PDF, DFX_CDF, TR_LIMDIST.
t0 = tic;
[cas,c,c0,g0,method] = ltratio_args(cas,c,opts);
if ~(isnumeric(x) && isreal(x)), error('ltratio:args','ltratio_%s: x must be real.',mode); end
F = zeros(size(x)); fin = isfinite(x);
F(isnan(x)) = NaN; if strcmp(mode,'cdf'), F(x == Inf) = 1; end
info = struct('method',method,'time',0);
if any(fin(:))
    switch method
        case 'single'
            F(fin) = single_values(x(fin),cas,c,c0,opts,mode);
        case 'double'
            o = rmfield_safe(opts,{'method'});
            o.c0 = c0; o.g0 = g0;
            [Fv,inf2] = tr_limdist(x(fin),cas,c,o,mode);
            F(fin) = Fv;
            fn = fieldnames(inf2);
            for k = 1:numel(fn), info.(fn{k}) = inf2.(fn{k}); end
        case 'auto'
            dbl = fin & ltratio_method(cas,c,c0,g0,x);
            sgl = fin & ~dbl;
            info.ndouble = nnz(dbl); info.ntail = 0;
            if any(sgl(:))
                F(sgl) = single_values(x(sgl),cas,c,c0,opts,mode);
                if strcmp(mode,'pdf'), tl = sgl & abs(F) < 1e-3; else, tl = sgl & F < 1e-3; end
                info.ntail = nnz(tl); dbl = dbl | tl;
            end
            if any(dbl(:))
                o2 = rmfield_safe(opts,{'method'}); o2.c0 = c0; o2.g0 = g0;
                ws = warning('off','tr_limdist:status');
                [Fv,inf2] = tr_limdist(x(dbl),cas,c,o2,mode);
                warning(ws);
                F(dbl) = Fv;
                info.err = NaN(size(x)); if isfield(inf2,'err'), info.err(dbl) = inf2.err; end
                info.status = zeros(size(x)); if isfield(inf2,'status'), info.status(dbl) = inf2.status; end
                % a point at which the conditional integral reports that its error estimate is not met gets the line
                % integral's value (the conditional integral is the only route with a fixed start)
                fb = info.status ~= 0 & g0 == 0; info.nfallback = nnz(fb);
                if any(fb(:)), F(fb) = single_values(x(fb),cas,c,c0,opts,mode); end
            end
    end
end
if strcmp(mode,'pdf'), F(F < 0) = 0; end            % a density below the error of the method
info.time = toc(t0);
end

function F = single_values(x,cas,c,c0,opts,mode)
model = find('KUT' == cas(2));
o = rmfield_safe(opts,{'method','g0'});
o.c0 = c0;
if cas(1) == 'S', o.start = 'stationary'; else, o.start = 'zero'; end
F = dfx_core(mode,x,model,c,o);
end

function o = rmfield_safe(o,names)
for k = 1:numel(names)
    if isfield(o,names{k}), o = rmfield(o,names{k}); end
end
end
