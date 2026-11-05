function method_sweep(cases, cs, c0s, xs, logname)
%METHOD_SWEEP  Per-point timings of the two representations in the MATLAB programs.
%   METHOD_SWEEP(cases, cs, c0s, xs, logname): for every case in CASES (cell array of case
%   strings), c in CS, c0 in C0S and x in XS, the distribution function and the density by the
%   single integral (DFX_CORE) and by the double integral (TR_LIMDIST), with the time of each
%   (the minimum of two calls for the distribution function, one call for the density, after a
%   warm-up call per configuration), the difference of the values and the double integral's
%   status.  One line per point in the log file LOGNAME (public/validation/).  Runs on one
%   computational thread, the convention of the paper's MATLAB timings.  Basis of the automatic
%   choice of the method in LTRATIO_DIST (November 2026).
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..'));
maxNumCompThreads(1); warning('off','all');
fid = fopen(fullfile(here, logname), 'w');
fprintf(fid, '%% %s  MATLAB %s, %d thread(s)\n', datestr(now), version, maxNumCompThreads);
fprintf(fid, '%% cas c c0 x | F_single F_double dF | t_cdf_single t_cdf_double | f_single f_double | t_pdf_single t_pdf_double | status_double\n');
for ic = 1:numel(cases)
    cas = cases{ic};
    for c = cs
        if cas(1) == 'S' && c >= 0, continue; end
        for c0 = c0s
            S = struct('c0', c0, 'method', 'single'); D = struct('c0', c0, 'method', 'double');
            try, ltratio_cdf(xs(1), cas, c, S); ltratio_cdf(xs(1), cas, c, D); ltratio_pdf(xs(1), cas, c, S); ltratio_pdf(xs(1), cas, c, D); catch, end
            for x = xs
                [F1, F2, f1, f2] = deal(NaN); [t1, t2, u1, u2] = deal(NaN); st = -1;
                try, tic; F1 = ltratio_cdf(x, cas, c, S); t1 = toc; tic; ltratio_cdf(x, cas, c, S); t1 = min(t1, toc); catch e, fprintf(fid, '%% single cdf failed: %s\n', e.message); end
                try, tic; [F2, i2] = ltratio_cdf(x, cas, c, D); t2 = toc; tic; ltratio_cdf(x, cas, c, D); t2 = min(t2, toc); st = 0; if isfield(i2, 'status'), st = max(i2.status); end; catch e, fprintf(fid, '%% double cdf failed: %s\n', e.message); end
                try, tic; f1 = ltratio_pdf(x, cas, c, S); u1 = toc; catch e, fprintf(fid, '%% single pdf failed: %s\n', e.message); end
                try, tic; f2 = ltratio_pdf(x, cas, c, D); u2 = toc; catch e, fprintf(fid, '%% double pdf failed: %s\n', e.message); end
                fprintf(fid, '%s %g %g %g | %.15g %.15g %.2e | %.4f %.4f | %.15g %.15g | %.4f %.4f | %d\n', cas, c, c0, x, F1, F2, F1 - F2, t1, t2, f1, f2, u1, u2, st);
            end
        end
    end
end
fclose(fid);
end
