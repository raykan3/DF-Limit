%ACC_TABLE  Table "The two representations against each other" of the paper: the single
%   integral (method single) against the double integral (method double) at seven points per
%   configuration (mean plus -2.5..2.5 sd), zero, stationary and fixed start, with the median
%   times per value of the distribution function.  Runs on the C engine (the MEX file in c/mex,
%   through LTRATIO, all threads) when it is on the path, otherwise on the MATLAB programs; the
%   engine and the thread count are recorded in the log.  Writes acc_table.log and acc_rows.tex.
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..'));
mexdir = fullfile(here,'..','..','c','mex');
usemex = exist(fullfile(mexdir, ['ltratio_mex.' mexext]), 'file') == 2 || exist(fullfile(mexdir, ['ltratio_mex.' mexext]), 'file') == 3;
if usemex, addpath(mexdir); end
logf = fullfile(here,'acc_table.log'); if exist(logf,'file'), delete(logf); end; diary(logf); fprintf('%s\n', datestr(now));
if usemex, fprintf('engine: the C MEX file (ltratio), %d threads\n', ltratio('threads')); else, fprintf('engine: the MATLAB programs, %d thread(s)\n', maxNumCompThreads); end
warning('off','all');
% the wrappers: with the MEX, LTRATIO (which takes method with a fixed start too); otherwise the MATLAB programs
if usemex
    cdf = @(x,cas,c,o) ltratio('cdf',x,cas,c,o); pdf = @(x,cas,c,o) ltratio('pdf',x,cas,c,o); mom = @(p,cas,c,o) ltratio('mom',p,cas,c,o);
else
    cdf = @(x,cas,c,o) ltratio_cdf(x,cas,c,o); pdf = @(x,cas,c,o) ltratio_pdf(x,cas,c,o); mom = @(p,cas,c,o) ltratio_mom(p,cas,c,o);
end
% columns: c, c0, start ('D' zero, 'S' stationary, 'F' fixed with g0 in the fourth column)
cfg = {0,0,'D',0; -5,0,'D',0; -20,0,'D',0; 2,0,'D',0; 5,0,'D',0; 0,-5,'D',0; 2,1,'D',0; -2.5,-5,'D',0; 5,2,'D',0; 5,-5,'D',0; -10,-20,'D',0; 2,2,'D',0; -2,-2,'D',0; -5,3,'D',0; -5,0,'S',0; -20,0,'S',0; ...
       -5,0,'F',1; -5,0,'F',3; 2,0,'F',1; -20,0,'F',1; -5,-5,'F',1};
rows = {};
for ic = 1:size(cfg,1)
    c = cfg{ic,1}; c0 = cfg{ic,2}; st = cfg{ic,3}; g0 = cfg{ic,4};
    dF = zeros(1,3); df = zeros(1,3); t1 = []; t2 = []; note = '';
    for model = 1:3
        kut = 'KUT'; cas = [st kut(model)]; if st == 'F', cas = ['D' kut(model)]; end
        o = struct('c0',c0); if g0 ~= 0, o.g0 = g0; end
        m = mom(2,cas,c,o); mu = m(1); sd = sqrt(m(2)-m(1)^2);
        xs = mu + sd*[-2.5 -1.5 -0.5 0 0.5 1.5 2.5];
        for x = xs
            tic; F1 = cdf(x,cas,c,setfield(o,'method','single')); ta = toc;
            tic; F2 = cdf(x,cas,c,setfield(o,'method','double')); tb = toc;
            f1 = pdf(x,cas,c,setfield(o,'method','single')); f2 = pdf(x,cas,c,setfield(o,'method','double'));
            % the double integral's status (nonzero where its error estimate is not met: the point is flagged and left out)
            s2 = 0;
            if usemex
                [~,~,st2] = ltratio_mex('double', 2, double(x), model, double(c), double(c0), double(g0), double(cas(1) == 'S')); [~,~,st3] = ltratio_mex('double', 1, double(x), model, double(c), double(c0), double(g0), double(cas(1) == 'S'));
                s2 = max(max(st2), max(st3));
            else
                [~,i2] = ltratio_cdf(x,cas,c,setfield(o,'method','double')); [~,j2] = ltratio_pdf(x,cas,c,setfield(o,'method','double'));
                if isfield(i2,'status'), s2 = max(i2.status); end; if isfield(j2,'status'), s2 = max(s2, max(j2.status)); end
            end
            fr = NaN; if abs(f2) > 1e-8, fr = abs(f1/f2-1); end
            fprintf('%s c=%5g c0=%5g g0=%g x=%10.6f | F %.13f %.13f diff %8.1e | f rel %8.1e | t %.3f/%.3f st %d\n', cas, c, c0, g0, x, F1, F2, F1-F2, fr, ta, tb, s2);
            if s2 > 0, note = [note sprintf(' %s x=%.1f(st%d)', cas, x, s2)]; continue; end
            dF(model) = max(dF(model), abs(F1-F2)); if ~isnan(fr), df(model) = max(df(model), fr); end
            t1(end+1) = ta; t2(end+1) = tb;
        end
    end
    fmt = @(v) fmtv(v);
    lab = sprintf('$%g$ & $%g$', c, c0); if st == 'F', lab = sprintf('$%g$ & $%g$, $\\gamma_0=%g$', c, c0, g0); end
    row = sprintf('%s & %s & %s & %s & %s & %s & %s & %.3f & %.3f\\\\', lab, fmt(dF(1)), fmt(dF(2)), fmt(dF(3)), fmt(df(1)), fmt(df(2)), fmt(df(3)), median(t1), median(t2));
    fprintf('ROW %s %s   %% flagged:%s\n', st, row, note); rows{end+1} = row;
end
fid = fopen(fullfile(here,'acc_rows.tex'),'w'); for k = 1:numel(rows), fprintf(fid,'%s\n',rows{k}); end; fclose(fid);
diary off
function s = fmtv(v)
if v <= 0, s = '$0$'; return; end
e = floor(log10(v)); m = round(v/10^e); if m == 10, m = 1; e = e + 1; end
s = sprintf('$%d\\cdot10^{%d}$', m, e);
end
