%ACC_TABLE  Table "The two representations against each other" of the paper: the single
%   integral (LTRATIO, method single) against the double integral (method double) at seven
%   points per configuration (mean plus -2.5..2.5 sd), zero and stationary start, with the
%   median times per value of the distribution function.  Writes acc_table.log and acc_rows.tex.
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..'));
logf = fullfile(here,'acc_table.log'); if exist(logf,'file'), delete(logf); end; diary(logf); fprintf('%s\n', datestr(now));
warning('off','all');
cfg = {0,0,'D'; -5,0,'D'; -20,0,'D'; 2,0,'D'; 5,0,'D'; 0,-5,'D'; 2,1,'D'; -2.5,-5,'D'; 5,2,'D'; 5,-5,'D'; -10,-20,'D'; 2,2,'D'; -2,-2,'D'; -5,3,'D'; -5,0,'S'; -20,0,'S'};
rows = {};
for ic = 1:size(cfg,1)
    c = cfg{ic,1}; c0 = cfg{ic,2}; st = cfg{ic,3};
    dF = zeros(1,3); df = zeros(1,3); t1 = []; t2 = []; note = '';
    for model = 1:3
        kut = 'KUT'; cas = [st kut(model)]; o = struct('c0',c0);
        m = ltratio_mom(2,cas,c,o); mu = m(1); sd = sqrt(m(2)-m(1)^2);
        xs = mu + sd*[-2.5 -1.5 -0.5 0 0.5 1.5 2.5];
        for x = xs
            tic; [F1,i1] = ltratio_cdf(x,cas,c,setfield(o,'method','single')); ta = toc;
            tic; [F2,i2] = ltratio_cdf(x,cas,c,setfield(o,'method','double')); tb = toc;
            f1 = ltratio_pdf(x,cas,c,setfield(o,'method','single')); [f2,j2] = ltratio_pdf(x,cas,c,setfield(o,'method','double'));
            s2 = 0; if isfield(i2,'status'), s2 = max(i2.status); end; if isfield(j2,'status'), s2 = max(s2, max(j2.status)); end
            fr = NaN; if abs(f2) > 1e-8, fr = abs(f1/f2-1); end
            fprintf('%s c=%5g c0=%5g x=%10.6f | F %.13f %.13f diff %8.1e | f rel %8.1e | t %.2f/%.2f st %d\n', cas, c, c0, x, F1, F2, F1-F2, fr, ta, tb, s2);
            if s2 > 0, note = [note sprintf(' %s x=%.1f(st%d)', cas, x, s2)]; continue; end
            dF(model) = max(dF(model), abs(F1-F2)); if ~isnan(fr), df(model) = max(df(model), fr); end
            t1(end+1) = ta; t2(end+1) = tb;
        end
    end
    fmt = @(v) fmtv(v);
    row = sprintf('$%g$ & $%g$ & %s & %s & %s & %s & %s & %s & %.2f & %.2f\\\\', c, c0, fmt(dF(1)), fmt(dF(2)), fmt(dF(3)), fmt(df(1)), fmt(df(2)), fmt(df(3)), median(t1), median(t2));
    fprintf('ROW %s %s   %% flagged:%s\n', st, row, note); rows{end+1} = row;
end
fid = fopen(fullfile(here,'acc_rows.tex'),'w'); for k = 1:numel(rows), fprintf(fid,'%s\n',rows{k}); end; fclose(fid);
diary off
function s = fmtv(v)
if v <= 0, s = '$0$'; return; end
e = floor(log10(v)); m = round(v/10^e); if m == 10, m = 1; e = e + 1; end
s = sprintf('$%d\\cdot10^{%d}$', m, e);
end
