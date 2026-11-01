% lim_mom: moments of the limiting t-ratios (TR_LIMMOM).
%   1. The null moments (c = c0 = 0) against the eight-digit values of
%      Nabeya (1999, Table 4), Models 1-3 = cases DK, DU, DT.
%   2. The table of the paper: mean and standard deviation of the limit of
%      the Dickey-Fuller t-ratio (c0 = 0) under phi = 1 + c/n, with a zero
%      and with a stationary start.
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
fid = fopen(fullfile(here,'lim_mom.txt'),'w');
out = @(varargin) [fprintf(varargin{:}), fprintf(fid,varargin{:})];
nab = [-0.42309565 1.14159508 -1.06171218 3.52380298 -4.57465413 17.73354875;
       -1.53296244 3.05599535 -6.71991749 16.34553684 -42.72333103 119.58621597;
       -2.18135582 5.32067592 -14.03580010 39.55443050 -118.06034567 370.99533008];
cs = {'DK','DU','DT'};
out('1. Null moments 1-6 against Nabeya (1999, Table 4)\n');
for k = 1:3
    [m,info] = tr_limmom(6,cs{k},0);
    out('%s: %s\n    Nabeya %s\n    max |diff| %.1e, est. quadrature error %.1e (%.2f s)\n', ...
        cs{k},sprintf('%14.8f',m),sprintf('%14.8f',nab(k,:)),max(abs(m-nab(k,:))),max(info.err),info.time);
end
out('\n2. Mean and SD of the limit of the Dickey-Fuller t-ratio (c0 = 0)\n');
cc = [0 -2.5 -5 -10 -20];
for k = 1:3
    for st = 0:1
        if st, cas = ['S' cs{k}(2)]; else, cas = cs{k}; end
        out('%s',cas);
        for c = cc
            if st && c == 0, out('%22s','--'); continue, end
            m = tr_limmom(2,cas,c,struct('c0',0));
            out('   %8.4f (%6.4f)',m(1),sqrt(m(2)-m(1)^2));
        end
        out('\n');
    end
end
fclose(fid);
