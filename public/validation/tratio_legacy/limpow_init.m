% limpow_init: limiting power of the 5% Dickey-Fuller t-tests under phi = 1 + c/n
% with a fixed start y0 = xi sigma sqrt(n)/sqrt(-2c) (Muller-Elliott scaling,
% J(0) = xi/sqrt(-2c)) or a stationary start (J(0) ~ N(0,-1/(2c))), no drift,
% at the limiting 5% critical values of the zero-start null.
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
cs = [-30 -20 -15 -10 -5 -2.5]; xis = [0 1 2 3];
cv = [-1.94086, -2.86157, -3.41045];          % refined below
D = {'DK','DU','DT'}; S = {'SK','SU','ST'};
fid = fopen(fullfile(here,'limpow_init.txt'),'a');
if ~exist('rows','var'), rows = 1:3; end
for r = rows
    cv(r) = tratio_icdf(0.05,Inf,0,D{r});
    fprintf(fid,'%s cv %.8f\n',D{r},cv(r));
    for xi = xis
        P = zeros(size(cs)); t0 = tic;
        for j = 1:numel(cs)
            g0 = xi/sqrt(-2*cs(j));
            P(j) = tr_limdist(cv(r),D{r},cs(j),struct('c0',0,'g0',g0));
        end
        fprintf(fid,'%s xi=%g: %s (%.0f s)\n',D{r},xi,sprintf('%.6f ',P),toc(t0));
    end
    P = zeros(size(cs)); t0 = tic;
    for j = 1:numel(cs), P(j) = tr_limdist(cv(r),S{r},cs(j),struct('c0',0)); end
    fprintf(fid,'%s stat: %s (%.0f s)\n',S{r},sprintf('%.6f ',P),toc(t0));
end
fclose(fid);
