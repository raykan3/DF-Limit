% lim_tables.m: limiting (n = Inf) quantiles, probabilities at Fuller's
% tabulated n = Inf values, and local power, from TR_LIMDIST (direct
% evaluation of the limit; no extrapolation).
addpath('..'); warning('off','all');
cs = {'DK','DU','DT'}; nm = {'\tau','\tau_\mu','\tau_\tau'};
P = [0.01 0.025 0.05 0.10 0.50 0.90 0.95 0.975 0.99];
tab = [-2.58 -2.23 -1.95 -1.62 -0.51 0.89 1.28 1.62 2.01;
       -3.42 -3.12 -2.86 -2.57 -1.57 -0.44 -0.08 0.23 0.60;
       -3.96 -3.67 -3.41 -3.13 -2.18 -1.25 -0.94 -0.66 -0.32];
ext = [-2.5658 -2.2274 -1.94087 -1.6167 -0.500134 0.887574 1.28311 1.6229 2.01451;
       -3.43034 -3.12208 -2.86157 -2.56679 -1.56643 -0.440304 -0.0784752 0.237988 0.606671;
       -3.95867 -3.66139 -3.41045 -3.12702 -2.1806 -1.24637 -0.940043 -0.659705 -0.325104];
fo = fopen('lim_tables.txt','w');
Q = zeros(3,9); tq = zeros(3,9);
for k = 1:3
    for j = 1:9
        x = ext(k,j); t0 = clock;
        for it = 1:6
            F = tr_limdist(x,cs{k},0); f = tr_limdist(x,cs{k},0,[],'pdf');
            dx = (F - P(j))/f; x = x - dx;
            if abs(dx) < 1e-11, break, end
        end
        Q(k,j) = x; tq(k,j) = etime(clock,t0);
    end
    Pt = tr_limdist(tab(k,:),cs{k},0);
    fprintf(fo,'%s\nquantile %s\nextrap   %s\nP[t<=tab] %s\ntime %s\n',nm{k},sprintf('%.8f ',Q(k,:)),sprintf('%.6f ',ext(k,:)),sprintf('%.6f ',Pt),sprintf('%.1f ',tq(k,:)));
    fprintf('%s done\n',cs{k}); fflush(stdout);
end
% local power of the 5% tests (c0 = 0), limiting critical values
cc = [-30 -20 -15 -10 -5 -2.5];
for k = 1:3
    pw = zeros(size(cc));
    for j = 1:numel(cc), pw(j) = tr_limdist(Q(k,3),cs{k},cc(j),struct('c0',0)); end
    fprintf(fo,'power %s c=[%s]: %s\n',cs{k},sprintf('%g ',cc),sprintf('%.6f ',pw));
end
fclose(fo);
type lim_tables.txt
