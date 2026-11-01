% val_lim: the local-to-unity limit (n = Inf, phi = c = 0) against known values
%   P[tau <= 0] = P[A <= 0]: 2 Phi(1) - 1 (DK), 0.95756559 (DU, Evans and Savin 1981),
%   and the one-dimensional inversion of the limit cf of A (DT);
%   MacKinnon (2010) asymptotic 1%, 5%, 10% critical values (response-surface estimates).
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
fid = fopen(fullfile(here,'val_lim.txt'),'w');
mk = {[-2.56574 -1.94100 -1.61682],[-3.43035 -2.86154 -2.56677],[-3.95877 -3.41049 -3.12705]};
cs = {'DK','DU','DT'};
for j = 1:3
  cas = cs{j};
  J = tr_logcf(0,[0;1],0,Inf,0,cas); su = 1/sqrt(imag(J(2)));
  f = @(s) imag(exp(tr_logcf(s*su,0*s,0*s,Inf,0,cas)))./s;
  tau = -36:0.01:8; s = exp(tau); v = f(s); v(~isfinite(v)) = 0; Pu = 0.5 - sum(v.*s)*0.01/pi;
  for N = [60 90]
    t0 = tic; F = tratio_cdf([0 mk{j}],Inf,0,cas,[],struct('N',N));
    fprintf(fid,'%s limit N=%d: P[t<=0] = %.10f, P[A<=0] = %.10f, diff %+.1e; at MacKinnon 1/5/10%%: %s (%.0f s)\n',cas,N,F(1),Pu,F(1)-Pu,mat2str(F(2:4),6),toc(t0));
  end
end
fprintf(fid,'2Phi(1)-1 = %.10f\n',erf(1/sqrt(2)));
fclose(fid);
