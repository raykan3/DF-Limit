% lim_gpcheck.m: TR_LIMDIST at x = 0 against a one-dimensional inversion of the
% moment generating function of X = A' - c0 B (P[tau <= 0] = P[X <= 0]),
%   P[X <= 0] = -(1/(2 pi i)) int M_X(s)/s ds,  Re s = sig < 0,
% for c0 ~= 0 (where M_X decays fast along the line), and the exact DK values
% P[J(1)^2 <= 1] at c0 = 0.
addpath('..'); warning('off','all');
fo = fopen('lim_gpcheck.txt','w');
for cas = {'DK','DU','DT'}
for cc = [0 -5; 2 1; -10 -20; -2.5 -5]'
  c = cc(1); c0 = cc(2);
  MX = @(s) tr_cflim(-1i*s, 1i*c0*s, cas{1}, c, 'ApB');
  Pg = zeros(1,2); k = 0;
  for hs = [0.01 0.005]
    k = k+1; sig = -0.03; tau = (-ceil(30/hs):ceil(30/hs))*hs; y = 0.05*sinh(tau); dy = 0.05*cosh(tau)*hs;
    keep = abs(y) < 1e8; y = y(keep); dy = dy(keep);
    S = sig + 1i*y; L = MX(S);
    Pg(k) = -real(sum(exp(L)./S.*dy))/(2*pi);
  end
  tic; [F,info] = tr_limdist(0,cas{1},c,struct('c0',c0)); t = toc;
  fprintf(fo,'%s c=%g c0=%g: inversion %.13f (step halved: %.1e)  limdist %.13f  diff %.1e  nbad %d  (%.1f s)\n',cas{1},c,c0,Pg(2),Pg(2)-Pg(1),F,F-Pg(2),info.nbad,t);
end
end
for c = [-20 -10 -2.5 0 1 2]
  if c == 0, v = 1; else, v = (exp(2*c)-1)/(2*c); end
  [F,info] = tr_limdist(0,'DK',c,struct('c0',0));
  fprintf(fo,'DK exact c=%g: %.15f err %.1e nbad %d\n',c,F,F-erf(1/sqrt(2*v)),info.nbad);
end
fclose(fo); type lim_gpcheck.txt
