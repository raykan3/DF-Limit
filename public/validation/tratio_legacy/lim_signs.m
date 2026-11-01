% lim_signs.m: sign conditions of Theorem thm:limdist (Re kap1 > 0 > Re kap2) on a grid
% over the region between the vertical line and the parabola, and on the vertical line,
% for b in E B exp(-8:0.5:6).  Output: lim_signs.txt
addpath('..'); warning('off','all');
% sign conditions Re k1 > 0 > Re k2 on the region between the vertical line
% Re s = s* and the parabola s* + mu th^2 - 2 i mu th (and on the vertical line)
for cas = {'DK','DU','DT'}
for c = [-20 -10 -2.5 0 1 2]
  hh = 1e-5; Lp = tr_cflim(0,[hh -hh],cas{1},c,'ApB'); EB = imag(Lp(1)-Lp(2))/(2*hh);
  nbv = 0; nbr = 0; nb = 0;
  for b = EB*exp(-8:0.5:6)
    LB = @(s) real(tr_cflim(0*s,-1i*s,cas{1},c,'B'));
    % crude saddle by bisection on the derivative
    lo = -1/(8*b^2)*4-10; hi = 0; [~,ok] = tr_cflim(0,-1i*hi,cas{1},c,'B');
    while true, [~,ok] = tr_cflim(0,-1i*(hi+1),cas{1},c,'B'); if ~ok, break, end, hi = hi+1; end
    hi = hi+1;
    for k=1:200
      m = (lo+hi)/2; [~,ok] = tr_cflim(0,-1i*m,cas{1},c,'B'); e = 1e-6*max(1,abs(m));
      if ok, [~,ok2] = tr_cflim(0,-1i*(m+e),cas{1},c,'B'); end
      if ~ok || ~ok2, hi = m; continue, end
      d = (LB(m+e)-LB(m-e))/(2*e); if d > b, hi = m; else, lo = m; end
      if hi-lo < 1e-10*(1+abs(m)), break, end
    end
    s = lo; e = 1e-3*max(1,abs(s)); g2 = (LB(s+e/10)-2*LB(s)+LB(s-e/10))/(e/10)^2; a1 = 1/sqrt(abs(g2));
    mu = max(a1/2,4.5); th = linspace(0,sqrt(41.5/(b*mu)),60);
    [TH,U] = meshgrid(th, linspace(0,1,25));
    S = s + U.*mu.*TH.^2 - 2i*mu*TH; S = S(:,2:end);
    [~,~,kap] = tr_cflim(0,-1i*S(:).',cas{1},c,'B'); kap = reshape(kap,2,[]);
    bad = real(kap(1,:)) <= 0 | (~strcmp(cas{1},'DK') & real(kap(2,:)) >= 0);
    nbr = nbr + nnz(bad); nb = nb + numel(bad);
    % vertical line (old theorem statement)
    yv = a1*sinh(linspace(0,8,80)); [~,~,kv] = tr_cflim(0,-1i*(s+1i*yv),cas{1},c,'B'); kv = reshape(kv,2,[]);
    nbv = nbv + nnz(real(kv(1,:)) <= 0 | (~strcmp(cas{1},'DK') & real(kv(2,:)) >= 0));
  end
  printf('%s c=%g: region bad %d of %d; vertical bad %d\n',cas{1},c,nbr,nb,nbv);
end
end
