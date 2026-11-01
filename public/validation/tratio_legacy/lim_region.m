% lim_region.m: the sign conditions Re kap_1 > 0 > Re kap_2 of Theorem
% thm:limdist on the region between the vertical line through s* and the
% parabola used by TR_LIMDIST (11 intermediate parabolas s* + u mu th^2 - 2 i mu th,
% u in [0,1]; u = 0 is the vertical line), at all nodes of a computation.
addpath('..'); warning('off','all');
fo = fopen('lim_region.txt','w');
for cas = {'DK','DU','DT'}
for cc = [0 0; -20 0; -10 0; -2.5 0; 1 0; 2 0; 0 -5; 2 1]'
  [F,info] = tr_limdist([-3 -1 0 1],cas{1},cc(1),struct('c0',cc(2),'region',11));
  fprintf(fo,'%s c=%g c0=%g: contour bad %d; region bad %d of %d\n',cas{1},cc(1),cc(2),info.nbad,info.nreg,info.ntot);
end
end
fclose(fo); type lim_region.txt
