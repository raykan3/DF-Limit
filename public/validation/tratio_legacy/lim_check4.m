% lim_check4: limiting distributions by TR_LIMDIST (closed-form cf, Gaussian reduction)
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
fid = fopen(fullfile(here,'lim_check4.txt'),'w');
cs = {'DK','DU','DT'};
xq = {[0 -2.565804 -1.940865 -1.616698 0.887588 1.622891], ...
      [0 -3.430357 -2.861572 -2.566788 -0.440304 0.237981], ...
      [0 -3.958686 -3.410453 -3.127023 -1.246366 -0.659700]};
for i = 1:3
  t0 = tic; F = tr_limdist(xq{i},cs{i},0); t1 = toc(t0);
  F2 = tr_limdist(xq{i},cs{i},0,struct('M',96,'Nv',120,'Nl',144));
  fprintf(fid,'%s F  = %s (%.1f s)\n   F2 = %s\n',cs{i},mat2str(F,11),t1,mat2str(F2,11));
end
fprintf(fid,'exact: DK x=0 %.12f; DU x=0 0.95756559 (AR paper); others: 0.01 0.05 0.10 0.90 0.975 (extrapolated quantiles)\n',2*normcdf(1)-1);
t0 = tic; f = tr_limdist([0 -1.940865],'DK',0,[],'pdf'); fprintf(fid,'DK pdf %s (%.1f s)\n',mat2str(f,10),toc(t0));
fclose(fid);
