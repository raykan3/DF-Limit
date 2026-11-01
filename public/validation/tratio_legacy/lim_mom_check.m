% lim_mom_check: moments of the limits (TR_LIMMOM) against Gauss-Legendre
% quadrature of the limiting densities (TR_LIMDIST, an independent route),
% for local alternatives and fixed or stationary starts.
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
fid = fopen(fullfile(here,'lim_mom_check.txt'),'w');
cfg = {{'DU',-5,struct('c0',0)},{'DT',-10,struct('c0',0)},{'DK',2,struct()},{'DK',-5,struct('g0',1)}, ...
       {'DU',-5,struct('g0',1.5)},{'SK',-5,struct()},{'SU',-5,struct('c0',0)},{'ST',-5,struct()}};
m = 160; b = (1:m-1)./sqrt(4*(1:m-1).^2-1); [V,D] = eig(diag(b,1)+diag(b,-1));
[z,ix] = sort(diag(D)); w = 2*V(1,ix).^2; z = z(:)';
for k = 1:numel(cfg)
    c = cfg{k}; o = c{3};
    [mm,info] = tr_limmom(4,c{1},c{2},o);
    q = zeros(1,4); mass = 0;
    for pc = [-12 -3; -3 1; 1 8]'                     % three panels
        x = (pc(1)+pc(2))/2 + (pc(2)-pc(1))/2*z; wx = (pc(2)-pc(1))/2*w;
        f = tr_limdist(x,c{1},c{2},o,'pdf');
        mass = mass + sum(wx.*f);
        for j = 1:4, q(j) = q(j) + sum(wx.*f.*x.^j); end
    end
    s = sprintf('%s c=%g c0=%g g0=%g: tr_limmom %s (%.2f s)\n      density   %s  mass %.10f  max rel diff %.1e\n', ...
        c{1},c{2},getf(o,'c0',c{2}),getf(o,'g0',0),sprintf('%14.9f',mm),info.time,sprintf('%14.9f',q),mass,max(abs(q-mm)./abs(mm)));
    fprintf('%s',s); fprintf(fid,'%s',s);
end
fclose(fid);
function v = getf(o,f,d), if isfield(o,f), v = o.(f); else, v = d; end, end
