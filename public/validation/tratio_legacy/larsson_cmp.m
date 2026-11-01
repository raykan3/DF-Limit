% Our limiting values at the grids of Larsson (1995), Tables B.2 and B.3.
% B.2: P(Z'_beta < z), Z' = int J dW / sqrt(int J^2), the t-ratio centered at the
%      true value (c = c0 = beta).
% B.3: P(Ztilde_beta < z), Ztilde = int J dJ / sqrt(int J^2), the Dickey-Fuller
%      t-ratio under the local alternative (c = beta, c0 = 0).
here = fileparts(mfilename('fullpath')); addpath(fullfile(here,'..')); warning('off','all');
bs = [-2 -1 0 1 2];
z2 = [-3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3];
z3 = [-3 -2.5 -2 -1.5 -1 -0.5 0 0.25 0.5 0.75 1 1.25 1.5 2 2.5 3 3.5 4 5 6 8 10];
fid = fopen(fullfile(here,'larsson_cmp.txt'),'w');
fprintf(fid,'B.2 P(Z''<z), c = c0 = beta\n   z %s\n',sprintf('   b=%2d   ',bs));
P = zeros(numel(z2),numel(bs)); for j = 1:numel(bs), P(:,j) = tr_limdist(z2,'DK',bs(j)); end
for i = 1:numel(z2), fprintf(fid,'%5.2f %s\n',z2(i),sprintf('%10.6f ',P(i,:))); end
fprintf(fid,'B.3 P(Ztilde<z), c = beta, c0 = 0\n   z %s\n',sprintf('   b=%2d   ',bs));
P = zeros(numel(z3),numel(bs)); for j = 1:numel(bs), P(:,j) = tr_limdist(z3,'DK',bs(j),struct('c0',0)); end
for i = 1:numel(z3), fprintf(fid,'%5.2f %s\n',z3(i),sprintf('%10.6f ',P(i,:))); end
fclose(fid);
