function P = mc_lim(cas,c,g0,c0,x,n,R,seed)
% Monte Carlo of the t-ratio at sample size n (phi = 1 + c/n), y0 = g0 sqrt(n)
% (cas D*) or stationary (cas S*), no drift; t for the null phi0 = 1 + c0/n.
randn('seed',seed); phi = 1 + c/n; phi0 = 1 + c0/n;
k = 1 + (cas(2)=='U') + 2*(cas(2)=='T');
if cas(1) == 'S', y = randn(R,1)/sqrt(1-phi^2); else, y = g0*sqrt(n)*ones(R,1); end
S = zeros(R,9);  % sums: 1:x 2:x2 3:xy 4:y 5:y2 6:t 7:t2 8:tx 9:ty
for t = 1:n
    yn = phi*y + randn(R,1);
    S(:,1) = S(:,1) + y; S(:,2) = S(:,2) + y.^2; S(:,3) = S(:,3) + y.*yn;
    S(:,4) = S(:,4) + yn; S(:,5) = S(:,5) + yn.^2;
    S(:,8) = S(:,8) + t*y; S(:,9) = S(:,9) + t*yn;
    y = yn;
end
tt = (1:n)'; St = sum(tt); Stt = sum(tt.^2);
switch k
  case 1
    Sxx = S(:,2); Sxy = S(:,3); Syy = S(:,5);
  case 2
    Sxx = S(:,2) - S(:,1).^2/n; Sxy = S(:,3) - S(:,1).*S(:,4)/n; Syy = S(:,5) - S(:,4).^2/n;
  case 3
    % partial out [1 t]: M = I - Z(Z'Z)^-1 Z'
    G = inv([n St; St Stt]);
    px = [S(:,1) S(:,8)]; py = [S(:,4) S(:,9)];
    Sxx = S(:,2) - sum((px*G).*px,2); Sxy = S(:,3) - sum((px*G).*py,2); Syy = S(:,5) - sum((py*G).*py,2);
end
ph = Sxy./Sxx; s2 = (Syy - ph.*Sxy)/(n-k); tr = (ph - phi0)./sqrt(s2./Sxx);
P = mean(tr <= x(:).',1);
end
