function w = faddeeva(z)
%FADDEEVA  The Faddeeva function w(z) = exp(-z^2) erfc(-i z) for complex z.
%   Weideman (1994, SIAM J. Numer. Anal. 31, 1497-1518), rational expansion
%   with N = 36 terms in the upper half plane, and w(z) = 2 exp(-z^2) - w(-z)
%   below the real axis.  Relative accuracy about 1e-14 in the region used here.
N = 36; M = 2*N; M2 = 2*M;
k = (-M+1:M-1)'; L = sqrt(N/sqrt(2));
theta = k*pi/M; t = L*tan(theta/2);
f = exp(-t.^2).*(L^2 + t.^2); f = [0; f];
a = real(fft(fftshift(f)))/M2; a = flipud(a(2:N+1));
w = complex(zeros(size(z)));
up = imag(z) >= 0;
w(up) = wup(z(up),a,L);
if any(~up(:))
    zm = z(~up);
    w(~up) = 2*exp(-zm.^2) - wup(-zm,a,L);
end
end
function w = wup(z,a,L)
Z = (L + 1i*z)./(L - 1i*z);
p = a(1)*ones(size(Z));
for k = 2:numel(a), p = p.*Z + a(k); end
w = 2*p./(L - 1i*z).^2 + (1/sqrt(pi))./(L - 1i*z);
end
