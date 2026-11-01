function Q = tr_nsstart(lam,s,cas,c)
%TR_NSSTART  Initial-value term of the limiting joint mgf of (A', B), fixed start.
%
%   Q = TR_NSSTART(lam,s,cas,c): with J the Ornstein-Uhlenbeck process
%   dJ = c J dr + dW started at J(0) = g0, and A' = A + c B, B the functionals
%   of TR_CFLIM (cases 'DK', 'DU', 'DT'),
%      log E exp(lam A' + s B) = log E_0 exp(lam A' + s B) + g0^2 Q(lam,s),
%   E_0 the expectation with a zero start: the start enters the logarithm of
%   the moment generating function as a quadratic form in g0, and Q is its
%   coefficient, in closed form.  lam, s: arrays of a common size (or scalars),
%   real or complex, with (Re lam, Re s) in the domain of the zero-start
%   moment generating function.
%
%   Cases DU and DT (Nabeya and Sorensen, 1994, Theorems 1 and 2; the
%   initial value f = c g0 or c^2 g0 of their determinants):
%      Q(lam,s) = c^(2k) e^c (s + lam^2/2 + c lam) g(z)/D(lam,s),
%   k = 1 (DU) or 2 (DT), z = 2s - c^2, D(lam,s) = e^c {F0 + F1 (-lam) + F2
%   lam^2} the Fredholm determinant of the zero start (TR_CFLIM), and
%      g(z) = -S/z - 2(C - 1)/z^2                              (DU),
%      g(z) = 12C/z^3 - (24 + 4z)(C - 1)/z^4 - (24 - z)S/z^3   (DT),
%   C = cos sqrt(z), S = sin sqrt(z)/sqrt(z); g is entire, with the Taylor
%   series sum_k (-1)^k (2k+2)/(2k+4)! z^k (DU) and sum_k (-1)^k [1/(2k+5)!
%   - 8/(2k+6)! + 24/(2k+7)! - 24/(2k+8)!] z^k (DT), used for |z| < 4; beyond,
%   the closed forms in theta = (-z)^(1/2) and exp(-2 theta) with exp(theta)
%   factored out, so that nothing overflows.  D(lam,s) is formed from the
%   coefficients F0, F1, F2 of TR_CFLIM (form 'F'), scaled by exp(-ls) there;
%   no branch is involved.
%
%   As a function of lam, Q has simple poles at the roots r_k, and the
%   partial fractions give the squared means of the tilted Gaussian pair:
%   with X_k ~ N(m_k, 2|kap_k|) independent under the tilt e^{sB} and
%   R = A' + 1/2 = (X_1^2 - X_2^2)/2,
%      Q(lam,s) - Q(0,s) = (lam/2) [m_1^2/(1 - 2 lam kap_1) - m_2^2/(1 - 2 lam kap_2)]/g0^2.
%
%   Case DK (the Riccati equations of E_x exp(i u J(1) + s B), TR_JBSTART):
%   under the tilt J(1) ~ N(g0 mq, 2 kap_1) and log M_B gains g0^2 aq, so
%   with R = (J(1)^2 - g0^2)/2,
%      Q(lam,s) = aq(s) - lam/2 + lam mq(s)^2/{2 (1 - 2 lam kap_1(s))}.
%
%   The closed forms replace the Richardson extrapolation of the finite-n
%   initial-value terms (the finite-sample paper's TR_LOGCF at n = Inf) used
%   in earlier versions, which was accurate to about 1e-8; they agree with it
%   to that accuracy (Proposition 2 of the paper).
%
%   See also TR_CFLIM, TR_JBSTART, TR_LIMDIST, TR_LIMMOM.
cas = upper(cas);
sz = size(lam); if isscalar(lam), sz = size(s); end
lam = lam(:).' + zeros(1,prod(sz)); s = s(:).' + zeros(1,prod(sz));
% the coefficients F0, F1, F2 of D(lam,s) e^-c = F0 - F1 lam + F2 lam^2 (TR_CFLIM's
% lambda is -lam), scaled by exp(-ls); the branch of ls is immaterial here
[ls,~,Ft] = tr_cflim(0,-1i*s,cas,c,'F');
Dt = Ft(1,:) - Ft(2,:).*lam + Ft(3,:).*lam.^2;       % D(lam,s) e^(-c-ls)
switch cas
    case 'DK'
        [aq,mq] = tr_jbstart(-1i*s,c);
        Q = aq - lam/2 + lam.*mq.^2.*Ft(1,:)./(2*Dt);  % 1 - 2 lam kap_1 = (F0 - lam F1)/F0
    case {'DU','DT'}
        k = 1; if strcmp(cas,'DT'), k = 2; end
        z = 2*s - c^2;
        lg = logg(z,k);
        Q = c^(2*k)*(s + lam.^2/2 + c*lam).*exp(lg - ls)./Dt;
    otherwise
        error('tr_nsstart: case must be DK, DU or DT.');
end
Q = reshape(Q,sz);
end

function lg = logg(z,k)
% log g(z) (any branch; it is exponentiated), g the entire function above
lg = complex(zeros(size(z)));
sm = abs(z) < 4;
if any(sm)
    % Taylor series: a_j z^j, 24 terms suffice for |z| < 4 (|a_j 4^j| < 1e-30 at j = 24)
    j = (0:23).';
    if k == 1
        a = (-1).^j.*(2*j + 2)./factorial(2*j + 4);
    else
        a = (-1).^j.*(1./factorial(2*j + 5) - 8./factorial(2*j + 6) + 24./factorial(2*j + 7) - 24./factorial(2*j + 8));
    end
    zz = z(sm); g = zeros(size(zz)); p = ones(size(zz));
    for m = 1:numel(j), g = g + a(m)*p; p = p.*zz; end
    lg(sm) = log(g);
end
if any(~sm)
    % theta = (-z)^(1/2), principal root (Re theta >= 0), e = exp(-2 theta):
    % C = e^theta (1 + e)/2, S = e^theta (1 - e)/(2 theta), with e^theta taken out
    zz = z(~sm); th = sqrt(-zz); e = exp(-2*th); em = exp(-th);
    Cs = (1 + e)/2; Ss = (1 - e)./(2*th);          % C e^-theta, S e^-theta
    if k == 1
        gs = -Ss./zz - 2*(Cs - em)./zz.^2;
    else
        gs = 12*Cs./zz.^3 - (24 + 4*zz).*(Cs - em)./zz.^4 - (24 - zz).*Ss./zz.^3;
    end
    lg(~sm) = th + log(gs);
end
end
