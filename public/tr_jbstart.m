function [aq,mq] = tr_jbstart(tB,c)
%TR_JBSTART  Initial-value terms of the limiting cf of (J(1), B), case DK.
%   For the Ornstein-Uhlenbeck process dJ = c J dr + dW started at J(0) = g0,
%      log E exp(i s J(1) + i tB B) = L0(s,tB) + aq g0^2 + i s g0 mq,
%   L0 the zero-start form ('JB' of TR_CFLIM), from the Riccati equations of
%   E_x exp(i s J(1) - lambda int J^2), lambda = -i tB, mu = (c^2 + 2 lambda)^(1/2):
%      aq = -lambda sinh(mu)/(mu D0),  mq = 1/D0,  D0 = cosh(mu) - (c/mu) sinh(mu)
%   (mq is the mean factor of J(1) under the B-tilt; mq = e^c at tB = 0).
%   Written with e = exp(-2 mu), Re mu >= 0, free of overflow; even in mu.
lam = -1i*tB; mu = sqrt(c^2 + 2*lam);
e = exp(-2*mu);
q = (1 - e)./mu;                                  % 2 sinh(mu) e^(-mu)/mu
sm = abs(mu) < 1e-4; q(sm) = 2 - 2*mu(sm) + 4/3*mu(sm).^2;
den = (1 + e) - c*q;                              % 2 D0 e^(-mu)
aq = -lam.*q./den;
mq = 2*exp(-mu)./den;
end
