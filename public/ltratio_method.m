function dbl = ltratio_method(cas,c,c0,g0,x)
%LTRATIO_METHOD  Automatic choice of the representation (opts.method = 'auto').
%
%   dbl = LTRATIO_METHOD(cas,c,c0,g0,x) is a logical array of the size of x,
%   true where the double integral (TR_LIMDIST) is the faster representation
%   and false where the single integral (DFX_CORE) is.  The rule, from the
%   per-point timings of the MATLAB programs themselves on one thread
%   (validation/method_sweep.m, November 2026; c in [-5,5], c0 in {-5,-1,1,3},
%   x in [-4,32], the six cases):
%
%     g0 ~= 0            double (the only route with a fixed start);
%     c0 == 0            single: 1.2 to 60 times faster everywhere;
%     cases DK, DU, DT, SK with c0 ~= 0
%                        double, except for c0 > 0 at x >= 0, where the
%                        single integral has no breakpoint and wins by 3 to
%                        30 times; with c0 < 0 the double wins by 2 to 10
%                        times at x < 0 and the single integral's cost
%                        spikes to 5-11 seconds at x of 4 to 8 (its saddle
%                        factor), while the double stays below 0.5;
%     cases SU, ST with c0 ~= 0
%                        single, except for c0 < 0 at x >= 0, where the
%                        single integral's saddle factor costs up to 2.7
%                        seconds against 0.5 by the double; at x < 0 the
%                        single wins by 1 to 4 times in these cases, whose
%                        double integral costs 0.5 seconds a value.
%
%   The rule of the C and Julia engines (dfl_automethod, automethod) is the
%   one of the first three lines for every start; it differs here in cases
%   SU and ST at x < 0, where the MATLAB double integral is the slower one,
%   and only there.  Where the two representations are within a factor of
%   two of each other the rule follows the C engine (case DT with c0 = -5 at
%   x >= 0 and c <= 2, case DK with c0 = -5 at c <= 0 and x >= 4, where the
%   single integral is 1.2 to 2.8 times faster; case SK with c0 = -1 at
%   x >= 16).
%
%   See also LTRATIO_DIST, LTRATIO_CDF, LTRATIO_PDF.
if g0 ~= 0, dbl = true(size(x)); return; end
if c0 == 0, dbl = false(size(x)); return; end
if any(strcmp(cas,{'SU','ST'}))
    dbl = (c0 < 0) & (x >= 0);
else
    dbl = ~((c0 > 0) & (x >= 0));
end
dbl = logical(dbl) & true(size(x));
end
