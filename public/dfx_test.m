%DFX_TEST  Checks of the MATLAB port of DFExact.jl against the stored reference values of the
%   Julia code (../julia/python_exact_reference.json), node doubling, the closed form of
%   P(tau <= 0) in case DK, and the stationary start, moments and quantiles of quickcheck.jl.
here = fileparts(mfilename('fullpath'));
r = jsondecode(fileread(fullfile(here, '..', 'julia', 'python_exact_reference.json')));
fails = {};
wF = 0; wf = 0;
for k = 1:numel(r)
    if iscell(r), e = r{k}; else, e = r(k); end
    F = dfx_cdf(e.q, e.model, e.c); f = dfx_pdf(e.q, e.model, e.c);
    wF = max(wF, abs(F - e.cdf)); wf = max(wf, abs(f / e.density - 1));
end
fprintf('reference values (12 points): |dF| %.1e, |df/f| %.1e\n', wF, wf);
if ~(wF < 1e-12 && wf < 1e-10), fails{end+1} = 'reference values'; end
% node doubling
wd = 0;
for model = 1:3
    for cq = [0 -2; 0 0.5; -5 -3; 3 2; 3 8]'
        F = dfx_cdf(cq(2), model, cq(1)); f = dfx_pdf(cq(2), model, cq(1));
        dfx_core('nodefac', 2); F2 = dfx_cdf(cq(2), model, cq(1)); f2 = dfx_pdf(cq(2), model, cq(1)); dfx_core('nodefac', 1);
        wd = max([wd, abs(F - F2), abs(f - f2) / abs(f)]);
    end
end
fprintf('node doubling: worst change %.1e\n', wd);
if ~(wd < 1e-10), fails{end+1} = 'node doubling'; end
% P(tau <= 0) in case DK: erf((2v)^(-1/2)), v = (e^(2c) - 1)/(2c)
w0 = 0;
for c = [-30 -20 -10 -2.5 0 1 2]
    if c == 0, v = 1; else, v = (exp(2*c) - 1) / (2*c); end
    w0 = max(w0, abs(dfx_cdf(0, 1, c) - erf(1 / sqrt(2*v))));
end
fprintf('P(tau <= 0), case DK, c = -30..2: worst error %.1e\n', w0);
% stationary start, moments, quantile (quickcheck.jl)
Fs = dfx_cdf(-2.86157, 2, -5, struct('start', 'stationary'));
fprintf('stationary start, M2, c = -5: power %.6f (0.125)\n', Fs);
if abs(Fs - 0.125) > 5e-4, fails{end+1} = 'stationary'; end
m = dfx_moments(2, 0, struct('pmax', 2));
fprintf('E tau_mu^2 at c = 0: %.10f (Nabeya 1999: 3.05599535)\n', m(2));
if abs(m(2) - 3.05599535) > 1e-8, fails{end+1} = 'moments'; end
q05 = dfx_quantile(0.05, 2);
fprintf('M2 5%% quantile: %.9f (-2.861568864)\n', q05);
if abs(q05 + 2.861568864) > 1e-8, fails{end+1} = 'quantile'; end
if isempty(fails), fprintf('dfx_test passed\n'); else, error('dfx_test failed: %s', strjoin(fails, ', ')); end
