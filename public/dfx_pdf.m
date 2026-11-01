function f = dfx_pdf(q, model, c, opts)
%DFX_PDF  Limiting density of the Dickey-Fuller t-ratio (port of DFExact.density).
%   f = DFX_PDF(q, model, c, opts); arguments as in DFX_CDF.
if nargin < 3, c = 0; end
if nargin < 4, opts = struct(); end
f = dfx_core('pdf', q, model, c, opts);
end
