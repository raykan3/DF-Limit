function [f,info] = ltratio_pdf(x,cas,c,opts)
%LTRATIO_PDF  Limiting density of the Dickey-Fuller t-ratio.
%
%   f = LTRATIO_PDF(x,cas,c,opts): the density of the limit of the t-ratio;
%   arguments and options as in LTRATIO_CDF.  In case DK with c0 > 0 the
%   density is infinite at x = -sqrt(2 c0), where the two support edges of
%   the kernel merge (a logarithmic singularity), and Inf is returned there.
%
%   See also LTRATIO_CDF, LTRATIO_ICDF, LTRATIO_MOM.
if nargin < 3 || isempty(c), c = 0; end
if nargin < 4 || isempty(opts), opts = struct; end
[f,info] = ltratio_dist(x,cas,c,opts,'pdf');
end
