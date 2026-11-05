% TIMING_FIXED  Seconds per value of the distribution function with a fixed start: the MATLAB programs (double
% integral, 1 thread) against the C MEX (single integral with the noncentral kernel, and the double integral), 1 and
% 10 threads; timeit medians (the fixed-start rows of the table in README.md and the cost paragraph of the paper).
%   Needs the MEX file in this folder and MATLAB; run from anywhere.
here = fileparts(fileparts(fileparts(mfilename('fullpath')))); addpath(fullfile(here,'public')); mexdir = fullfile(here,'c','mex');
cd(here);   % not this folder: its engine replacements would shadow the MATLAB programs from the current directory
pts = {'DK',-5,0,1,-2; 'DK',2,0,1,1; 'DU',-5,0,1,-2.86; 'DU',-5,0,1,0.5; 'DU',2,0,1,-3; 'DT',-5,0,2,-3; 'DT',2,0,1,-2; 'DU',-5,-5,1,-2; 'DT',-2.5,-5,1,-1.1};
fprintf('%-18s %12s %12s %12s %12s %12s   %18s %18s\n', 'case c c0 g0 x', 'MATLAB dbl', 'MEX1 single', 'MEX1 double', 'MEX10 single', 'MEX10 double', 'F single', 'F double');
for k = 1:size(pts,1)
    cas = pts{k,1}; c = pts{k,2}; c0 = pts{k,3}; g0 = pts{k,4}; x = pts{k,5};
    o = struct('c0',c0,'g0',g0);
    % the MATLAB programs: public/ ahead of c/mex on the path, one thread
    rmpath(mexdir); addpath(fullfile(here,'public')); maxNumCompThreads(1);
    tm = timeit(@() ltratio_cdf(x, cas, c, setfield(o,'method','double')));
    Fm = ltratio_cdf(x, cas, c, setfield(o,'method','double'));
    % the MEX: c/mex ahead
    addpath(mexdir);
    ltratio('threads', 1);
    ts1 = timeit(@() ltratio('cdf', x, cas, c, setfield(o,'method','single')));
    td1 = timeit(@() ltratio('cdf', x, cas, c, setfield(o,'method','double')));
    ltratio('threads', 10);
    ts10 = timeit(@() ltratio('cdf', x, cas, c, setfield(o,'method','single')));
    td10 = timeit(@() ltratio('cdf', x, cas, c, setfield(o,'method','double')));
    Fs = ltratio('cdf', x, cas, c, setfield(o,'method','single')); Fd = ltratio('cdf', x, cas, c, setfield(o,'method','double'));
    fprintf('%-18s %12.4f %12.4f %12.4f %12.4f %12.4f   %18.15f %18.15f  (MATLAB %18.15f)\n', sprintf('%s %g %g %g %g', cas, c, c0, g0, x), tm, ts1, td1, ts10, td10, Fs, Fd, Fm);
end
