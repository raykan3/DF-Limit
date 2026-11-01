function T = ltratio_paper(which)
%LTRATIO_PAPER  Reproduces the tables of the paper.
%
%   LTRATIO_PAPER reproduces, and prints next to the published values, the
%   tables of Bao, Broda and Kan, "The Exact Limiting Distribution of the
%   Dickey-Fuller t-Ratio" (LaTeX labels in parentheses):
%      'quant'    quantiles of the three limits at c = 0 and c = -5, and the
%                 asymptotic critical values of MacKinnon (2010)   (tab:quant)
%      'larsson'  the limiting distribution of tau under local alternatives
%                 against Larsson (1995), panels A and B           (tab:larsson)
%      'init'     limiting power of the 5% tests with a fixed start of xi
%                 stationary standard deviations and with a stationary
%                 start                                            (tab:init)
%      'mom'      mean and standard deviation under local alternatives,
%                 zero and stationary start                        (tab:mom)
%   LTRATIO_PAPER(which) does one of them (a string or a cell array of
%   strings); the default is all four, about ten minutes.  T = LTRATIO_PAPER(...)
%   returns a structure with the computed arrays.  The table of the two
%   representations against each other (tab:acc) is produced by
%   validation/acc_table.m, which writes its rows as LaTeX.
%
%   See also LTRATIO_CDF, LTRATIO_ICDF, LTRATIO_MOM, LTRATIO_TEST.
if nargin < 1 || isempty(which), which = {'quant','larsson','init','mom'}; end
if ischar(which), which = {which}; end
cs = {'DK','DU','DT'}; nm = {'tau','tau_mu','tau_tau'};
T = struct;
t0 = tic;
for w = which
    switch lower(w{1})
        case 'quant'
            fprintf('\n=== Table quant: quantiles of the limits (zero start, c0 = 0) ===\n');
            p = [0.01 0.025 0.05 0.10 0.25 0.50 0.90 0.95 0.975 0.99];
            mck = {[-2.56574 NaN -1.94100 -1.61682], [-3.43035 NaN -2.86154 -2.56677], [-3.95877 NaN -3.41049 -3.12705]};
            pub = {[-2.5658 -2.2274 -1.9409 -1.6167 -1.0913 -0.5001 0.8875 1.2831 1.6230 2.0145; -3.2587 -2.9735 -2.7368 -2.4751 -2.0674 -1.6623 -1.0126 -0.8321 -0.6579 -0.4310], ...
                   [-3.4304 -3.1221 -2.8616 -2.5668 -2.0874 -1.5664 -0.4403 -0.0785 0.2380 0.6067; -3.7216 -3.4345 -3.1943 -2.9258 -2.4999 -2.0632 -1.3233 -1.1126 -0.9143 -0.6634], ...
                   [-3.9587 -3.6614 -3.4105 -3.1270 -2.6685 -2.1806 -1.2464 -0.9400 -0.6597 -0.3251; -4.1316 -3.8452 -3.6044 -3.3338 -2.9002 -2.4477 -1.6594 -1.4348 -1.2267 -0.9653]};
            fprintf('%-8s %4s', '', 'c'); fprintf(' %8.3g', p); fprintf('\n');
            Q = zeros(3,2,numel(p));
            for j = 1:3
                for ic = 1:2
                    c = [0 -5]; c = c(ic);
                    q = ltratio_icdf(p,cs{j},c); Q(j,ic,:) = q;
                    fprintf('%-8s %4g', nm{j}, c); fprintf(' %8.4f', q); fprintf('   (computed)\n');
                    fprintf('%-8s %4s', '', ''); fprintf(' %8.4f', pub{j}(ic,:)); fprintf('   (paper)\n');
                    if c == 0
                        fprintf('%-8s %4s', 'M10', ''); fprintf(' %8.4f', mck{j}(1)); fprintf(' %8s %8.4f %8.4f', '', mck{j}(3), mck{j}(4)); fprintf('   (MacKinnon 2010, 1%%, 5%%, 10%%)\n');
                    end
                end
            end
            T.quant = Q;
        case 'larsson'
            fprintf('\n=== Table larsson: P[tau <= z] under phi = e^(beta/n), case DK, zero start ===\n');
            zA = [-2 -1.5 -1 0 0.5 1 2 3 5 8];
            pubA = [0.026405 0.015333; 0.075778 0.043064; 0.171927 0.094033; 0.424176 0.215288; 0.545650 0.277173; 0.661230 0.342599; 0.845905 0.479265; 0.947926 0.611544; 0.997702 0.818787; 0.999998 0.962540];
            fprintf('A. Dickey-Fuller test (c = beta, c0 = 0)\n   z     beta=1 (computed, paper)      beta=2 (computed, paper)\n');
            A = zeros(numel(zA),2);
            for k = 1:numel(zA)
                A(k,1) = ltratio_cdf(zA(k),'DK',1); A(k,2) = ltratio_cdf(zA(k),'DK',2);
                fprintf('%5.1f   %.6f %.6f            %.6f %.6f\n', zA(k), A(k,1), pubA(k,1), A(k,2), pubA(k,2));
            end
            zB = [-2 -1.5 -1 0 1 1.5 2 3];
            pubB = [0.039149 0.074729; 0.109743 0.191272; 0.244738 0.322495; 0.647192 0.650436; 0.917900 0.901895; 0.969410 0.960805; 0.990543 0.987266; 0.999515 0.999303];
            fprintf('B. t-ratio centered at the true value (c = c0 = beta)\n   z     beta=-2 (computed, paper)     beta=2 (computed, paper)\n');
            B = zeros(numel(zB),2);
            for k = 1:numel(zB)
                B(k,1) = ltratio_cdf(zB(k),'DK',-2,struct('c0',-2)); B(k,2) = ltratio_cdf(zB(k),'DK',2,struct('c0',2));
                fprintf('%5.1f   %.6f %.6f            %.6f %.6f\n', zB(k), B(k,1), pubB(k,1), B(k,2), pubB(k,2));
            end
            T.larssonA = A; T.larssonB = B;
        case 'init'
            fprintf('\n=== Table init: limiting power of the 5%% tests and the initial observation ===\n');
            cv = [-1.94087325 -2.86156886 -3.41045363];
            cc = [-30 -20 -15 -10 -5 -2.5]; xi = [0 1 2 3];
            pub = {[1.000 0.998 0.967 0.758 0.318 0.144; 1.000 0.999 0.974 0.792 0.361 0.173; 1.000 0.999 0.988 0.877 0.499 0.277; 1.000 1.000 0.997 0.958 0.720 0.499; 1.000 0.998 0.973 0.789 0.362 0.176], ...
                   [0.995 0.844 0.593 0.304 0.117 0.072; 0.996 0.860 0.619 0.325 0.124 0.074; 0.998 0.901 0.691 0.391 0.152 0.081; 0.999 0.948 0.795 0.507 0.208 0.098; 0.996 0.859 0.618 0.325 0.125 0.074], ...
                   [0.940 0.606 0.366 0.182 0.083 0.059; 0.945 0.621 0.379 0.188 0.084 0.059; 0.959 0.666 0.417 0.206 0.086 0.058; 0.975 0.734 0.483 0.241 0.090 0.057; 0.945 0.621 0.379 0.188 0.084 0.059]};
            P = zeros(3,5,numel(cc));
            fprintf('%-8s %-12s', '', 'start'); fprintf(' %7g', cc); fprintf('\n');
            for j = 1:3
                for r = 1:5
                    for k = 1:numel(cc)
                        if r <= 4
                            g0 = xi(r)/sqrt(-2*cc(k));
                            P(j,r,k) = ltratio_cdf(cv(j),cs{j},cc(k),struct('g0',g0));
                        else
                            P(j,r,k) = ltratio_cdf(cv(j),['S' cs{j}(2)],cc(k));
                        end
                    end
                    if r <= 4, lab = sprintf('xi=%d',xi(r)); else, lab = 'stationary'; end
                    fprintf('%-8s %-12s', nm{j}, lab); fprintf(' %7.3f', squeeze(P(j,r,:))); fprintf('   (computed)\n');
                    fprintf('%-8s %-12s', '', ''); fprintf(' %7.3f', pub{j}(r,:)); fprintf('   (paper)\n');
                end
            end
            T.init = P;
        case 'mom'
            fprintf('\n=== Table mom: mean and standard deviation under local alternatives ===\n');
            cc = [0 -2.5 -5 -10 -20];
            pub = {[-0.4231 -1.2636 -1.7066 -2.3352 -3.2366; 0.9811 0.6548 0.5867 0.5461 0.5239; NaN -1.3590 -1.7775 -2.3874 -3.2747; NaN 0.6500 0.5883 0.5490 0.5261], ...
                   [-1.5330 -1.8373 -2.0960 -2.5924 -3.4084; 0.8403 0.6949 0.6381 0.5836 0.5458; NaN -1.8347 -2.1226 -2.6279 -3.4404; NaN 0.7100 0.6424 0.5853 0.5473], ...
                   [-2.1814 -2.2927 -2.4738 -2.8644 -3.5905; 0.7499 0.7086 0.6640 0.6110 0.5652; NaN -2.2862 -2.4731 -2.8779 -3.6118; NaN 0.7133 0.6684 0.6129 0.5664]};
            Mm = zeros(3,4,numel(cc));
            fprintf('%-8s %-12s', '', 'start'); fprintf(' %8g', cc); fprintf('\n');
            for j = 1:3
                for k = 1:numel(cc)
                    m = ltratio_mom(2,cs{j},cc(k)); Mm(j,1,k) = m(1); Mm(j,2,k) = sqrt(m(2)-m(1)^2);
                    if cc(k) < 0, m = ltratio_mom(2,['S' cs{j}(2)],cc(k)); Mm(j,3,k) = m(1); Mm(j,4,k) = sqrt(m(2)-m(1)^2); else, Mm(j,3:4,k) = NaN; end
                end
                labs = {'zero mean','zero sd','stat. mean','stat. sd'};
                for r = 1:4
                    fprintf('%-8s %-12s', nm{j}, labs{r}); fprintf(' %8.4f', squeeze(Mm(j,r,:))); fprintf('   (computed)\n');
                    fprintf('%-8s %-12s', '', ''); fprintf(' %8.4f', pub{j}(r,:)); fprintf('   (paper)\n');
                end
            end
            T.mom = Mm;
        otherwise
            error('ltratio_paper: unknown table %s', w{1});
    end
end
fprintf('\n(%.0f s)\n', toc(t0));
end
