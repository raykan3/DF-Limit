acc_table.m    the tables "The two representations against each other" of the Supplementary Material
               (tab:acc: zero and stationary start; tab:accfixed: fixed start): the line
               integral against the conditional integral at seven points per
               configuration, with the median times; writes acc_table.log (every point)
               and acc_rows.tex (the LaTeX rows; the fixed-start rows carry gamma_0 in
               the label).  About 25 minutes on the MATLAB programs, 5 on the C MEX file.
acc_table.log, acc_rows.tex   the run behind the tables in the paper (the C MEX file, 6 Nov 2026,
               after the agreement rule of the inner inversions); acc_table_20261103.log,
               acc_rows_20261103.tex the run of 3 Nov, with the 1e-8 cell of issue #4
ltratio_test.log   a run of ltratio_test (81 checks, all passing)
ltratio_test_20261105_auto.log   the run after the automatic choice of the representation (114 checks, all passing)
ltratio_test_20261106_agree.log  the run after the agreement rule of the inner inversions (tr_limdist, agreepaths)
method_sweep.m   per-point timings of the two representations in the MATLAB programs (one
               thread; c in [-5,5], c0 in {-5,-1,1,3}, x in [-4,32], the six cases, plus
               c0 = 0), the basis of the automatic choice in ltratio_method.m (November
               2026); writes method_sweep_<case>.log; method_sweep_summary.py aggregates
               the logs (method_sweep_summary.txt) and compares the rule with the C engine's
