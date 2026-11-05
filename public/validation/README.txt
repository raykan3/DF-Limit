acc_table.m    the table "The two representations against each other" of the paper
               (tab:acc): the single integral against the double integral at seven
               points per configuration, with the median times; writes acc_table.log
               (every point) and acc_rows.tex (the LaTeX rows).  About 25 minutes.
acc_table.log, acc_rows.tex   the run behind the table in the paper
ltratio_test.log   a run of ltratio_test (81 checks, all passing)
ltratio_test_20261105_auto.log   the run after the automatic choice of the representation (114 checks, all passing)
method_sweep.m   per-point timings of the two representations in the MATLAB programs (one
               thread; c in [-5,5], c0 in {-5,-1,1,3}, x in [-4,32], the six cases, plus
               c0 = 0), the basis of the automatic choice in ltratio_method.m (November
               2026); writes method_sweep_<case>.log; method_sweep_summary.py aggregates
               the logs (method_sweep_summary.txt) and compares the rule with the C engine's
