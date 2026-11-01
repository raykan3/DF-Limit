tratio_legacy/ -- the limit computations of the earlier, combined version of the
finite-sample paper (DF-Test, Section 7 of that version), moved here when the
limit results became a paper of their own.  The scripts call the finite-sample
front ends with n = Inf (tratio_cdf(x,Inf,c,cas,...)), an interface that no
longer exists: they are kept with their logs as a record of the numbers printed
in that version (quantiles and local power, the Larsson comparison, the checks
of the double integral against finer rules and a one-dimensional inversion, the
sign conditions on the parabola, the moments and their checks), not as
runnable code.  The corresponding current scripts are ltratio_test.m,
ltratio_paper.m and acc_table.m.
