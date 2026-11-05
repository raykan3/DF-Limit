# Summary of the method_sweep_*.log files: per point the faster representation in the MATLAB programs, aggregated by
# case, c, c0 and the sign of x, against the rule of the C engine (dfl_automethod: double iff c0 != 0, zero start and
# not (c0 > 0 and x >= 0)).  Usage: python3 method_sweep_summary.py [logs...]
import sys, glob, os, collections
D = os.path.dirname(os.path.abspath(__file__))
files = sys.argv[1:] or sorted(glob.glob(os.path.join(D, 'method_sweep_*.log')))
rows = []
for f in files:
    for line in open(f):
        if line.startswith('%'): continue
        a, b, c, d, e, s = [p.strip() for p in line.split('|')]
        cas, cc, c0, x = a.split(); F1, F2, dF = map(float, b.split()); t1, t2 = map(float, c.split())
        f1, f2 = map(float, d.split()); u1, u2 = map(float, e.split()); st = int(s)
        rows.append(dict(cas=cas, c=float(cc), c0=float(c0), x=float(x), F1=F1, F2=F2, dF=dF, t1=t1, t2=t2, f1=f1, f2=f2, u1=u1, u2=u2, st=st))
def crule(r): return 2 if (r['c0'] != 0 and not (r['c0'] > 0 and r['x'] >= 0)) else 1   # dfl_automethod: no start argument
def mrule(r):   # ltratio_method.m
    if r['c0'] == 0: return 1
    if r['cas'] in ('SU', 'ST'): return 2 if (r['c0'] < 0 and r['x'] >= 0) else 1
    return 2 if not (r['c0'] > 0 and r['x'] >= 0) else 1
print(f"{len(rows)} points from {len(files)} files")
print("\nPer point (cdf): t_single t_double ratio=tD/tS  winner  C-rule  [pdf: u_single u_double]  dF  status")
agg = collections.OrderedDict()
for r in rows:
    win = 1 if r['t1'] <= r['t2'] else 2
    flag = '' if win == crule(r) else ' <-- differs from C'
    bad = ' DF!' if abs(r['dF']) > 1e-9 else ''
    bad += ' st!' if r['st'] not in (0,) else ''
    print(f"{r['cas']} c={r['c']:5g} c0={r['c0']:4g} x={r['x']:5g}  {r['t1']:.3f} {r['t2']:.3f} {r['t2']/r['t1']:6.2f}  {'S' if win==1 else 'D'}  {'S' if crule(r)==1 else 'D'}   [{r['u1']:.3f} {r['u2']:.3f}]  {r['dF']:.1e} {r['st']}{bad}{flag}")
    key = (r['cas'], r['c0'], 'x<0' if r['x'] < 0 else 'x>=0')
    agg.setdefault(key, []).append(r)
print("\nAggregated by case, c0 and the sign of x (over c and x): points where the single wins / total, median ratio tD/tS, time range")
for key, rs in agg.items():
    ns = sum(1 for r in rs if r['t1'] <= r['t2']); n = len(rs)
    ratios = sorted(r['t2'] / r['t1'] for r in rs)
    med = ratios[len(ratios) // 2]
    rule = 'S' if crule(rs[0]) == 1 else 'D'
    print(f"{key[0]} c0={key[1]:4g} {key[2]:5s}: single wins {ns:2d}/{n:2d}, median tD/tS {med:5.2f}, single {min(r['t1'] for r in rs):.2f}-{max(r['t1'] for r in rs):.2f} s, double {min(r['t2'] for r in rs):.2f}-{max(r['t2'] for r in rs):.2f} s; C rule {rule}")
tot = lambda f: sum(f(r) for r in rows)
print("\nTotal time over all points (cdf): single %.1f s, double %.1f s, best-of %.1f s, C rule %.1f s, MATLAB rule (ltratio_method) %.1f s" % (
    tot(lambda r: r['t1']), tot(lambda r: r['t2']), tot(lambda r: min(r['t1'], r['t2'])), tot(lambda r: r['t1'] if crule(r) == 1 else r['t2']), tot(lambda r: r['t1'] if mrule(r) == 1 else r['t2'])))
print("Worst point under the MATLAB rule relative to the best: %.2f" % max((r['t1'] if mrule(r) == 1 else r['t2']) / min(r['t1'], r['t2']) for r in rows))
print("Points where the MATLAB rule and the C rule differ:", sum(1 for r in rows if mrule(r) != crule(r)), "of", len(rows))
