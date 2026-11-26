#!/usr/bin/env python3
"""cmp.py WIN_RUNS... -- LINUX_RUNS...  : per-row medians, ratio = how much better Windows is (>1 = Windows better)."""
import sys, csv, statistics as st, collections
a = sys.argv[1:]; i = a.index('--'); W, L = a[:i], a[i+1:]
def load(files):
    d = collections.defaultdict(list); unit = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            k = (r['device'], r['probe'], r['variant'], float(r['x']), float(r['y']))
            d[k].append(float(r['value'])); unit[k] = r['unit']
    return d, unit
w, wu = load(W); l, lu = load(L)
lower = {'ns', 'us', 'ms', 's'}
rows = []
for k in l:
    if k not in w or k[1] == 'check': continue
    wm, lm = st.median(w[k]), st.median(l[k])
    if wm <= 0 or lm <= 0: continue
    u = lu[k]
    adv = lm / wm if u in lower else wm / lm
    # overlap test
    ov = not (min(w[k]) > max(l[k]) or max(w[k]) < min(l[k]))
    rows.append((adv, k, u, wm, lm, ov, min(l[k]), max(l[k]), min(w[k]), max(w[k])))
rows.sort(reverse=True)
mode = 'all'
print(f"rows compared {len(rows)}; linux-only {sum(1 for k in l if k not in w)}; windows-only {sum(1 for k in w if k not in l)}")
byp = collections.defaultdict(list)
for r in rows: byp[(r[1][0], r[1][1], r[1][2])].append(r[0])
print("\n== per probe/variant: median windows advantage (n rows, min..max)")
for k, v in sorted(byp.items(), key=lambda kv: -st.median(kv[1])):
    print(f"{k[0]:5} {k[1]:10} {k[2]:22} {st.median(v):6.2f}x  n={len(v):3}  {min(v):.2f}..{max(v):.2f}")
if '-v' in sys.argv[0:1] or True:
    print("\n== rows where sets do not overlap and |adv-1|>10%")
    for adv, k, u, wm, lm, ov, lmin, lmax, wmin, wmax in rows:
        if ov or abs(adv - 1) < 0.10: continue
        print(f"{adv:6.2f}x {k[0]:5} {k[1]:10} {k[2]:22} x={k[3]:<12g} y={k[4]:<6g} win {wm:10.4g} [{wmin:.4g}..{wmax:.4g}]  lin {lm:10.4g} [{lmin:.4g}..{lmax:.4g}] {u}")
