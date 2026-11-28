#!/usr/bin/env python3
"""ceiling.py WIN_RUNS... -- LINUX_RUNS... : is the Linux work at its ceiling target (docs/PLAN.md,
"Ceiling target")? Exit 0 = target met, 1 = open gaps remain.

A row is a GAP when the Windows median is more than ROW_LIM better AND the two sets of runs do not
overlap. A probe/variant is a GAP when its median Windows advantage exceeds GROUP_LIM. A probe/variant
with fewer than 3 rows is judged by its rows only. A gap is accepted when
docs/ceiling-exceptions.tsv names it with a cause outside the code (power, driver, permission) or
as `tried`: one bounded experiment was run and recorded in docs/measurements.md without closing
it. Everything else is OPEN. Rows where Linux is ahead are never work items."""
import sys, csv, statistics as st, collections, os, fnmatch
ROW_LIM, GROUP_LIM = 1.10, 1.05
OUTSIDE = {'power', 'driver', 'permission', 'tried'}
a = sys.argv[1:]; i = a.index('--'); W, L = a[:i], a[i + 1:]
def load(files):
    d = collections.defaultdict(list); unit = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            k = (r['device'], r['probe'], r['variant'], float(r['x']), float(r['y']))
            d[k].append(float(r['value'])); unit[k] = r['unit']
    return d, unit
w, _ = load(W); l, lu = load(L)
lower = {'ns', 'us', 'ms', 's'}
exc = []
ef = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'ceiling-exceptions.tsv')
for r in csv.reader(open(ef), delimiter='\t'):
    if r and not r[0].startswith('#'): exc.append(r)  # pattern  cause  reason
def accepted(k):
    name = '/'.join(k[:3])
    for pat, cause, why in exc:
        if fnmatch.fnmatch(name, pat): return cause, why
    return None
rows, groups = [], collections.defaultdict(list)
for k in l:
    if k not in w or k[1] == 'check': continue
    wm, lm = st.median(w[k]), st.median(l[k])
    if wm <= 0 or lm <= 0: continue
    adv = lm / wm if lu[k] in lower else wm / lm
    overlap = not (min(w[k]) > max(l[k]) or max(w[k]) < min(l[k]))
    groups[k[:3]].append(adv)
    if adv > ROW_LIM and not overlap: rows.append((adv, k))
gg = [(st.median(v), g, len(v)) for g, v in groups.items() if st.median(v) > GROUP_LIM and len(v) >= 3]
med = st.median(x for v in groups.values() for x in v)
print(f"{len(W)} Windows runs, {len(L)} Linux runs, {sum(map(len, groups.values()))} rows; "
      f"median Windows advantage {med:.3f}")
open_ = set()
print(f"\n== probe/variant medians over {GROUP_LIM}x")
for m, g, n in sorted(gg, reverse=True):
    a_ = accepted(g + (0, 0))
    if a_ is None or a_[0] not in OUTSIDE: open_.add(g)
    print(f"{m:5.2f}x n={n:<3} {'/'.join(g):32} {('ACCEPTED ' + a_[0] + ': ' + a_[1]) if a_ else 'OPEN'}")
print(f"\n== rows over {ROW_LIM}x beyond run-to-run overlap: {len(rows)}")
byg = collections.Counter(k[:3] for _, k in rows)
for g, n in byg.most_common():
    a_ = accepted(g + (0, 0))
    if a_ is None or a_[0] not in OUTSIDE: open_.add(g)
    print(f"  {n:3} rows {'/'.join(g):32} {('ACCEPTED ' + a_[0]) if a_ else 'OPEN'}")
print(f"\nceiling target: {'MET' if not open_ else f'NOT MET, {len(open_)} open probe/variant(s)'}")
sys.exit(bool(open_))
