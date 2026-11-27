#!/usr/bin/env python3
"""proftab.py FILES...: selected rows side by side, one column per file, plus Windows runs 23-26."""
import sys,csv,glob,os,statistics as st,collections
hdr="device,probe,variant,x,y,value,unit,ms".split(',')
def L(fs,h=None):
    d=collections.defaultdict(list)
    for f in fs:
        for r in csv.reader(open(f)):
            if r[0]=='device' or len(r)<7: continue
            d[(r[0],r[1],r[2],float(r[3]))].append(float(r[5]))
    return {k:st.median(v) for k,v in d.items()}
win=L(glob.glob(os.path.expanduser('~/workspace/zstrata/docs/data/runs/run2[3-6]-windows.csv')))
cols=[(os.path.basename(f).replace('prof-','').replace('.csv',''),L([f])) for f in sys.argv[1:]]
keys=[('cross','chain-long','gpu-kernel',1000),('cross','chain-long','gpu-kernel',1e6),('cross','chain-long','gpu-kernel',1e7),('cross','chains','gpu-kernel',1),('cross','chains','gpu-kernel',1<<20),('cross','chains','gpu-resident',1),('cross','nets','gpu-kernel',64),('cross','nets','gpu-kernel',4096),('cross','nets','gpu-resident',64),
('gpu','chase','scalar',4096),('gpu','chase','scalar',1<<20),('gpu','chase','vector',8<<20),('gpu','chase','scalar',128<<20),('gpu','chase','vector',128<<20),('gpu','chase','scalar',512<<20),('gpu','chase','vector',512<<20),('gpu','lds','read-fp32',1),
('link','roundtrip','empty-resubmit',0),('link','roundtrip','tiny-dispatch-readback',4),('link','wait','long-block-wall',0),('link','upload','staged',4096),('link','upload','staged',1<<20),('link','upload','staged',1<<30),('link','upload','direct-write',1<<30),('link','download','staged',1<<30),('link','download','staged',1<<20)]
print(f"{'row':52}{'windows':>10}"+''.join(f"{n:>11}" for n,_ in cols))
for k in keys:
    print(f"{k[1]+' '+k[2]+' '+format(k[3],'g'):52}{win.get(k,float('nan')):10.4g}"+''.join(f"{c.get(k,float('nan')):11.4g}" for _,c in cols))
