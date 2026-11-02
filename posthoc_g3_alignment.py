"""POST HOC: cosine between each G3 arm's first-epoch direction and the exact descent direction,
at the G3 damping (0.006) and at low damping (0.0005), beds 200-201.  Run: PYTHONPATH=. python3 posthoc_g3_alignment.py"""
import numpy as np, sys
sys.argv=['x']
import importlib.util
src=open('g3_router.py').read().split('ARMS = ')[0]   # definitions only, no runs
for damp in (0.006, 0.0005):
    ns={}
    exec(src.replace('bch = Beach()', f'bch = Beach(damp={damp})'), ns)
    bch, direction, FREQS, H, srcp = ns['bch'], ns['direction'], ns['FREQS'], ns['H'], ns['src']
    from beach.medium import rough_bed
    for seed in (200,201):
        s0=rough_bed(bch,seed); P=[];sig={}
        for q in (0.0,0.5):
            P=[];sigc=[]
            for w,r,wr in FREQS:
                eta=bch.solve(s0,w,srcp); P.append(abs(H[r]@eta)**2+abs(H[wr]@eta)**2); sigc.append(float(np.quantile(bch.stress(eta),q)) if q>0 else 0.0)
            g=direction('gd',s0,P,sigc)
            for arm in ('teacher_scour','teacher_scour_one','natural'):
                d=direction(arm,s0,P,sigc)
                print(damp,seed,'q',q,arm,'cos to -grad %.3f'%(d@g/np.linalg.norm(d)/np.linalg.norm(g)))
