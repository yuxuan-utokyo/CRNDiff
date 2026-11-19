import numpy as np
from scipy.spatial import cKDTree
z=np.load(r"rebuttal\out\R1\r1_cells.npz"); c=np.load(r"rebuttal\out\R1\r1_clumps.npz")
xy,cl=c['ours_xy'],c['clump']; main=xy[cl==0]; t=cKDTree(main)
y=z['yval'].astype(str); vn=z['real_xy'][y=='Neuronal']; tr=np.load(r"rebuttal\out\R1\train_neuronal_umap.npy")
rv=vn[t.query(vn)[0]<=.5]; rt=tr[t.query(tr)[0]<=.5]
for n,p in [('OURS main pooled',main),('real val main',rv),('real train main',rt)]:
    print(n,len(p),'sd',np.round(p.std(0),3),'p1-p99 x',np.round(np.percentile(p[:,0],[1,99]),2),'y',np.round(np.percentile(p[:,1],[1,99]),2))
# upper arm: fixed by the real shape: above the block, y > 3.0
for n,p in [('OURS',main),('real val',rv),('real train',rt)]:
    print(n,'share y>2.9',round(float((p[:,1]>2.9).mean()),3))
# coverage: real main cells with an OURS cell within r, and vice versa
for r in (.05,.1,.2):
    print('r',r,'real train covered by OURS',round(float((t.query(rt)[0]<=r).mean()),3),'real val covered',round(float((t.query(rv)[0]<=r).mean()),3),'OURS covered by train',round(float((cKDTree(rt).query(main)[0]<=r).mean()),3))
