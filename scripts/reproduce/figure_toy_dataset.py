"""Redraw only the frozen toy construction; no sampling or model execution."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TOY_DATA = REPO / "assets" / "toy" / "data"
TOY_RUNS = REPO / "results" / "toy" / "runs"
TOY_FIG  = REPO / "results" / "toy"
FIG_OUT  = REPO / "out" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)
OUT = FIG_OUT
DATA = TOY_DATA
META = json.loads((TOY_FIG / 'figmeta.json').read_text(encoding='utf-8'))
STEM = META['data_stem']
CFG_PATH = DATA / (STEM + '.json')
PMF_PATH = DATA / (STEM + '_pmf.npz')
cfg = json.loads(CFG_PATH.read_text(encoding='utf-8'))
z = np.load(PMF_PATH, allow_pickle=False)
comp, mix = z['components'], z['mixture']
names = cfg['components']
prior = np.asarray(cfg['priors'])
C = cfg['C']
g = np.arange(C)

def request(key):
    spec = cfg['requests'][key]
    idx = [names.index(n) for n in spec['components']]
    return np.einsum('i,ijk->jk', np.asarray(spec['weights']), comp[idx])

pR, pT = request('diag_rare'), request('diag_twin')
for p in (mix, pR, pT):
    assert np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
np.testing.assert_allclose(mix, np.einsum('i,ijk->jk', prior, comp), atol=1e-14, rtol=0)
tv = float(.5 * np.abs(pR-pT).sum())
delta = float(max(np.abs(pR.sum(a)-pT.sum(a)).max() for a in (0,1)))
assert delta < 1e-12, delta

import sys
sys.path.insert(0, str(REPO))
from toy2d.metrics import responsibilities
BG_PATH = TOY_FIG / 'figdata.npz'
bg = np.load(BG_PATH, allow_pickle=False)['bg']
labels = responsibilities(comp, prior, bg).argmax(1)
cross_dim_delta = float(max(np.abs(p.sum(0)-p.sum(1)).max() for p in (pR,pT)))
assert cross_dim_delta < 1e-12, cross_dim_delta
ORANGE, BLUE, GREY = '#eb6834', '#6aa6ee', '#b8c1cd'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.size':3,'ytick.major.size':3,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.2,2.8),facecolor='white')
panel_height = 2.05 / 2.8
ax=fig.add_axes([.45/7.2,.17,2.7/7.2,panel_height])
am=fig.add_axes([3.9/7.2,.17,3.15/7.2,panel_height])
for indices,col in (([0,2,4,6],GREY),([3,7],BLUE),([1,5],ORANGE)):
    sel=np.isin(labels,indices)
    ax.scatter(bg[sel,0],bg[sel,1],s=2.8,color=col,alpha=.5,linewidths=0,rasterized=True)
for i,nm in enumerate(names):
    x,y=cfg['clusters']['centers'][nm]
    v=np.array([x,y])-np.asarray(cfg['center'])
    pos=np.array([x,y])+v/np.linalg.norm(v)*8
    col=ORANGE if i in (1,5) else BLUE if i in (3,7) else '#66717d'
    ax.text(*pos,'$c_'+str(i)+'$',ha='center',va='center',fontsize=9,color=col)
display_y_span=70.0
display_x_span=display_y_span*2.7/2.05
ax.set(xlim=(cfg['center'][0]-display_x_span/2,cfg['center'][0]+display_x_span/2),
       ylim=(0,display_y_span),aspect='equal',
       xticks=[0,20,40,60],yticks=[0,20,40,60],xlabel='Dim 0',ylabel='Dim 1')
ax.set_title('(A) Data distribution',fontsize=10,pad=8)
am.plot(g,pR.sum(1),color=ORANGE,lw=2.5)
am.plot(g,pT.sum(1),color=BLUE,lw=1.6,ls=(0,(3,2)))
am.set(xlim=(-.5,C-.5),ylim=(0,float(pR.sum(1).max())*1.22),
       xticks=[0,20,40,60],yticks=[0,.05,.1],xlabel='Dim 0 / Dim 1',ylabel='Probability')
am.set_title('(B) Shared marginal',fontsize=10,pad=8)
am.tick_params(pad=3)
point_handles=[Line2D([],[],color=c,marker='o',ls='',markersize=5)
               for c in (ORANGE,BLUE,GREY)]
line_handles=[Line2D([],[],color=ORANGE,lw=2),
              Line2D([],[],color=BLUE,lw=1.6,ls=(0,(3,2)))]
legend_options=dict(loc='upper center',frameon=True,fontsize=7.5,
                    facecolor='white',edgecolor='#d5d5d0',framealpha=1,
                    fancybox=True,handlelength=1.5,borderaxespad=.35,
                    borderpad=.45,labelspacing=.35)
legend_a=ax.legend(point_handles,['Request','Twin','Others'],ncol=3,
          columnspacing=.65,handletextpad=.35,**legend_options)
legend_b=am.legend(line_handles,[r'Request: $c_1,c_5$',r'Twin: $c_3,c_7$'],ncol=2,
          columnspacing=.8,handletextpad=.4,**legend_options)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
for artist in fig.findobj(matplotlib.text.Text):
    if not artist.get_visible() or not artist.get_text():continue
    box=artist.get_window_extent(renderer)
    if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
        outside.append(artist.get_text())
assert not outside,outside
np.testing.assert_allclose(ax.bbox.height,am.bbox.height,atol=1e-8,rtol=0)
np.testing.assert_allclose(ax.title.get_window_extent(renderer).y0,
                           am.title.get_window_extent(renderer).y0,atol=1e-8,rtol=0)
legend_intersections=[]
for axis,legend,points in ((ax,legend_a,bg),
        (am,legend_b,np.column_stack([g,pR.sum(1)]))):
    box=legend.get_window_extent(renderer)
    xy=axis.transData.transform(points)
    count=int(((xy[:,0]>=box.x0)&(xy[:,0]<=box.x1)&
               (xy[:,1]>=box.y0)&(xy[:,1]<=box.y1)).sum())
    legend_intersections.append(count)
assert legend_intersections==[0,0],legend_intersections
backup=OUT/'_archive_20260910_three_panel'
backup.mkdir(exist_ok=True)
for ext in ('pdf','png','svg','json'):
    old=OUT/('figure_toy_dataset.'+ext)
    if old.exists() and not (backup/old.name).exists():shutil.copy2(old,backup/old.name)
for ext in ('pdf','png','svg'):
    fig.savefig(OUT/('figure_toy_dataset.'+ext),dpi=450,facecolor='white')
plt.close(fig)
caption=(r'\textbf{Toy data and shared marginals.} '
    r'(A) A fixed subset of training samples on the $64\times64$ count grid. '
    r'Orange identifies the requested components $\{c_1,c_5\}$, blue their marginal twin $\{c_3,c_7\}$, '
    r'and grey the remaining components; colours use maximum posterior component responsibility. '
    r'Each axial component has prior mass $0.245$ and each diagonal component $0.005$. '
    r'(B) Exact normalized marginals of the equal-weight request and twin laws. '
    r'Both coordinates share the same marginal, so a single overlay suffices. '
    r'The two laws occupy different pairs of modes in two dimensions despite identical one-dimensional marginals.')
(OUT/'figure_toy_dataset_caption.tex').write_text(caption+'\n',encoding='utf-8')
report={'data_stem':STEM,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (CFG_PATH,PMF_PATH,BG_PATH)},
    'joint_TV':tv,'max_marginal_absolute_difference':delta,'max_cross_dimension_difference':cross_dim_delta,
    'prior_masses':prior.tolist(),'n_displayed_training_points':len(bg),
    'argmax_component_counts':np.bincount(labels,minlength=len(names)).tolist(),
    'rendering':{'left':'archived training subset, no jitter, no new draw, fixed marker size and alpha',
                 'right':'exact normalized marginals; dim0 and dim1 verified identical'},
    'dimensions_inches':[7.2,2.8],'panel_widths_inches':[2.7,3.15],
    'legend_style':{'font_size_pt':7.5,'location':'upper center','frame':'white with light grey border'},
    'qa':{'text_outside_canvas':outside,'pmf_normalization_passed':True,
       'equal_panel_heights':True,'aligned_title_baselines':True,
       'legend_data_intersections':legend_intersections},
    'caption':caption,'experiments_rerun':False}
(OUT/'figure_toy_dataset.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))

