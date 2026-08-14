from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def _surface(mask):
    mask=np.asarray(mask,dtype=bool)
    if not np.any(mask): return mask
    return mask ^ binary_erosion(mask,structure=np.ones((3,)*mask.ndim),border_value=0)

def _surface_distances(a,b,spacing=None):
    sa,sb=_surface(a),_surface(b)
    if not np.any(sa) or not np.any(sb): return np.array([],float),np.array([],float)
    dt_b=distance_transform_edt(~sb,sampling=spacing)
    dt_a=distance_transform_edt(~sa,sampling=spacing)
    return dt_b[sa], dt_a[sb]

def binary_segmentation_metrics(y_true,y_pred,spacing=None) -> dict:
    a=np.asarray(y_true,dtype=bool); b=np.asarray(y_pred,dtype=bool)
    if a.shape!=b.shape: raise ValueError('Masks must have identical shapes')
    tp=int(np.sum(a&b)); fp=int(np.sum(~a&b)); fn=int(np.sum(a&~b)); tn=int(np.sum(~a&~b))
    def div(n,d): return float(n/d) if d else None
    dice=div(2*tp,2*tp+fp+fn); iou=div(tp,tp+fp+fn)
    d1,d2=_surface_distances(a,b,spacing)
    both=np.concatenate([d1,d2]) if d1.size and d2.size else np.array([],float)
    return {
      'pixels':int(a.size),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
      'dice':dice,'iou':iou,'precision':div(tp,tp+fp),'recall_sensitivity':div(tp,tp+fn),
      'specificity':div(tn,tn+fp),'pixel_accuracy':div(tp+tn,a.size),
      'balanced_accuracy':None if div(tp,tp+fn) is None or div(tn,tn+fp) is None else (div(tp,tp+fn)+div(tn,tn+fp))/2,
      'volume_similarity':div(2*(np.sum(a)-np.sum(b)),np.sum(a)+np.sum(b)),
      'hausdorff':float(np.max(both)) if both.size else None,
      'hd95':float(np.quantile(both,.95)) if both.size else None,
      'assd':float(np.mean(both)) if both.size else None,
    }

def evaluate_segmentation(y_true,y_pred,labels=None,spacing=None,include_background=False) -> dict:
    a=np.asarray(y_true); b=np.asarray(y_pred)
    if a.shape!=b.shape: raise ValueError('Masks must have identical shapes')
    labs=np.asarray(labels if labels is not None else np.unique(np.concatenate([a.reshape(-1),b.reshape(-1)])))
    rows=[]
    for lab in labs:
        if not include_background and lab==0: continue
        m=binary_segmentation_metrics(a==lab,b==lab,spacing=spacing); m['class']=str(lab); rows.append(m)
    keys=['dice','iou','precision','recall_sensitivity','specificity','pixel_accuracy','hd95','assd']
    macro={}
    for k in keys:
        vals=[r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        macro[k]=float(np.mean(vals)) if vals else None
    return {'shape':list(a.shape),'labels':[str(x) for x in labs.tolist()],'per_class':rows,'macro':macro,'include_background':bool(include_background)}
