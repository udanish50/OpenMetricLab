from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity, normalized_mutual_information

def _gray(x):
    a=np.asarray(x,dtype=float)
    if a.ndim==3: a=np.mean(a[...,:3],axis=-1)
    return a

def evaluate_registration(fixed,registered,moving=None,data_range=None) -> dict:
    a=_gray(fixed); b=_gray(registered)
    if a.shape!=b.shape: raise ValueError('Fixed and registered images must have identical shapes')
    if data_range is None:
        data_range=float(max(a.max(),b.max())-min(a.min(),b.min())) or 1.0
    e=b-a; mse=float(np.mean(e*e)); rmse=float(np.sqrt(mse))
    denom=float(np.sqrt(np.mean(a*a)))
    nrmse=rmse/denom if denom>0 else None
    psnr=float(20*np.log10(data_range)-10*np.log10(mse)) if mse>0 else float('inf')
    av=a-a.mean(); bv=b-b.mean(); ncc=float(np.sum(av*bv)/np.sqrt(np.sum(av*av)*np.sum(bv*bv))) if np.sum(av*av)>0 and np.sum(bv*bv)>0 else None
    ssim=float(structural_similarity(a,b,data_range=data_range))
    nmi=float(normalized_mutual_information(a,b,bins=64))
    out={'mse':mse,'rmse':rmse,'nrmse':nrmse,'psnr_db':psnr,'ssim':ssim,'ncc':ncc,'nmi':nmi}
    if moving is not None:
        before=evaluate_registration(a,moving,None,data_range=data_range)
        out['before']=before
        def gain(beforev,afterv,higher):
            if beforev is None or afterv is None: return None
            return float(afterv-beforev) if higher else float(beforev-afterv)
        out['improvement']={'ssim':gain(before.get('ssim'),ssim,True),'ncc':gain(before.get('ncc'),ncc,True),'nmi':gain(before.get('nmi'),nmi,True),'rmse':gain(before.get('rmse'),rmse,False)}
    return out

def landmark_tre(fixed_xy,registered_xy,spacing=(1.0,1.0)) -> dict:
    a=np.asarray(fixed_xy,dtype=float); b=np.asarray(registered_xy,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2: raise ValueError('Landmarks must be Nx2 arrays')
    sp=np.asarray(spacing,dtype=float).reshape(1,2)
    d=np.sqrt(np.sum(((b-a)*sp)**2,axis=1))
    return {'n':int(len(d)),'mean_tre':float(np.mean(d)),'median_tre':float(np.median(d)),'rmse_tre':float(np.sqrt(np.mean(d*d))),'p95_tre':float(np.quantile(d,.95)),'max_tre':float(np.max(d))}
