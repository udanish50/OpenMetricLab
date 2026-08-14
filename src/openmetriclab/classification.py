from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_auc_score,
    average_precision_score, log_loss,
)

def evaluate_classification(y_true, y_pred, probabilities=None, classes=None) -> dict:
    y=np.asarray(y_true).reshape(-1); p=np.asarray(y_pred).reshape(-1)
    if y.shape != p.shape: raise ValueError("y_true and y_pred must have the same shape")
    if y.size < 2: raise ValueError("At least two samples are required")
    labels=np.asarray(classes if classes is not None else np.unique(np.concatenate([y,p])))
    cm=confusion_matrix(y,p,labels=labels)
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    _,_,f1_macro,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    _,_,f1_weighted,_=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)
    out={
      'n':int(y.size),'classes':[str(x) for x in labels.tolist()],
      'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),
      'f1_macro':float(f1_macro),'f1_weighted':float(f1_weighted),
      'mcc':float(matthews_corrcoef(y,p)),'cohen_kappa':float(cohen_kappa_score(y,p)),
      'confusion_matrix':cm.astype(int).tolist(),
      'per_class':[{'class':str(c),'precision':float(a),'recall':float(b),'f1':float(d),'support':int(s)} for c,a,b,d,s in zip(labels,pr,rc,f1,sup)],
    }
    if probabilities is not None:
        probs=np.asarray(probabilities,dtype=float)
        if probs.ndim==1: probs=probs.reshape(-1,1)
        if probs.shape[0] != y.size: raise ValueError("probabilities row count must match labels")
        eps=1e-15; probs=np.clip(probs,eps,1-eps)
        if len(labels)==2 and probs.shape[1] in (1,2):
            pos=probs[:,0] if probs.shape[1]==1 else probs[:,1]
            probs2=np.c_[1-pos,pos]
            y01=(y==labels[1]).astype(int)
            out['roc_auc']=float(roc_auc_score(y01,pos))
            out['average_precision']=float(average_precision_score(y01,pos))
            out['brier']=float(np.mean((pos-y01)**2))
            out['log_loss']=float(log_loss(y01,probs2,labels=[0,1]))
        elif probs.shape[1]==len(labels):
            rowsum=probs.sum(axis=1,keepdims=True); probs=probs/np.maximum(rowsum,eps)
            onehot=np.zeros_like(probs); index={v:i for i,v in enumerate(labels.tolist())}
            for i,v in enumerate(y.tolist()): onehot[i,index[v]]=1
            out['roc_auc_ovr_macro']=float(roc_auc_score(y,probs,labels=labels,multi_class='ovr',average='macro'))
            out['average_precision_macro']=float(average_precision_score(onehot,probs,average='macro'))
            out['brier_multiclass']=float(np.mean(np.sum((probs-onehot)**2,axis=1)))
            out['log_loss']=float(log_loss(y,probs,labels=labels))
    return out
