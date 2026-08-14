from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error,
    r2_score, explained_variance_score, max_error,
)

def evaluate_regression(y_true, y_pred) -> dict:
    y=np.asarray(y_true,dtype=float).reshape(-1)
    p=np.asarray(y_pred,dtype=float).reshape(-1)
    if y.shape != p.shape: raise ValueError("y_true and y_pred must have the same shape")
    ok=np.isfinite(y)&np.isfinite(p)
    y,p=y[ok],p[ok]
    if y.size < 2: raise ValueError("At least two finite pairs are required")
    e=p-y; ae=np.abs(e)
    denom=np.abs(y)+np.abs(p)
    smape=float(100*np.mean(np.where(denom>0, 2*ae/denom, 0.0)))
    nz=np.abs(y)>1e-12
    mape=float(100*np.mean(ae[nz]/np.abs(y[nz]))) if np.any(nz) else None
    wape=float(100*np.sum(ae)/np.sum(np.abs(y))) if np.sum(np.abs(y))>1e-12 else None
    pear=float(pearsonr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else None
    spear=float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else None
    mse=float(mean_squared_error(y,p))
    return {
        'n':int(y.size),'mae':float(mean_absolute_error(y,p)),'mse':mse,
        'rmse':float(np.sqrt(mse)),'median_ae':float(median_absolute_error(y,p)),
        'max_error':float(max_error(y,p)),'r2':float(r2_score(y,p)),
        'explained_variance':float(explained_variance_score(y,p)),
        'bias':float(np.mean(e)),'smape_percent':smape,'mape_percent':mape,
        'mape_coverage':float(np.mean(nz)),'wape_percent':wape,
        'pearson_r':pear,'spearman_rho':spear,
        'error_p50':float(np.quantile(ae,.50)),'error_p90':float(np.quantile(ae,.90)),
        'error_p95':float(np.quantile(ae,.95)),'error_p99':float(np.quantile(ae,.99)),
    }
