import numpy as np
from openmetriclab import evaluate_regression,evaluate_classification,evaluate_segmentation,evaluate_registration,landmark_tre

def test_regression_perfect():
    r=evaluate_regression([1,2,3],[1,2,3]); assert r['rmse']==0 and r['r2']==1

def test_classification_perfect():
    r=evaluate_classification([0,1,1],[0,1,1],[[.9,.1],[.1,.9],[.2,.8]],[0,1]); assert r['accuracy']==1

def test_segmentation_perfect():
    a=np.zeros((16,16),int); a[4:12,4:12]=1
    r=evaluate_segmentation(a,a); assert r['macro']['dice']==1 and r['macro']['iou']==1

def test_registration_perfect():
    a=np.arange(64,dtype=float).reshape(8,8)
    r=evaluate_registration(a,a); assert r['rmse']==0 and r['ssim']==1

def test_tre():
    r=landmark_tre([[0,0],[1,1]],[[3,4],[1,1]]); assert abs(r['mean_tre']-2.5)<1e-12
