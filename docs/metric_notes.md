# Metric interpretation and edge-case policies

OpenMetricLab computes the metrics directly from their mathematical definitions. No external metric library is called.

## Regression
- MAE/MSE/RMSE use finite truth-prediction pairs only.
- R² is reported as undefined (`None`) when the ground-truth total sum of squares is zero.
- Explained variance is undefined when ground-truth population variance is zero.
- MAPE excludes exactly-zero/near-zero targets and reports `mape_coverage`; OpenMetricLab never hides the problem by adding an arbitrary epsilon to the target.
- sMAPE uses `2|y-p|/(|y|+|p|)` and defines a `0/0` term as zero.

## Classification
- Balanced accuracy is the unweighted mean of per-class recall.
- Macro-F1 is the unweighted mean of per-class F1; weighted-F1 weights by support.
- Multiclass MCC uses the confusion-matrix generalization.
- Cohen's κ uses empirical marginal class frequencies.
- ROC points are evaluated after **groups of tied scores**, so AUC is invariant to row ordering within a tie.
- Average precision is the step integral of precision over increases in recall at distinct score thresholds.
- Probability rows must be finite and non-negative. Rows not summing to 1 are normalized explicitly and this is disclosed in `probabilities_normalized`.

## Segmentation
- Dice/IoU are computed class-wise on exact masks.
- If both class masks are empty, Dice/IoU are defined as 1 and surface distances as 0 (`surface_status=both_empty`).
- If only one class mask is empty, Dice/IoU are 0 and geometric surface distances are undefined (`surface_status=one_empty`).
- Python surface distances are exact nearest-neighbour Euclidean distances between axial boundary pixels/voxels, respecting physical spacing.
- HD95 is the 95th percentile of the combined directed surface-distance samples; ASSD is their mean.
- The browser may deterministically sample exceptionally large boundaries for responsiveness and labels that condition in the report.

## Registration
- MSE/RMSE/NRMSE, PSNR and normalized cross-correlation are direct array computations.
- SSIM is computed locally from luminance, variance and covariance terms using the SSIM equation.
- NMI uses the Studholme-style `(H(A)+H(B))/H(A,B)` histogram definition.
- Image similarity does not prove geometric correctness. Landmark TRE, when available, addresses a different question and should be reported alongside intensity similarity.
