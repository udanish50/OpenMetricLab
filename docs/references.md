# Metric definitions and implementation provenance

OpenMetricLab v0.2.0 does **not** call scikit-learn, SciPy, scikit-image, MONAI, OpenCV, or another metric package to calculate scores. The formulas are implemented in OpenMetricLab itself.

The implementation follows standard definitions for:
- regression error and coefficient-of-determination families;
- confusion-matrix precision/recall/F1, balanced accuracy, Cohen's κ, multiclass MCC;
- ROC-AUC, precision-recall average precision, log loss and Brier scores;
- Dice/Jaccard overlap and symmetric surface-distance summaries;
- SSIM, PSNR, normalized cross-correlation and normalized mutual information;
- target registration error (TRE).

Reference libraries may be useful for independent verification during research, but they are intentionally not runtime dependencies or implementation backends.
