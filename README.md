# OpenMetricLab

**OpenMetricLab** is a local-first evaluation studio for comparing ground truth with model predictions. The metric engine is implemented directly in OpenMetricLab from the mathematical definitions; it does not call scikit-learn, SciPy, or scikit-image metric functions. It covers four common scientific/ML workflows:

- **Regression:** MAE, MSE, RMSE, median absolute error, max error, R², explained variance, bias, MAPE (with zero-target coverage), sMAPE, WAPE, Pearson/Spearman correlation, and error quantiles.
- **Classification:** accuracy, balanced accuracy, macro/weighted F1, MCC, Cohen's κ, confusion matrix, per-class precision/recall/F1, and probability-aware ROC-AUC, average precision, Brier score, and log loss.
- **Segmentation:** Dice, IoU/Jaccard, precision, sensitivity/recall, specificity, pixel/balanced accuracy, Hausdorff distance, HD95, ASSD, and per-class/macro summaries.
- **Image registration:** MSE/RMSE/NRMSE, PSNR, SSIM, normalized cross-correlation, NMI, before→after improvement, and landmark TRE summaries.

The browser interface processes user files locally and is designed for desktop and mobile. It includes public/reproducible examples rather than only toy hard-coded scores.

## Python
```python
from openmetriclab import evaluate_regression
print(evaluate_regression([1,2,3],[1.1,1.8,3.2]))
```

## Example suite
`examples/manifest.json` records provenance and SHA-256 hashes. Bundled example CSVs/images come from public/open datasets or deterministic fixtures. These files are data only: OpenMetricLab does not use scikit-learn, SciPy, or scikit-image to compute the reported metrics. The image perturbations are demonstrations, not claims about published model performance.

## Privacy
The website performs evaluation in the browser. Uploaded truth/prediction files are not sent to a server by OpenMetricLab.

## License
MIT.

## Metric-engine policy
OpenMetricLab v0.2.0 keeps the scientific runtime deliberately small: **NumPy is the only runtime dependency**. Regression, classification, ROC/PR/AP, segmentation overlap/surface distances, SSIM, NMI, PSNR, NCC, and TRE are implemented in `src/openmetriclab/` and independently implemented again in the browser where applicable. Edge-case policies are documented in `docs/metric_notes.md`.
