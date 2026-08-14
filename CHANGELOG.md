# Changelog

## 0.1.0 — 2026-08-14
- Initial open-source release with regression, classification, segmentation, registration, browser UI, Python API, and eight reproducible examples.
## 0.2.0 — native metric engine
- Removed scikit-learn, SciPy, and scikit-image runtime metric dependencies.
- Implemented regression and classification metrics directly from formulas.
- Added tie-aware ROC-AUC and average precision.
- Added direct segmentation surface distances and explicit empty-mask policies.
- Added direct SSIM, NMI, PSNR, NCC, and TRE implementations.
- Reduced runtime dependencies to NumPy only.
- Added hand-checkable and dependency-guard tests plus browser-engine tests.

