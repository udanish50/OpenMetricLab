# Reference implementations and definitions

OpenMetricLab intentionally follows established metric conventions.

- scikit-learn model evaluation: regression and classification metrics.
- scikit-image metrics: SSIM, PSNR, NMI, Hausdorff-style image metrics.
- MONAI metrics: Dice and Hausdorff/surface-distance conventions used in medical imaging.
- OpenCV: Enhanced Correlation Coefficient / normalized correlation concepts for image alignment.

When a browser implementation is simplified for responsiveness, the interface labels that limitation rather than presenting it as bit-for-bit equivalent to every library implementation. In particular, browser segmentation surface distances may deterministically sample very large boundaries; the Python implementation uses SciPy distance transforms.
