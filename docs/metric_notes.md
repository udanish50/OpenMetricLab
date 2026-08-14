# Metric interpretation notes

No single metric is universally sufficient. OpenMetricLab deliberately groups metrics by the question they answer.

- Error magnitudes (MAE/RMSE) retain target units; R² is unitless and can be negative.
- MAPE is undefined at zero target values, so the implementation reports the coverage used for MAPE instead of silently substituting an epsilon.
- Classification probability metrics are shown only when probabilities are supplied.
- Dice/IoU describe region overlap; HD95/ASSD describe boundary/surface discrepancy and depend on pixel/voxel spacing.
- Registration intensity similarity does not prove anatomical/geometric correctness. Landmark TRE, when available, addresses a different question and should be reported alongside image similarity.
