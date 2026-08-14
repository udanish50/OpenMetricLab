# Example provenance

Examples are deterministic and intended for interactive evaluation testing.

- Diabetes, Iris, Wine, Breast Cancer Wisconsin, and Digits are loaded from scikit-learn. Predictions are generated on deterministic held-out splits by conventional estimators.
- `segmentation_horse_*` uses the scikit-image `data.horse` silhouette as a reference mask; the prediction is a deterministic perturbation.
- `registration_shepp_*` uses the standard Shepp–Logan phantom from `skimage.data.shepp_logan_phantom`; moving/registered variants are deterministic geometric perturbations.
- The three-label segmentation geometry is explicitly synthetic and is included to exercise multiclass reporting.

The fixtures are not presented as benchmark claims or state-of-the-art model results.
