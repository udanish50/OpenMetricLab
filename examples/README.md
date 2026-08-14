# Example provenance

Examples are deterministic and intended for interactive evaluation testing.

- Diabetes, Iris, Wine, Breast Cancer Wisconsin, and Digits are bundled as static public-data CSV fixtures. They were prepared from open/public dataset distributions; runtime metric computation does not import scikit-learn.
- `segmentation_horse_*` uses a bundled public horse-silhouette reference mask; the prediction is a deterministic perturbation.
- `registration_shepp_*` uses the standard Shepp–Logan phantom as a bundled image fixture; moving/registered variants are deterministic geometric perturbations.
- The three-label segmentation geometry is explicitly synthetic and is included to exercise multiclass reporting.

The fixtures are not presented as benchmark claims or state-of-the-art model results.
