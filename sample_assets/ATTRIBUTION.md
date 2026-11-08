# Sample Assets Attribution & Metadata

This directory contains the minimal set of 4 curated sample images bundled for the interactive **Pump Impeller Quality Inspector** demonstration. The full dataset archives (`casting_data.zip` and `unsupported_data.zip`) are excluded from version control and deployment containers.

---

## 1. Sample Manifest & Attributions

| Sample ID | Bundled Filename | Original Source Filename | Dataset / Origin | Creator / Author | License | SHA-256 Hash | Intended Demo Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sample_unsupported`** | `sample_unsupported.jpg` | `pump_housing_005_housing_for_modern_pump_at_westonzo.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Housing_for_modern_pump_at_Westonzoyland_Pumping_Station_-_geograph.org.uk_-_7258660.jpg) / geograph.org.uk | David Smith | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) | `3fa4f4aba070244023a617ea412c06b05536b5dbf8c71ab47c693deafcc9f156` | **Unsupported Image** ($P < 0.0049 \implies$ Blocked at Gatekeeper) |
| **`sample_pass`** | `sample_pass.jpeg` | `casting_data/test/ok_front/cast_ok_0_1093.jpeg` | [Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Raviraj Joshi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) | `0e96c6bceb022170e5d7ca5254f2b8dd36957ffc41d1b5703f5b1f741acab7c7` | **Pass Candidate** ($P_{\text{def}} < 0.50 \implies$ Normal release procedure) |
| **`sample_review`** | `sample_review.jpeg` | `casting_data/train/def_front/cast_def_0_2518.jpeg` | [Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Raviraj Joshi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) | `f796665b933cff42cb07da39868779a01ea84d8538b5d1c6659457ec80f1b9c2` | **Manual Review** ($0.50 \le P_{\text{def}} < 0.76 \implies$ Review station inspection) |
| **`sample_hold`** | `sample_hold.jpeg` | `casting_data/test/def_front/cast_def_0_1292.jpeg` | [Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Raviraj Joshi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) | `109108f2b1ff50909921f94cb803dd5ba7809d4cf8d5da1c785dd94d958a15eb` | **Hold** ($P_{\text{def}} \ge 0.76 \implies$ Supervisor review) |

---

## 2. Licensing Summary

- **Casting Product Images** (`sample_pass.jpeg`, `sample_review.jpeg`, `sample_hold.jpeg`): Dedicated to the public domain under CC0 by creator Raviraj Joshi.
- **Pump Housing Image** (`sample_unsupported.jpg`): Licensed under Creative Commons Attribution-ShareAlike 2.0 Generic (CC BY-SA 2.0) by David Smith via Geograph / Wikimedia Commons.

