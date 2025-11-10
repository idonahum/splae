# SPLAE - Segmentation Pseudo Label Accuracy Estimation

To use EPL and IPLC, you need the Med SAM model. Download it from: https://github.com/OpenGVLab/SAM-Med2D

## Overview
SPLAE is a framework for estimating the accuracy of pseudo labels in medical image segmentation tasks. It supports domain adaptation, benchmarking, and evaluation of segmentation models across multiple datasets and adaptation methods.

## Installation
1. **Python Version**: Install Python 3.10
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets
- **M&Ms Dataset**: [https://www.ub.edu/mnms/](https://www.ub.edu/mnms/)
- **Prostate Dataset**: [https://liuquande.github.io/SAML/](https://liuquande.github.io/SAML/)

### Dataset Processing
For each dataset, split the 3D images into 2D images. You can use the provided scripts:
- `procces_mnm_dataset.py`
- `process_msm_dataset.py`

**Required Structure:**
```
<ds_root>/
    <domain>/
        train/
            gt_masks/
            images/
        valid/
            gt_masks/
            images/
        test/
            gt_masks/
            images/
```

## Training a Source Model
- Use `train_source.py` to train a segmentation model on the source domain.
- Adjust dataset paths in the script as needed.

## Domain Adaptation
- Adapt to the target domain using the provided scripts:
  - `adapt_dpl.py`
  - `adapt_iplc.py`
  - `adapt_tent.py`

## Using SPLAE
- Use `splae_test.py` for pseudo label accuracy estimation.
- Map source-target pairs in `dataset_pairs_sample.json`.

## Additional Benchmarks
- The codebase implements ATLAS RCA and SAM2 RCA.
- For SAM2 RCA, follow the installation instructions at: [https://github.com/mcosarinsky/In-Context-RCA](https://github.com/mcosarinsky/In-Context-RCA)

---
For questions or issues, please open an issue in this repository.
