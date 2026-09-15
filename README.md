# L-a-b-Chrominance-and-Gabor-Texture-Features-for-Leaf-Health-Classification

A Gradient Boosting Machine (GBM) image classification model for identifying the health condition of kangkong (*Ipomoea aquatica*) leaves. This study extends Article 1 by evaluating a leaner feature combination that retains L\*a\*b\* chrominance and Gabor texture features while removing multi-scale Local Binary Patterns (LBP).

The model classifies kangkong leaves into three categories:

- **Healthy**
- **Discolored**
- **Diseased**

## About the Study

Article 2 investigates whether a reduced feature combination can maintain or improve classification performance while using fewer feature dimensions.

The Stage D lean configuration combines the baseline image features with L\*a\*b\* chrominance histograms and Gabor filter features. Multi-scale LBP was deliberately removed because the feature-importance analysis from Stages B and C identified it as the weakest per-feature contributor among the newly added feature blocks.

### Feature Configuration

| Feature Group | Description | Feature Count |
|---|---|---:|
| Baseline features | Color, texture, engineered color and lesion features | 571 |
| L\*a\*b\* chrominance | Histograms of the a and b color channels | 128 |
| Gabor filter bank | Multi-orientation and multi-frequency texture responses | 16 |
| **Total** | | **715** |

The feature extractor resizes images to 128 × 128 pixels before segmentation and feature extraction.

## Model Architecture

The classification model uses `GradientBoostingClassifier` from Scikit-learn.

The training pipeline includes:

1. Image loading and validation.
2. Leaf segmentation using HSV-based color thresholds and morphological processing.
3. Extraction of baseline, L\*a\*b\*, and Gabor features.
4. Stratified train-test splitting.
5. Class-balanced sample weighting.
6. Five-fold stratified cross-validation.
7. Randomized hyperparameter search.
8. Final evaluation on the held-out test set.

### Dataset Split

The implementation uses a stratified 85/15 development-test split with `random_state=42`.

| Dataset | Samples |
|---|---:|
| Development set | 787 |
| Final test set | 139 |
| **Total** | **926** |

The same split is used to support comparison with Article 1 and Stages B/C.

## Hyperparameter Optimization

RandomizedSearchCV evaluates 30 parameter combinations using five-fold stratified cross-validation. The optimization metric is macro F1-score.

### Best Hyperparameters

| Parameter | Selected Value |
|---|---|
| `n_estimators` | 45 |
| `learning_rate` | 0.15 |
| `max_depth` | 4 |
| `subsample` | 1.0 |
| `max_features` | `sqrt` |
| `min_samples_leaf` | 4 |
| `min_samples_split` | 4 |

**Best cross-validation macro F1-score:** 0.9318 ± 0.0110

## Final Test Results

The trained model was evaluated on 139 held-out test images.

| Metric | Score |
|---|---:|
| Accuracy | 96.40% |
| Macro Precision | 96.55% |
| Macro Recall | 96.45% |
| Macro F1-Score | 96.45% |

### Per-Class Classification Results

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Healthy | 1.0000 | 0.9778 | 0.9888 | 45 |
| Discolored | 0.9184 | 0.9783 | 0.9474 | 46 |
| Diseased | 0.9783 | 0.9375 | 0.9574 | 48 |
| **Macro Average** | **0.9655** | **0.9645** | **0.9645** | **139** |

## Confusion Matrix

### Raw Counts

The confusion matrix shows the number of correctly and incorrectly classified images for each class.

![Lean Stage D Final Test Confusion Matrix — Counts](outputs_stageD_lean/final_test_confusion_matrix.png)

### Row-Normalized Confusion Matrix

The row-normalized matrix presents the proportion of predictions for each true class.

![Lean Stage D Final Test Confusion Matrix — Normalized](outputs_stageD_lean/final_test_confusion_matrix_normalized.png)

### Confusion Matrix Values

Rows represent the true labels, while columns represent the predicted labels.

| True Label | Healthy | Discolored | Diseased |
|---|---:|---:|---:|
| Healthy | 44 | 1 | 0 |
| Discolored | 0 | 45 | 1 |
| Diseased | 0 | 3 | 45 |

The largest source of misclassification is between the discolored and diseased classes.

- Discolored predicted as diseased: 1/46 (2.2%).
- Diseased predicted as discolored: 3/48 (6.2%).

## Comparison with Article 1 and Previous Stages

The following comparison uses the same test split reported by the training script.

| Model | Accuracy | Macro F1-Score |
|---|---:|---:|
| Article 1 Baseline | 93.53% | 93.63% |
| Stage B | 94.24% | 94.32% |
| Stage C | 94.24% | 94.34% |
| **Lean Stage D** | **96.40%** | **96.45%** |

Compared with the Article 1 baseline, Lean Stage D achieves:

- Accuracy improvement: +2.87 percentage points.
- Macro F1-score improvement: +2.82 percentage points.

## Misclassified Test Images

The final test evaluation identified five misclassified images.

| True Class | Predicted Class | Image |
|---|---|---|
| Diseased | Discolored | `dataset/diseased/image (310).jpg` |
| Diseased | Discolored | `dataset/diseased/image (417).jpg` |
| Diseased | Discolored | `dataset/diseased/image (362).jpg` |
| Healthy | Discolored | `dataset/healthy/image (94).jpg` |
| Discolored | Diseased | `dataset/discolored/image (40).jpg` |

The complete list is saved in:

`outputs_stageD_lean/final_test_misclassified.txt`

## Prediction Visualization

The prediction visualization script generates a four-panel figure containing:

1. Original image.
2. Leaf segmentation mask.
3. Segmented leaf region.
4. Prediction confidence scores.

The script supports PNG and PDF output at 300 DPI for use in research documentation.

### Example Prediction Figure

![Stage D Prediction Example](figures/stageD_prediction_example.png)

## Project Structure

```text
GBM-Image-Analysis/
│
├── dataset/
│   ├── healthy/
│   ├── discolored/
│   └── diseased/
│
├── models/
│   ├── gbm_model_cv_stageD_lean.pkl
│   └── gate_model.pkl
│
├── outputs_stageD_lean/
│   ├── cv_results_summary.txt
│   ├── final_test_report.txt
│   ├── final_test_confusion_matrix.txt
│   ├── final_test_confusion_matrix.png
│   ├── final_test_confusion_matrix_normalized.png
│   └── final_test_misclassified.txt
│
├── figures/
│   ├── stageD_prediction_example.png
│   ├── stageD_prediction_example.pdf
│   ├── stageD_performance_metrics.png
│   └── stageD_performance_metrics.pdf
│
├── feature_extractor_stageD_lean.py
├── train_cv_stageD_lean.py
├── predict_visualize_stageD.py
└── plot_performance_metrics_stageD.py
```

## Installation

Install the required Python libraries:

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib joblib
```

## Usage

### 1. Train the Lean Stage D Model

Ensure that the dataset is organized into the three class folders:

```text
dataset/
├── healthy/
├── discolored/
└── diseased/
```

Run:

```bash
python train_cv_stageD_lean.py
```

The training script performs feature extraction, hyperparameter optimization, final testing, and model saving.

The trained model is saved as:

```text
models/gbm_model_cv_stageD_lean.pkl
```

### 2. Generate a Prediction Visualization

Edit the `TEST_IMAGE` variable in `predict_visualize_stageD.py` to specify the kangkong leaf image to analyze.

```python
TEST_IMAGE = "test/test (80).jpg"
```

Run:

```bash
python predict_visualize_stageD.py
```

The script generates a prediction figure in the `figures/` directory.

### 3. Generate Performance Metrics

Run:

```bash
python plot_performance_metrics_stageD.py
```

This generates the performance metrics chart in PNG and PDF formats.

## Output Files

| File | Description |
|---|---|
| `cv_results_summary.txt` | Cross-validation scores and hyperparameter rankings |
| `final_test_report.txt` | Accuracy, precision, recall, F1-score, and confusion analysis |
| `final_test_confusion_matrix.txt` | Raw confusion matrix values |
| `final_test_confusion_matrix.png` | Confusion matrix in raw counts |
| `final_test_confusion_matrix_normalized.png` | Row-normalized confusion matrix |
| `final_test_misclassified.txt` | Paths and labels of misclassified test images |
| `gbm_model_cv_stageD_lean.pkl` | Trained GBM model and associated metadata |

## Reproducibility

The experiment uses fixed random seeds for the train-test split, cross-validation, and randomized hyperparameter search.

- Train-test split: `random_state=42`
- Cross-validation: 5-fold `StratifiedKFold`
- Hyperparameter search: 30 randomized configurations
- Class balancing: Balanced sample weights

The feature extractor and training script must remain compatible to preserve the expected feature dimensions and model input format.

## Research Context

This work forms **Article 2** of the GBM Image Analysis series. It evaluates the lean Stage D feature configuration as a continuation of the baseline and feature-ablation experiments presented in Article 1.

The primary objective is to assess the classification performance of combining baseline features, L\*a\*b\* chrominance, and Gabor texture features without the multi-scale LBP addition.
