# Child Malnutrition Detection System using Deep Learning

An advanced, high-performance Computer Vision pipeline utilizing Deep Convolutional Neural Networks (CNN) and Transfer Learning to classify whether a child is healthy or malnourished from clinical/field images.

This repository provides a standardized architecture featuring automated image data cleaning, on-the-fly geometric and color metric data augmentations, dynamic learning rate adaptation, and comprehensive analytical evaluation suites (including Loss/Accuracy trends, Confusion Matrices, and ROC/AUC curves).

---

## 📌 Project Overview & Architecture

Detecting childhood malnutrition early is a vital task for public health interventions. This project builds a production-grade binary classification model utilizing **MobileNetV2** (pre-trained on the ImageNet dataset) as an advanced feature extractor, augmented with custom dense, batch normalization, and dropout layers to maximize classification accuracy while preventing overfitting on small-scale datasets.

### Machine Learning Pipeline Stack:
* **Framework:** TensorFlow 2.x & Keras
* **Input Pipeline:** Optimized asynchronous `tf.data.Dataset` API with prefetching (`AUTOTUNE`)
* **Feature Extraction Base:** MobileNetV2 (frozen weights for feature preservation)
* **Regularization:** Random horizontal flipping, rotations, zooms, brightness/contrast scaling, and a 40% Dropout channel
* **Metrics Suite:** Scikit-Learn evaluation primitives (`classification_report`, `confusion_matrix`, `roc_curve`, `auc`)

---

## 📁 Repository Directory Structure

To run the training script successfully, arrange your workspace according to the following layout:

```
├── dataset/
│   ├── train/
│   │   ├── annotations.csv          # Classification labels for training
│   │   └── [train_images...jpg/png] # Raw training child photographs
│   └── valid/
│       ├── annotations.csv          # Classification labels for validation
│       └── [valid_images...jpg/png] # Raw validation child photographs
├── metrics_and_graphs/              # Automatically generated upon training completion
│   ├── accuracy_curve.png
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── train.py                         # Complete end-to-end model training engine
└── README.md                        # Documentation (This file)
```

> **Note on Annotations Format:** The pipeline expects standard CSV files containing `filename` and `class` columns specifying either `Healthy` or `Malnourished`. It automatically deduplicates redundant rows per image.

---

## ⚙️ Model Architecture Details

The system transforms raw photographic tensors into binary probability logits through a carefully engineered neural pipeline:

1. **Input Tensors:** Images resized to uniform dimensions (224 × 224 × 3)

2. **Data Augmentation:** Active preprocessing block applied at every batch step:
   - Horizontal flips
   - Rotations (up to ±15°)
   - Digital zooming (up to ±15%)
   - Contrast & brightness adjustments (±10%)

3. **Deep Feature Extractor:** Frozen MobileNetV2 topology capturing abstract features

4. **Regularization & Dense Projections:**
   - Global Average Pooling 2D layer
   - Dense intermediate layer (256 neurons, ReLU activation)
   - Batch Normalization
   - Dropout layer (0.40) to mitigate overfitting

5. **Output Logit:** Single-neuron layer with sigmoid activation mapping to range [0, 1]

---

## 📋 Prerequisites & Environment Setup

Ensure you have a Python environment (v3.8+) configured with an accessible CPU or an NVIDIA GPU with CUDA/cuDNN support.

Install the necessary dependencies:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Usage & Training Execution

To trigger the end-to-end loading, modeling, training, and metrics extraction pipeline, run:

```bash
python train.py
```

### What happens under the hood:

1. **Data Ingestion:** Parses train and validation CSVs, mapping `Healthy → 0` and `Malnourished → 1`

2. **Asynchronous Batching:** Normalizes images to 224 × 224 × 3, rescales pixel values to [0, 1], and batches them efficiently

3. **Training Execution:** Runs up to 30 epochs with automated callbacks:
   - **EarlyStopping:** Halts training if validation loss plateaus for 6 consecutive epochs, restoring the best parameters
   - **ReduceLROnPlateau:** Reduces learning rate by factor of 0.2 if loss plateaus for 3 epochs

4. **Export:** Saves the final model as `malnourished_healthy_model_advanced.keras`

---

## 📊 Performance Metrics & Analytical Outputs

Once training is complete, a `metrics_and_graphs/` directory is automatically generated with these analytical outputs:

### 1. Classification Report (`classification_report.txt`)

Provides detailed statistical breakdowns:

* **Precision:** Minimizes false positives (ensures healthy children aren't misdiagnosed as malnourished)
* **Recall/Sensitivity:** Minimizes false negatives (critical for ensuring no malnourished child is undetected)
* **F1-Score:** Harmonic mean of precision and recall

### 2. Confusion Matrix (`confusion_matrix.png`)

Cross-tabulation matrix tracking:

* **True Negatives (TN):** Healthy children diagnosed accurately
* **True Positives (TP):** Malnourished children caught accurately
* **False Negatives (FN):** Malnourished children misclassified as healthy
* **False Positives (FP):** Healthy children misclassified as malnourished

### 3. Loss & Accuracy Curves (`loss_curve.png` & `accuracy_curve.png`)

Visualizes train vs validation performance across training intervals to demonstrate smooth convergence without overfitting.

### 4. Receiver Operating Characteristic (ROC) Curve (`roc_curve.png`)

Plots True Positive Rate against False Positive Rate across threshold variations. The closer the Area Under Curve (AUC) score to 1.000, the more robust the diagnostic power.

---

## 🛠️ Performance Optimization & Hyperparameter Tuning

To adjust baseline behaviors for custom datasets, modify these parameters directly in `train.py`:

```python
IMG_SIZE = (224, 224)   # Target spatial resolution
BATCH_SIZE = 32         # Lower to 16 if GPU hits Out-Of-Memory (OOM)
EPOCHS = 30             # Maximum epoch iterations
LEARNING_RATE = 1e-4    # Initial optimizer step size
```

### Fine-Tuning (Advanced)

Once initial frozen base metrics stabilize, you can unfreeze the core base network layers:

```python
# Unfreeze base model
base_model.trainable = True

# Recompile with ultra-low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 📜 License

This architecture is free to use for academic, clinical, and open-source validation deployments.

---

## 📧 Support & Questions

For issues, questions, or contributions, please open an issue on the repository or contact the development team.