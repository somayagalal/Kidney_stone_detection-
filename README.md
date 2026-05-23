# DL Kidney Stone Detection

This notebook builds and compares two deep learning models for kidney stone detection from ultrasound images:

- **Optimized Grayscale CNN**
- **MobileNetV2 Transfer Learning Model**

The notebook file is:

```text
DL_Kidney_stone_detection (1).ipynb
```

The goal is binary image classification:

| Class | Label |
|---|---:|
| Normal | 0 |
| stone | 1 |

## Project Overview

The notebook follows a complete deep learning pipeline:

1. Import required libraries.
2. Load the ultrasound dataset.
3. Preview images from both classes.
4. Resize images to a fixed shape.
5. Build feature and label arrays.
6. Save and reload processed data using pickle.
7. Normalize pixel values.
8. Split the dataset into training, validation, and testing sets.
9. Apply data augmentation.
10. Train an optimized grayscale CNN.
11. Evaluate the CNN using accuracy, loss, confusion matrix, and ROC-AUC.
12. Train a MobileNetV2 transfer learning model.
13. Compare both models.
14. Select the final recommended model.

## Dataset Structure

The notebook expects the dataset to be stored in this format:

```text
dataset/
├── Normal/
└── stone/
```

The class list used in the notebook is:

```python
CATEGORIES = ["Normal", "stone"]
```

The dataset contains:

```text
Total images: 2000
```

## Image Preprocessing

The notebook reads ultrasound images in grayscale using OpenCV:

```python
cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
```

The original sample image shape is:

```text
512 x 512
```

All images are resized to:

```text
224 x 224
```

The CNN input shape becomes:

```text
(224, 224, 1)
```

Pixel values are normalized from the original `0-255` range into the `0-1` range:

```python
X = X.astype("float32") / 255.0
```

Notebook output:

```text
Minimum pixel value: 0.0
Maximum pixel value: 1.0
```

## Dataset Split

The dataset is split into training, validation, and testing sets using a 70/15/15 ratio.

| Split | Shape |
|---|---|
| Training | `(1400, 224, 224, 1)` |
| Validation | `(300, 224, 224, 1)` |
| Testing | `(300, 224, 224, 1)` |

The split is stratified, which helps keep the class balance consistent across training, validation, and testing.

## Data Augmentation

Data augmentation is used to increase variation in the training set and reduce overfitting.

The notebook applies:

- Rotation
- Zoom
- Width shift
- Height shift
- Horizontal flip

Notebook output:

```text
Original training size: 1400
Augmented training size: 4200
Increase: 3.0 x
```

The augmented data is saved into:

```text
dataset/augmented/
```

The augmented arrays are also saved as pickle files:

```text
X_train_augmented.pickle
y_train_augmented.pickle
```

## Model 1: Optimized Grayscale CNN

The first model is a custom CNN trained from scratch on grayscale ultrasound images.

### Architecture

The CNN is built using:

- `Conv2D`
- `BatchNormalization`
- `MaxPooling2D`
- `GlobalAveragePooling2D`
- `Dense`
- `Dropout`
- `Dense(1, activation="sigmoid")`

The sigmoid output is used because this is a binary classification problem.

### Hyperparameter Search

The notebook tests combinations of:

| Hyperparameter | Values |
|---|---|
| Convolution layers | 2, 3 |
| Filters | 32 |
| Dense units | 64, 128 |
| Dropout rate | 0.5 |
| Learning rate | 0.0001 |
| Batch size | 16 |
| Epochs | 10 |

### Best CNN Parameters

The best CNN configuration found in the notebook is:

```python
{
    "conv_layers": 3,
    "filters": 32,
    "dense_units": 64,
    "dropout_rate": 0.5,
    "learning_rate": 0.0001
}
```

The best model is saved as:

```text
best_grayscale_cnn_model.keras
```

### CNN Model Size

| Parameter Type | Count |
|---|---:|
| Total parameters | 63,749 |
| Trainable parameters | 21,185 |
| Non-trainable parameters | 192 |
| Optimizer parameters | 42,372 |

### CNN Test Results

```text
Test Loss: 0.1735
Test Accuracy: 0.9700
ROC-AUC: 0.9962
```

CNN confusion matrix:

```text
[[150   0]
 [  9 141]]
```

Interpretation:

- Correct Normal predictions: 150
- Normal images predicted as stone: 0
- Stone images predicted as Normal: 9
- Correct stone predictions: 141

The CNN performs well, but it misses 9 stone cases.

## Model 2: MobileNetV2 Transfer Learning

The second model uses MobileNetV2 as a pretrained feature extractor.

MobileNetV2 expects RGB input, so the grayscale image channel is repeated three times:

```python
X_rgb = np.repeat(X, 3, axis=-1)
```

The MobileNetV2 input shape is:

```text
(224, 224, 3)
```

Notebook output:

```text
MobileNetV2 validation shape: (300, 224, 224, 3)
MobileNetV2 testing shape: (300, 224, 224, 3)
```

### MobileNetV2 Architecture

The model uses:

```text
MobileNetV2 base model
→ GlobalAveragePooling2D
→ Dense(128, activation="relu")
→ Dropout(0.5)
→ Dense(1, activation="sigmoid")
```

The MobileNetV2 base layers are frozen:

```python
for layer in base_model.layers:
    layer.trainable = False
```

This means the notebook trains only the custom classification head.

### MobileNetV2 Model Size

| Parameter Type | Count |
|---|---:|
| Total parameters | 2,422,081 |
| Trainable parameters | 164,097 |
| Non-trainable parameters | 2,257,984 |

### MobileNetV2 Training

The model is trained using:

- `Adam(learning_rate=0.0001)`
- `binary_crossentropy`
- `accuracy`
- `EarlyStopping`

Early stopping monitors validation loss and restores the best weights:

```python
EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)
```

### MobileNetV2 Test Results

```text
MobileNetV2 Test Loss: 0.0165
MobileNetV2 Test Accuracy: 1.0000
ROC-AUC: 1.0000
```

MobileNetV2 confusion matrix:

```text
[[150   0]
 [  0 150]]
```

Interpretation:

- Correct Normal predictions: 150
- Normal images predicted as stone: 0
- Stone images predicted as Normal: 0
- Correct stone predictions: 150

MobileNetV2 correctly classified every image in the test set.

## Model Comparison

| Metric | Optimized Grayscale CNN | MobileNetV2 |
|---|---:|---:|
| Input shape | `224x224x1` | `224x224x3` |
| Test accuracy | 0.9700 | 1.0000 |
| Test loss | 0.1735 | 0.0165 |
| ROC-AUC | 0.9962 | 1.0000 |
| False positives | 0 | 0 |
| False negatives | 9 | 0 |

## Final Model Decision

The notebook chooses **MobileNetV2** as the final model.

Final notebook output:

```text
MobileNetV2 performed better based on test accuracy.
MobileNetV2 also achieved the higher ROC-AUC score.

Recommendation:
Use MobileNetV2 as the final model for this experiment.
```

## How to Run the Notebook

1. Create and activate a Python environment.

2. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn jupyter
```

3. Make sure the dataset is placed in the expected folder structure:

```text
dataset/
├── Normal/
└── stone/
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```text
DL_Kidney_stone_detection (1).ipynb
```

6. Run all cells in order.

## Generated Files

The notebook may generate these files:

```text
X.pickle
y.pickle
X_train_augmented.pickle
y_train_augmented.pickle
best_grayscale_cnn_model.keras
dataset/augmented/
```

## Notes

- The notebook prints a TensorFlow warning that native Windows TensorFlow versions newer than 2.10 do not use GPU acceleration directly.
- If GPU training is required on Windows, consider WSL2 or TensorFlow-DirectML.
- The reported results are strong, but real medical deployment would require external validation, patient-level splitting, and clinical review.

## Conclusion

This notebook demonstrates a full deep learning workflow for kidney stone detection using ultrasound images. The optimized grayscale CNN achieves strong performance, but MobileNetV2 performs best with:

```text
Test Accuracy: 1.0000
ROC-AUC: 1.0000
False Negatives: 0
```

Therefore, **MobileNetV2 is the recommended final model for this experiment**.
