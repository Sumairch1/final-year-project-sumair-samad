# File **Face-Mask Classification for Public-Health Compliance Using Lightweight CNNs** 

---

# Overview
This notebook builds a binary image classifier to detect face-mask usage (*WithMask* vs *WithoutMask*).  
It prepares the Face-Mask-12K dataset, performs preprocessing (normalization, optional augmentation), and trains multiple CNN models (transfer learning and a custom lightweight CNN).  
Models are evaluated with accuracy, precision, recall, F1, confusion matrices, and timing to compare effectiveness and efficiency.

---

# Dataset
- **Path/URL:** Kaggle ID: `ashishjangra27/face-mask-12k-images-dataset`; Local path used: `/content/face_mask_12k/Face Mask Dataset/` (with `Train/`, `Validation/`, `Test/` subfolders)  
- **Target column:** Not specified in notebook  
- **Feature column(s):** Not specified in notebook  
- **Feature count/types:** Not specified in notebook

---

# Features & Preprocessing
- Directory-based image loading via `ImageDataGenerator.flow_from_directory` with class folders (`WithMask`, `WithoutMask`) and splits (`Train`, `Validation`, `Test`).
- Resizing:
  - Most models: `target_size=(224, 224)`
  - InceptionV3: `target_size=(299, 299)`
- Scaling: `ImageDataGenerator(rescale=1./255)` for all splits.
- Data augmentation (training only, in augmented runs):
  - `ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')`
- Label mode: `class_mode='binary'`.
- Exploratory visuals included: class-count bar charts; sample image grids per class/split.

---

# Models
- **Transfer Learning Backbones** (pretrained on ImageNet, `include_top=False`):
  - `VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)`
  - `ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)`
  - `InceptionV3(input_shape=(299,299,3), weights='imagenet', include_top=False)`
  - **Classification head (for each backbone):** `GlobalAveragePooling2D → Dense(128, activation='relu') → Dropout(0.5) → Dense(1, activation='sigmoid')` (base layers initially frozen)
  - **Compilation (binary):** `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
- **Custom Lightweight CNN (baseline):**
  - Sequential conv–pool blocks (32/64/128 filters) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
  - **Compilation (binary):** `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
- **Training setup (all models):** batch size 32; up to 20 epochs; `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and `ModelCheckpoint(save_best_only=True)`.

---

# Evaluation
- **Metrics:** model `evaluate` (accuracy, loss); `classification_report` (precision, recall, F1); `confusion_matrix`.
- **Visualizations:** confusion matrix heatmap; training history (accuracy/loss); class distribution bar charts; sample image grids.
- **Tuning:** No grid/random search; early stopping and checkpointing used.
- **Timing:** Training and testing wall-clock time recorded with `time.time()`.

---

# Environment & Requirements
- **Libraries:** tensorflow, scikit-learn, numpy, matplotlib, seaborn, opencv-python, os, time, zipfile, random, collections
- **Install example:**
  ```bash
  pip install tensorflow scikit-learn seaborn opencv-python matplotlib numpy
  ```
