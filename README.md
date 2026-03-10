# 🔬 Roadmap: Attention, Segmentation & Cancer Detection

Extending the baseline CNN chest X-ray classifier toward a clinically meaningful diagnostic tool.

---

## Current Baseline

| Metric | Value |
|--------|-------|
| Architecture | Custom CNN (4 conv blocks) |
| Classes | Normal · COVID-19 · Pneumonia · Lung Opacity |
| Test Accuracy | ~86.7% |
| Explainability | Grad-CAM |

---

## Phase 1 — Attention Mechanisms

Attention gates allow the model to *suppress irrelevant activations* and focus on diagnostically meaningful regions before classification. This is especially valuable for chest X-rays where pathology occupies a small, specific area.

### 1.1 Channel Attention (Squeeze-and-Excitation)

Recalibrates feature maps by learning *which channels* matter most for a given input.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Multiply

def squeeze_excitation_block(x, ratio=16):
    filters  = x.shape[-1]
    se       = GlobalAveragePooling2D()(x)
    se       = Dense(filters // ratio, activation='relu')(se)
    se       = Dense(filters, activation='sigmoid')(se)
    se       = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])

# Usage — insert after any Conv2D block:
# x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
# x = squeeze_excitation_block(x, ratio=16)
```

### 1.2 Spatial Attention (CBAM)

Focuses on *where* in the image to look, generating a 2D attention map over spatial positions.

```python
from tensorflow.keras.layers import Lambda
import tensorflow as tf

def spatial_attention(x):
    avg = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    cat = tf.keras.layers.Concatenate()([avg, mx])
    att = Conv2D(1, (7,7), padding='same', activation='sigmoid')(cat)
    return Multiply()([x, att])
```

### 1.3 Self-Attention / Vision Transformer Block

Patches the feature map and applies multi-head self-attention — capturing long-range dependencies that convolutions miss.

```python
# Recommended: use keras-cv or huggingface ViT
# from transformers import TFViTModel
# Or build a lightweight patch-based attention block on top of CNN features.
```

**Integration point:** Add after Block 3 or Block 4 of the baseline CNN before the classifier head.

---

## Phase 2 — Semantic Segmentation

Classification tells you *what* is in the image. Segmentation tells you *where*. For pathology, this is the difference between a diagnosis and a clinically actionable finding.

### 2.1 U-Net for Lesion Mask Prediction

U-Net is the standard architecture for medical image segmentation. It uses skip connections to preserve spatial detail lost during downsampling.

```
Input (299×299×1)
    │
 Encoder (contracting path)
    ├── Conv Block → 64 filters
    ├── Pool → Conv Block → 128 filters
    ├── Pool → Conv Block → 256 filters
    └── Pool → Bottleneck → 512 filters
    │
 Decoder (expanding path)
    ├── UpSample + Skip → Conv → 256
    ├── UpSample + Skip → Conv → 128
    └── UpSample + Skip → Conv → 64
    │
 Output: 1×299×299 (binary mask) or 4×299×299 (per-class masks)
```

**Required:** Pixel-level annotations (segmentation masks). Public sources:
- [JSRT dataset](http://db.jsrt.or.jp/eng.php) — lung field masks
- [Montgomery / Shenzhen](https://openi.nlm.nih.gov/) — TB & lung contour masks
- [COVID-19 CT Lung Segmentation](https://zenodo.org/record/3757476) — CT lesion masks
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — bounding box annotations

### 2.2 Attention U-Net

Adds attention gates at each skip connection so the decoder focuses only on relevant encoder features.

```python
def attention_gate(x, g, inter_channels):
    """
    x: encoder feature map (skip connection)
    g: decoder gating signal (upsampled)
    """
    Wx = Conv2D(inter_channels, (1,1), padding='same')(x)
    Wg = Conv2D(inter_channels, (1,1), padding='same')(g)
    psi = tf.keras.layers.Add()([Wx, Wg])
    psi = tf.keras.layers.Activation('relu')(psi)
    psi = Conv2D(1, (1,1), padding='same', activation='sigmoid')(psi)
    return Multiply()([x, psi])
```

This produces sharper, more localised masks — critical for pinpointing small lesions.

---

## Phase 3 — Cancer & Nodule Detection

### 3.1 Lung Nodule Detection Pipeline

```
Input X-ray / CT slice
     │
 Segmentation (U-Net)   →  Lung mask
     │
 ROI extraction          →  Crop to lung region
     │
 Nodule detector         →  Bounding boxes (YOLO / Faster R-CNN)
     │
 Malignancy classifier   →  Benign / Malignant (CNN or ViT)
     │
 Grad-CAM overlay        →  Highlighted region for radiologist
```

### 3.2 Recommended Datasets for Cancer

| Dataset | Type | Labels |
|---------|------|--------|
| [LUNA16](https://luna16.grand-challenge.org/) | CT scans | Nodule locations + diameter |
| [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | CT scans | Malignancy score per nodule |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Chest X-ray | 14 pathology labels |
| [VinDr-CXR](https://vindr.ai/datasets/cxr) | Chest X-ray | Bounding boxes + findings |

### 3.3 Loss Functions for Imbalanced Segmentation

Standard cross-entropy fails when lesions occupy <5% of pixels. Use:

```python
# Dice Loss — optimises overlap directly
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

# Focal Loss — down-weights easy negatives
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce   = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t   = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return alpha * tf.pow(1 - p_t, gamma) * bce

# Combined (standard in medical segmentation)
def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)
```

---

## Phase 4 — Evaluation Metrics

Classification and segmentation require different metrics.

| Task | Metric | Why |
|------|--------|-----|
| Classification | F1 per class | Handles class imbalance |
| Classification | AUC-ROC | Threshold-independent |
| Segmentation | Dice coefficient | Measures mask overlap |
| Segmentation | IoU (Jaccard) | Standard in object detection |
| Detection | FROC | Standard for nodule benchmarks |
| Clinical | Sensitivity @ fixed specificity | Matches radiologist workflow |

```python
# Dice coefficient (numpy)
def dice_coefficient(y_true, y_pred, threshold=0.5):
    y_pred_bin   = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_bin)
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin) + 1e-6)
```

---

## Suggested Next Steps

1. **Collect segmentation masks** — start with LUNA16 or VinDr-CXR
2. **Add SE blocks** to the existing CNN (zero architectural change, immediate gain)
3. **Train a U-Net** on lung field segmentation as a preprocessing step
4. **Stack classifer on segmented ROI** — train the existing CNN only on the lung region
5. **Add Attention U-Net** for lesion-level masks
6. **Benchmark with Grad-CAM vs attention maps** — qualitative validation with a radiologist

---

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas* (2018)
- Hu et al., *Squeeze-and-Excitation Networks* (2018)
- Woo et al., *CBAM: Convolutional Block Attention Module* (2018)
- Wang et al., *ChestX-ray14: Hospital-scale Chest X-ray Database* (2017)
