# DLCA3D-MSA: Dilated Learning for Cerebral Aneurysms with Multi-Scale Anchors

## Model Overview

**DLCA3D-MSA** is a novel 3D deep learning architecture for detecting cerebral aneurysms in CT Angiography (CTA) volumes. The model combines dilated convolutions, spatial pyramid pooling, and multi-scale anchor-based detection to achieve 97.5% sensitivity with 13.8 false positives per case.

## Novel Architecture Components

### 1. Multi-Scale Anchor Detection
- **5 anchor sizes**: `[4, 6, 10, 30, 60]` voxels
- Detects aneurysms ranging from small (4mm) to large (60mm)
- Each anchor predicts: `[confidence, z_offset, y_offset, x_offset, size]`

### 2. Dilated Atrous Convolution (DAC) Block
```python
class DACblock(nn.Module):
    # Cascaded dilated convolutions: 1→3→5 dilation rates
    # Captures multi-scale contextual information without resolution loss
    # Residual connection: out = x + dilate1 + dilate2 + dilate3 + dilate4
```

### 3. 3D Spatial Pyramid Pooling (SPP)
```python
class SPPblock(nn.Module):
    # Multi-resolution pooling: [2x2x2, 3x3x3, 5x5x5, 6x6x6]
    # Upsampled and concatenated for feature fusion
    # Handles varying aneurysm sizes and spatial contexts
```

### 4. Coordinate-Aware Processing
- Takes both **image data** and **coordinate grids** as input
- Network learns spatial anatomical relationships
- Improves localization accuracy in 3D space

### 5. Patch-Based 3D Processing
- Splits large volumes into **144³ patches** with **16-voxel margins**
- Enables full-brain processing on limited GPU memory
- Overlapping patches ensure no boundary artifacts

## Pipeline Architecture

### Pre-Processing Step

**Input**: DICOM CT Angiography series
**Output**: Preprocessed 3D NIfTI volumes

```bash
python ./utils/pre_process.py --input="./raw_data/" --output="./train_data/"
```

**Processing Steps**:
1. **DICOM to NIfTI Conversion**
   - Reads DICOM series with spatial metadata
   - Computes voxel spacing from slice positions
   - Converts to standardized NIfTI format

2. **Intensity Normalization**
   - Mean: -535.85, Std: 846.87 (HU units)
   - Clips values to remove outliers
   - Standardizes contrast across datasets

3. **Spatial Resampling**
   - Target spacing: `(0.39, 0.39, 0.39)` mm³
   - Cubic interpolation for images
   - Nearest neighbor for labels
   - Maintains anatomical proportions

4. **Volume Cropping**
   - Crops to `512³` voxels centered on brain
   - Uses center of mass for optimal positioning
   - Pads with minimum intensity values

5. **Bounding Box Generation**
   - Extracts aneurysm locations from labels
   - Generates 3D bounding boxes with centers and sizes
   - Saves as NumPy arrays for training

### Training Step

**Input**: Preprocessed 3D volumes + bounding box annotations
**Output**: Trained model checkpoint

```bash
python train.py -j=16 -b=12 --input="train_data/" --output="./checkpoint/"
```

**Training Configuration**:
- **Patch Size**: `128³` voxels
- **Batch Size**: 12 patches
- **Learning Rate**: 0.01 with momentum 0.9
- **Weight Decay**: 1e-4
- **Epochs**: 500

**Data Augmentation**:
- Random flipping (enabled)
- Random cropping: 30% probability
- No scaling/rotation (preserves anatomical accuracy)

**Loss Function**:
- **Classification Loss**: Focal loss for confidence scores
- **Regression Loss**: Smooth L1 loss for bounding box coordinates
- **Hard Negative Mining**: Selects 800 negative samples per positive

**Training Process**:
1. **Patch Extraction**: Random 128³ crops from 512³ volumes
2. **Coordinate Grid Generation**: Creates spatial coordinate maps
3. **Multi-Scale Prediction**: Outputs 5 anchor predictions per voxel
4. **Loss Computation**: Combines classification + regression losses
5. **Backpropagation**: Updates network weights

### Validation Step

**Validation Strategy**:
- **Split**: Typically 80/20 train/validation
- **Metrics**: Sensitivity, specificity, false positive rate
- **Evaluation**: Per-case and per-lesion analysis

**Validation Process**:
1. **Full Volume Inference**: Processes entire 512³ volumes
2. **Patch Reconstruction**: Combines overlapping patch predictions
3. **Non-Maximum Suppression**: Removes duplicate detections
4. **Threshold Optimization**: Finds optimal confidence threshold
5. **Performance Metrics**: Computes sensitivity/FP rate curves

### Testing Step

**Input**: Test CTA volumes
**Output**: Aneurysm detection predictions

```bash
python inference.py -j=1 -b=1 --resume="./checkpoint/trained_model.ckpt" \
    --input="./test_image/brain_CTA" --output="./prediction/brain_CTA"
```

**Testing Pipeline**:

1. **Volume Loading**
   - Loads NIfTI test volumes
   - Applies same normalization as training
   - Pads to network input size

2. **Patch-Based Inference**
   - Splits volume into overlapping 144³ patches
   - Processes each patch through network
   - Generates coordinate-aware predictions

3. **Prediction Reconstruction**
   - Combines patch outputs using `SplitComb` class
   - Handles overlapping regions with averaging
   - Reconstructs full-volume prediction map

4. **Post-Processing**
   - Applies confidence threshold (default: -3)
   - Converts network outputs to world coordinates
   - Applies non-maximum suppression (IoU: 0.05)

5. **Output Generation**
   - Saves bounding box predictions as NumPy arrays
   - Format: `[confidence, z, y, x, size]` per detection
   - Generates visualization overlays

## Performance Results

| Metric | Value |
|--------|-------|
| **Sensitivity** | 97.5% |
| **False Positives per Case** | 13.8 |
| **Processing Time** | ~2-3 minutes per case |
| **Memory Requirements** | 8GB GPU RAM |

## Key Innovations Summary

1. **DLCA3D-MSA Architecture**: Novel combination of dilated convolutions + spatial pyramid pooling + multi-scale anchors
2. **Coordinate-Aware Learning**: Spatial position encoding improves anatomical localization
3. **Patch-Based 3D Processing**: Enables full-brain analysis on limited hardware
4. **Multi-Scale Detection**: Handles aneurysms from 4mm to 60mm diameter
5. **End-to-End Pipeline**: Complete DICOM-to-detection workflow

This approach represents a significant advancement in automated cerebral aneurysm detection, combining multiple novel architectural components for robust 3D medical image analysis.
