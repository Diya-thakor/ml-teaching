# Equivariance & Invariance Demo

Interactive Streamlit app demonstrating key concepts from **Bernhard Kainz's lecture on Equivariance and Invariance in Deep Learning**.

## Concepts Demonstrated

### Invariance
**Definition**: Output stays constant despite input transformation

**Example**: Image classification
- Input: Cat image → Output: "cat"
- Input: Shifted cat image → Output: "cat" (same!)
- Formula: `f(x) = f(S_v(x))` where S is shift operator

### Equivariance
**Definition**: Output undergoes the same transformation as input

**Example**: Semantic segmentation
- Input: Cat image → Output: Cat segmentation mask
- Input: Shifted cat image → Output: Shifted cat mask
- Formula: `f(S_v(x)) = S_v(f(x))` - f and S commute

## Features

### Tab 1: Convolution Equivariance
- **Interactive demonstration** showing how convolution preserves spatial relationships
- **Two paths comparison**:
  - Path A: Input → Shift → Conv
  - Path B: Input → Conv → Shift
- Both paths produce **identical outputs**, proving equivariance
- Multiple test patterns and kernels
- Adjustable shift amounts

### Tab 2: Max Pooling Invariance
- **1D Signal Example** (reproducing lecture slides 104-115)
  - Shows how max pooling **breaks equivariance**
  - Simple 0→1→0 signal demonstrates dramatic output changes with 1-pixel shifts
  - Visual pooling windows show exactly which values are selected

- **2D Pattern Example**
  - Demonstrates **approximate invariance** for small shifts
  - Shows how max pooling windows affect output
  - Compares original vs shifted pooling results

## Key Insights from Lecture

1. **Convolutional layers** are shift equivariant (theoretically via Fourier transform)
2. **Max pooling** provides approximate shift invariance BUT breaks equivariance
3. **Problem**: Striding violates Nyquist sampling theorem → aliasing
4. **Solution**: Blur before downsampling (anti-aliasing)
5. **Beyond translation**: Can extend to group equivariance (rotations, etc.)

## Run the App

```bash
streamlit run app.py
```

## Requirements

```
streamlit
numpy
matplotlib
scipy
```

## Reference

Based on lecture materials by **Bernhard Kainz**:
- Deep Learning – Equivariance and Invariance
- Imperial College London / TU Wien

Key paper referenced:
- R. Zhang. "Making Convolutional Networks Shift-Invariant Again." ICML, 2019.
