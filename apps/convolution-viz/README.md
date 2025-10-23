# Convolution & Pooling Visualizer

Interactive step-by-step visualization of Conv2D, MaxPool, and AvgPool operations.

## Features

- **Conv2D Visualization**: See how convolution kernels slide over input matrices
- **MaxPool/AvgPool Visualization**: Watch pooling operations in action
- **Step-by-Step**: Navigate through each position to understand the complete process
- **Customizable Parameters**:
  - Input matrix size (4×4 to 10×10)
  - Kernel/pool size
  - Stride
  - Padding (Conv2D only)
- **Pre-defined Kernels**:
  - Edge detection (horizontal/vertical)
  - Sharpen
  - Blur
  - Random
- **High-quality Visualizations**: Shows input, operation, and output at each step

## Installation

```bash
pip install streamlit numpy matplotlib
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### Conv2D Mode

1. **Select "Conv2D"** from operation dropdown
2. **Set input size** - creates random integer matrix
3. **Choose kernel**: Pre-defined (edge detection, sharpen, blur) or random
4. **Set parameters**: kernel size, stride, padding
5. **Navigate steps**: Use slider to see each convolution step
6. **Watch the magic**:
   - Red box shows current receptive field on input
   - Element-wise multiplication shown
   - Output builds up step by step

### Pooling Mode

1. **Select "MaxPool2D" or "AvgPool2D"**
2. **Set input size**
3. **Set parameters**: pool size, stride
4. **Navigate steps**: See how pooling samples the input
5. **Observe**:
   - Red box highlights current pool region
   - Max value highlighted in red (for MaxPool)
   - Output accumulates step by step

## Understanding the Visualization

### Conv2D
- **Left**: Input matrix with highlighted receptive field (red box)
- **Middle-Left**: Kernel values
- **Middle-Right**: Element-wise product (receptive field × kernel)
- **Right**: Output matrix being built (current position highlighted)

### Pooling
- **Left**: Input matrix with highlighted pool region (red box)
- **Middle**: Pool region detail with max value highlighted
- **Right**: Output matrix being built

## Output Size Formula

```
Output Size = ⌊(Input Size + 2×Padding - Kernel Size) / Stride⌋ + 1
```

## Examples

### Edge Detection
- Kernel: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
- Detects horizontal edges by finding intensity gradients

### Sharpen
- Kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
- Enhances edges by subtracting neighboring pixels

### MaxPool (2×2, stride=2)
- Reduces spatial dimensions by 2×
- Keeps strongest activations
- Common in CNNs for downsampling

## Educational Value

This tool helps understand:
- How convolution creates feature maps
- The role of stride in spatial downsampling
- How padding preserves spatial dimensions
- Difference between max and average pooling
- Why kernel design matters (edge detection vs blur)

## Technical Details

- Built with Streamlit for interactivity
- Matplotlib for high-quality visualizations
- NumPy for efficient array operations
- 300 DPI figures for publication-quality output
