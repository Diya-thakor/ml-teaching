import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d

# ------------------------------
# Streamlit setup
# ------------------------------
st.set_page_config(page_title="Equivariance & Invariance Demo", layout="wide")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'

st.title("Equivariance & Invariance in CNNs")
st.markdown("""
**Educational demonstration of two key properties from Bernhard Kainz's lecture:**

**Invariance** - Output stays the same:
- Image classification: "cat" → shift image → still "cat"

**Equivariance** - Output transforms like input:
- Segmentation: cat mask → shift image → cat mask shifts too

**This app demonstrates both concepts with interactive examples.**
""")

# ------------------------------
# Helpers
# ------------------------------
def shift_array(arr, shift_x, shift_y):
    """Shift array by (shift_x, shift_y) using roll"""
    return np.roll(np.roll(arr, shift_x, axis=1), shift_y, axis=0)

def max_pool_2d(arr, pool_size=2):
    """Simple 2D max pooling with stride = pool_size"""
    h, w = arr.shape
    out_h, out_w = h // pool_size, w // pool_size
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            window = arr[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(window)

    return output

@st.cache_data
def create_test_patterns():
    """Create various test patterns"""
    patterns = {}

    # Simple cross pattern
    cross = np.zeros((8, 8))
    cross[3:5, :] = 1
    cross[:, 3:5] = 1
    patterns["Cross"] = cross

    # Diagonal pattern
    diag = np.zeros((8, 8))
    for i in range(8):
        diag[i, i] = 1
        if i < 7:
            diag[i, i+1] = 0.5
    patterns["Diagonal"] = diag

    # Square pattern
    square = np.zeros((8, 8))
    square[2:6, 2:6] = 1
    patterns["Square"] = square

    # Random pattern
    np.random.seed(42)
    patterns["Random"] = np.random.rand(8, 8)

    return patterns

@st.cache_data
def create_kernels():
    """Create various convolution kernels"""
    kernels = {}

    # Edge detector (vertical)
    kernels["Vertical Edge"] = np.array([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]])

    # Edge detector (horizontal)
    kernels["Horizontal Edge"] = np.array([[-1, -2, -1],
                                            [ 0,  0,  0],
                                            [ 1,  2,  1]])

    # Sharpening kernel
    kernels["Sharpen"] = np.array([[ 0, -1,  0],
                                    [-1,  5, -1],
                                    [ 0, -1,  0]])

    # Blur kernel
    kernels["Blur"] = np.ones((3, 3)) / 9

    return kernels

def plot_heatmap(ax, data, title, vmin=None, vmax=None, cmap='viridis'):
    """Plot a heatmap with grid and values"""
    h, w = data.shape

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    # Grid
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    # Values
    for i in range(h):
        for j in range(w):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha='center', va='center', fontsize=7)
            text.set_bbox(dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_title(title, fontsize=10, weight='bold')
    ax.axis('off')
    return im

# ------------------------------
# Main UI
# ------------------------------
tab1, tab2 = st.tabs(["Convolution Equivariance", "Max Pooling Invariance"])

# ------------------------------
# TAB 1: Convolution Equivariance
# ------------------------------
with tab1:
    st.header("Convolution is Equivariant to Translation")
    st.markdown("""
    **Equivariance** means: `conv(shift(input)) = shift(conv(input))`

    If you shift the input, then apply convolution, you get the same result as
    applying convolution first, then shifting the output by the same amount.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        patterns = create_test_patterns()
        kernels = create_kernels()

        pattern_name = st.selectbox("Input Pattern", list(patterns.keys()), key="eq_pattern")
        kernel_name = st.selectbox("Kernel", list(kernels.keys()), key="eq_kernel")

        shift_x = st.slider("Shift X (horizontal)", -3, 3, 1, key="eq_shift_x")
        shift_y = st.slider("Shift Y (vertical)", -3, 3, 1, key="eq_shift_y")

        st.markdown(f"""
        **Shift**: ({shift_x}, {shift_y})

        We will compare:
        1. Path A: Shift input → Apply conv
        2. Path B: Apply conv → Shift output

        If equivariance holds, both paths give the same result.
        """)

    with col2:
        input_img = patterns[pattern_name]
        kernel = kernels[kernel_name]

        # Path A: Shift then convolve
        shifted_input = shift_array(input_img, shift_x, shift_y)
        path_a_output = convolve2d(shifted_input, kernel, mode='valid')

        # Path B: Convolve then shift
        conv_output = convolve2d(input_img, kernel, mode='valid')
        path_b_output = shift_array(conv_output, shift_x, shift_y)

        # Compute difference
        difference = np.abs(path_a_output - path_b_output)
        max_diff = difference.max()

        # Visualization - 2 rows, 5 columns
        # Row 1: Input → Shift → Shifted Input → Kernel → Conv(Shifted)
        # Row 2: Input → Kernel → Conv(Input) → Shift → Shifted(Conv)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # Row 1: Path A (S → C)
        plot_heatmap(axes[0, 0], input_img, "Input", cmap='Blues')
        axes[0, 1].text(0.5, 0.5, f'SHIFT\n({shift_x}, {shift_y})',
                       transform=axes[0, 1].transAxes,
                       fontsize=14, weight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[0, 1].axis('off')

        plot_heatmap(axes[0, 2], shifted_input, f"Shifted Input", cmap='Blues')
        plot_heatmap(axes[0, 3], kernel, "Kernel", vmin=-2, vmax=5, cmap='RdBu_r')
        plot_heatmap(axes[0, 4], path_a_output, "Output\n(S → C)", cmap='viridis')

        # Row 2: Path B (C → S)
        plot_heatmap(axes[1, 0], input_img, "Input", cmap='Blues')
        plot_heatmap(axes[1, 1], kernel, "Kernel", vmin=-2, vmax=5, cmap='RdBu_r')
        plot_heatmap(axes[1, 2], conv_output, "Conv Output", cmap='viridis')

        axes[1, 3].text(0.5, 0.5, f'SHIFT\n({shift_x}, {shift_y})',
                       transform=axes[1, 3].transAxes,
                       fontsize=14, weight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[1, 3].axis('off')

        plot_heatmap(axes[1, 4], path_b_output, "Output\n(C → S)", cmap='viridis')

        # Add row labels
        fig.text(0.02, 0.75, 'Path A:\nS → C', fontsize=14, weight='bold',
                va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        fig.text(0.02, 0.25, 'Path B:\nC → S', fontsize=14, weight='bold',
                va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.3))

        plt.tight_layout(rect=[0.03, 0, 1, 1])
        st.pyplot(fig)
        plt.close()

        # Show difference
        fig2, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_heatmap(ax, difference, f"Difference: |Path A - Path B|\nMax: {max_diff:.2e}",
                    vmin=0, vmax=max(1e-10, max_diff), cmap='Reds')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Result
        if max_diff < 1e-10:
            st.success(f"✓ **Equivariance confirmed!** Both paths produce identical outputs (difference < {max_diff:.2e})")
        else:
            st.info(f"Paths differ slightly due to boundary effects (max difference: {max_diff:.4f})")

# ------------------------------
# TAB 2: Max Pooling Invariance
# ------------------------------
with tab2:
    st.header("Max Pooling: Approximate Invariance vs Breaking Equivariance")
    st.markdown("""
    **Key insight from the lecture**: Max pooling has a dual nature:
    - It provides **approximate invariance** when the max value stays within the pooling window
    - But it **breaks equivariance** when shifts cause different values to be selected

    **Example from lecture**: A simple 1D signal shifting by 1 pixel can produce completely different pooling outputs!
    """)

    # Add 1D signal example first
    st.subheader("1D Example: How Max Pooling Breaks Equivariance")
    st.markdown("""
    **Reproducing the lecture example** (slides 104-115):
    A simple signal that goes 0→1→0. Watch what happens when we shift it by 1 pixel!
    """)

    # Create 1D signal (like in PDF)
    signal_1d = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    shift_1d = st.slider("Shift 1D signal by", 0, 3, 1, key="shift_1d")

    # Shift the signal
    shifted_signal_1d = np.roll(signal_1d, shift_1d)

    # Max pool with window size 2, stride 2
    def pool_1d(signal, window=2):
        output = []
        for i in range(0, len(signal), window):
            if i + window <= len(signal):
                output.append(np.max(signal[i:i+window]))
        return np.array(output)

    pooled_original_1d = pool_1d(signal_1d)
    pooled_shifted_1d = pool_1d(shifted_signal_1d)

    # Visualize 1D example
    fig_1d, axes_1d = plt.subplots(2, 2, figsize=(14, 6))

    # Original signal
    axes_1d[0, 0].plot(signal_1d, 'o-', linewidth=2, markersize=8, color='blue')
    axes_1d[0, 0].set_ylim([-0.2, 1.2])
    axes_1d[0, 0].set_title("Original Signal", fontsize=12, weight='bold')
    axes_1d[0, 0].grid(True, alpha=0.3)
    # Show pooling windows
    for i in range(0, len(signal_1d), 2):
        axes_1d[0, 0].axvspan(i-0.5, i+1.5, alpha=0.2, color='red')

    # Pooled original
    x_pooled = np.arange(len(pooled_original_1d)) * 2 + 0.5
    axes_1d[0, 1].plot(x_pooled, pooled_original_1d, 's-', linewidth=2, markersize=10, color='green')
    axes_1d[0, 1].set_ylim([-0.2, 1.2])
    axes_1d[0, 1].set_title("After Max Pool (window=2, stride=2)", fontsize=12, weight='bold')
    axes_1d[0, 1].grid(True, alpha=0.3)

    # Shifted signal
    axes_1d[1, 0].plot(shifted_signal_1d, 'o-', linewidth=2, markersize=8, color='orange')
    axes_1d[1, 0].set_ylim([-0.2, 1.2])
    axes_1d[1, 0].set_title(f"Shifted Signal (shift={shift_1d})", fontsize=12, weight='bold')
    axes_1d[1, 0].grid(True, alpha=0.3)
    # Show pooling windows
    for i in range(0, len(shifted_signal_1d), 2):
        axes_1d[1, 0].axvspan(i-0.5, i+1.5, alpha=0.2, color='red')

    # Pooled shifted
    axes_1d[1, 1].plot(x_pooled, pooled_shifted_1d, 's-', linewidth=2, markersize=10, color='green')
    axes_1d[1, 1].set_ylim([-0.2, 1.2])
    axes_1d[1, 1].set_title("After Max Pool (window=2, stride=2)", fontsize=12, weight='bold')
    axes_1d[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_1d)
    plt.close()

    # Compare outputs
    are_equal = np.array_equal(pooled_original_1d, pooled_shifted_1d)
    if are_equal:
        st.success("✓ Outputs are identical! Invariance holds for this shift.")
    else:
        st.error(f"✗ **Equivariance broken!** Pooled outputs are different despite the input being just shifted.\n\nOriginal pooled: {pooled_original_1d}\n\nShifted pooled: {pooled_shifted_1d}")
        st.markdown("**This demonstrates why max pooling breaks shift equivariance!** The same pattern shifted produces different feature maps.")

    st.markdown("---")

    # 2D Example
    st.subheader("2D Example: How Pooling Creates Translation Invariance")

    # Add visual explanation
    st.info("""
    **The Big Idea (in plain English):**

    Imagine you have a **bright object** in an image and you shift it by a few pixels.

    Without pooling → The output completely changes position
    With pooling → The output stays in roughly the same spot!

    **Why?** Pooling divides the image into big regions (like a grid). As long as the bright object
    stays within the same grid cell, the max value comes from that cell. So small movements don't matter!

    **Try it:** Move the sliders below and watch how the **green star** (max activation) barely moves
    even though the bright feature is shifting around.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Configuration")

        # Create a larger image with a bright feature
        object_type = st.selectbox("Feature Type", ["Bright Spot", "Vertical Edge", "Corner"], key="obj_type")
        pool_size_2d = st.selectbox("Pooling Size", [2, 3, 4], index=1, key="pool_size_2d")

        # Allow shifts within pooling window to demonstrate invariance
        max_shift = pool_size_2d - 1  # Stay within one pooling cell
        shift_x_2d = st.slider("Shift X (horizontal)", 0, max_shift, 0, key="shift_x_2d")
        shift_y_2d = st.slider("Shift Y (vertical)", 0, max_shift, 0, key="shift_y_2d")

        st.markdown(f"""
        **Shift**: ({shift_x_2d}, {shift_y_2d})
        **Pool size**: {pool_size_2d}×{pool_size_2d}

        **Watch**: As you shift the feature, the max value in the pooled output stays in approximately the same location (within the same pooling region or nearby).
        """)

    with col2:
        # Create image size based on pool size - 4 pooling regions per dimension
        img_size = pool_size_2d * 4  # 8x8 for pool=2, 12x12 for pool=3, 16x16 for pool=4
        img_base = np.zeros((img_size, img_size))

        # Place feature in the middle pooling region
        center_region = img_size // 2

        # Add feature based on type
        if object_type == "Bright Spot":
            # Create a bright spot that fits in pooling region
            spot_size = max(2, pool_size_2d - 1)
            start = center_region - spot_size // 2
            img_base[start:start+spot_size, start:start+spot_size] = 0.9
            img_base[start+spot_size//2, start+spot_size//2] = 1.0
            # Add glow
            glow_size = spot_size + 2
            glow_start = start - 1
            img_base[max(0, glow_start):min(img_size, glow_start+glow_size),
                    max(0, glow_start):min(img_size, glow_start+glow_size)] += 0.3

        elif object_type == "Vertical Edge":
            # Create a vertical edge
            edge_len = pool_size_2d * 2
            edge_start = center_region - edge_len // 2
            img_base[edge_start:edge_start+edge_len, center_region:center_region+2] = [[0.7, 0.9]] * edge_len

        else:  # Corner
            # Create an L-shaped corner
            corner_size = pool_size_2d
            img_base[center_region:center_region+corner_size, center_region:center_region+corner_size] = 0.8
            img_base[center_region:center_region+corner_size*2, center_region:center_region+2] = 0.9

        # Add small noise for realism
        img_base += np.random.rand(img_size, img_size) * 0.05
        img_base = np.clip(img_base, 0, 1.0)  # Keep values in [0, 1]

        # Shift the image
        img_shifted = shift_array(img_base, shift_x_2d, shift_y_2d)

        # Max pool both images
        output_original = max_pool_2d(img_base, pool_size_2d)
        output_shifted = max_pool_2d(img_shifted, pool_size_2d)

        # Find locations of max values in pooled outputs
        max_val_orig = output_original.max()
        max_val_shift = output_shifted.max()

        # Get positions of maximum values (convert to regular Python ints)
        max_pos_orig = tuple(int(x) for x in np.unravel_index(np.argmax(output_original), output_original.shape))
        max_pos_shift = tuple(int(x) for x in np.unravel_index(np.argmax(output_shifted), output_shifted.shape))

        # Compute similarity
        difference_inv = np.abs(output_original - output_shifted)
        max_diff = difference_inv.max()
        mean_diff = difference_inv.mean()

        # Create clearer visualization with ALL numbers visible
        h_out, w_out = output_original.shape

        # Create a large, clear figure
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.4)

        # ============ TOP ROW: ORIGINAL ============
        # 1. Original input with ALL values shown
        ax1 = fig.add_subplot(gs[0, 0:2])
        im = ax1.imshow(img_base, cmap='hot', interpolation='nearest', vmin=0, vmax=1.0)

        # Draw pooling grid
        for i in range(h_out + 1):
            ax1.axhline(i * pool_size_2d - 0.5, color='cyan', linewidth=4, alpha=0.9)
        for j in range(w_out + 1):
            ax1.axvline(j * pool_size_2d - 0.5, color='cyan', linewidth=4, alpha=0.9)

        # Highlight max region
        rect = patches.Rectangle((max_pos_orig[1]*pool_size_2d-0.5, max_pos_orig[0]*pool_size_2d-0.5),
                                pool_size_2d, pool_size_2d,
                                edgecolor='lime', linewidth=6, facecolor='none')
        ax1.add_patch(rect)

        # Show ALL pixel values
        for i in range(img_size):
            for j in range(img_size):
                val = img_base[i, j]
                # Check if this pixel is in the highlighted region
                in_highlight = (i >= max_pos_orig[0] * pool_size_2d and
                              i < (max_pos_orig[0] + 1) * pool_size_2d and
                              j >= max_pos_orig[1] * pool_size_2d and
                              j < (max_pos_orig[1] + 1) * pool_size_2d)

                if in_highlight:
                    # Highlighted region - bigger, bold
                    text = ax1.text(j, i, f'{val:.1f}',
                                  ha='center', va='center', fontsize=13, weight='bold',
                                  color='lime')
                    text.set_bbox(dict(boxstyle='round,pad=0.3',
                                      facecolor='black', alpha=0.85,
                                      edgecolor='lime', linewidth=2))
                else:
                    # Other values - smaller
                    text = ax1.text(j, i, f'{val:.1f}',
                                  ha='center', va='center', fontsize=9,
                                  color='white' if val > 0.4 else 'black', alpha=0.8)

        ax1.text(0.5, 1.06, f"STEP 1: Original Image ({img_size}×{img_size}) with {pool_size_2d}×{pool_size_2d} Pooling",
                transform=ax1.transAxes, fontsize=14, weight='bold', ha='center')
        ax1.text(0.5, 1.01, f"Green box = region with max value",
                transform=ax1.transAxes, fontsize=11, ha='center', style='italic', color='lime')
        ax1.set_xticks([])
        ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        # 2. Arrow
        ax_arrow1 = fig.add_subplot(gs[0, 2])
        ax_arrow1.text(0.5, 0.5, '→\nMAX\nPOOL', ha='center', va='center',
                      fontsize=24, weight='bold', color='blue',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax_arrow1.axis('off')

        # 3. Pooled output with ALL values shown
        ax2 = fig.add_subplot(gs[0, 3])
        # Scale up for visibility
        from scipy.ndimage import zoom
        output_orig_large = zoom(output_original, 8, order=0)
        im2 = ax2.imshow(output_orig_large, cmap='hot', interpolation='nearest', vmin=0, vmax=1.0)

        # Show ALL pooled values
        for i in range(h_out):
            for j in range(w_out):
                val = output_original[i, j]
                x_pos = (j + 0.5) * 8
                y_pos = (i + 0.5) * 8

                if (i, j) == max_pos_orig:
                    # Max value - very prominent
                    ax2.plot(x_pos, y_pos, 'w*', markersize=70,
                            markeredgecolor='lime', markeredgewidth=6, zorder=10)
                    text = ax2.text(x_pos, y_pos, f'{val:.1f}',
                                  ha='center', va='center', fontsize=22, weight='bold',
                                  color='lime', zorder=11)
                    text.set_bbox(dict(boxstyle='round,pad=0.5',
                                      facecolor='black', alpha=0.9,
                                      edgecolor='lime', linewidth=4))
                else:
                    # Other values - show clearly but smaller
                    text = ax2.text(x_pos, y_pos, f'{val:.1f}',
                                  ha='center', va='center', fontsize=16,
                                  color='white' if val > 0.4 else 'black',
                                  weight='normal')
                    text.set_bbox(dict(boxstyle='round,pad=0.4',
                                      facecolor='black' if val > 0.4 else 'white',
                                      alpha=0.7, edgecolor='gray', linewidth=2))

        ax2.text(0.5, 1.06, f"STEP 2: Pooled Output ({h_out}×{h_out})",
                transform=ax2.transAxes, fontsize=14, weight='bold', ha='center')
        ax2.text(0.5, 1.01, f"★ = Max value at {max_pos_orig}",
                transform=ax2.transAxes, fontsize=11, ha='center',
                color='lime', weight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor('lime')
            spine.set_linewidth(4)

        # ============ BOTTOM ROW: SHIFTED ============
        # 4. Shifted input with ALL values
        ax3 = fig.add_subplot(gs[1, 0:2])
        im = ax3.imshow(img_shifted, cmap='hot', interpolation='nearest', vmin=0, vmax=1.0)

        # Draw pooling grid
        for i in range(h_out + 1):
            ax3.axhline(i * pool_size_2d - 0.5, color='cyan', linewidth=4, alpha=0.9)
        for j in range(w_out + 1):
            ax3.axvline(j * pool_size_2d - 0.5, color='cyan', linewidth=4, alpha=0.9)

        # Highlight max region
        rect = patches.Rectangle((max_pos_shift[1]*pool_size_2d-0.5, max_pos_shift[0]*pool_size_2d-0.5),
                                pool_size_2d, pool_size_2d,
                                edgecolor='orange', linewidth=6, facecolor='none')
        ax3.add_patch(rect)

        # Show ALL pixel values
        for i in range(img_size):
            for j in range(img_size):
                val = img_shifted[i, j]
                # Check if in highlighted region
                in_highlight = (i >= max_pos_shift[0] * pool_size_2d and
                              i < (max_pos_shift[0] + 1) * pool_size_2d and
                              j >= max_pos_shift[1] * pool_size_2d and
                              j < (max_pos_shift[1] + 1) * pool_size_2d)

                if in_highlight:
                    # Highlighted region - bigger, bold
                    text = ax3.text(j, i, f'{val:.1f}',
                                  ha='center', va='center', fontsize=13, weight='bold',
                                  color='orange')
                    text.set_bbox(dict(boxstyle='round,pad=0.3',
                                      facecolor='black', alpha=0.85,
                                      edgecolor='orange', linewidth=2))
                else:
                    # Other values
                    text = ax3.text(j, i, f'{val:.1f}',
                                  ha='center', va='center', fontsize=9,
                                  color='white' if val > 0.4 else 'black', alpha=0.8)

        ax3.text(0.5, 1.06, f"STEP 3: Shifted Input ({img_size}×{img_size}) - Moved by ({shift_x_2d}, {shift_y_2d}) pixels",
                transform=ax3.transAxes, fontsize=14, weight='bold', ha='center')
        ax3.text(0.5, 1.01, f"Orange = max pooling region",
                transform=ax3.transAxes, fontsize=11, ha='center', style='italic', color='orange')
        ax3.set_xticks([])
        ax3.set_yticks([])
        for spine in ax3.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        # 5. Arrow
        ax_arrow2 = fig.add_subplot(gs[1, 2])
        ax_arrow2.text(0.5, 0.5, '→\nMAX\nPOOL', ha='center', va='center',
                      fontsize=24, weight='bold', color='blue',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax_arrow2.axis('off')

        # 6. Pooled output with ALL values shown
        ax4 = fig.add_subplot(gs[1, 3])
        output_shift_large = zoom(output_shifted, 8, order=0)
        im4 = ax4.imshow(output_shift_large, cmap='hot', interpolation='nearest', vmin=0, vmax=1.0)

        # Show ALL pooled values
        for i in range(h_out):
            for j in range(w_out):
                val = output_shifted[i, j]
                x_pos = (j + 0.5) * 8
                y_pos = (i + 0.5) * 8

                if (i, j) == max_pos_shift:
                    # Max value - very prominent
                    ax4.plot(x_pos, y_pos, 'w*', markersize=70,
                            markeredgecolor='orange', markeredgewidth=6, zorder=10)
                    text = ax4.text(x_pos, y_pos, f'{val:.1f}',
                                  ha='center', va='center', fontsize=22, weight='bold',
                                  color='orange', zorder=11)
                    text.set_bbox(dict(boxstyle='round,pad=0.5',
                                      facecolor='black', alpha=0.9,
                                      edgecolor='orange', linewidth=4))
                else:
                    # Other values - show clearly
                    text = ax4.text(x_pos, y_pos, f'{val:.1f}',
                                  ha='center', va='center', fontsize=16,
                                  color='white' if val > 0.4 else 'black',
                                  weight='normal')
                    text.set_bbox(dict(boxstyle='round,pad=0.4',
                                      facecolor='black' if val > 0.4 else 'white',
                                      alpha=0.7, edgecolor='gray', linewidth=2))

        ax4.text(0.5, 1.06, f"STEP 4: Pooled Output ({h_out}×{h_out})",
                transform=ax4.transAxes, fontsize=14, weight='bold', ha='center')
        ax4.text(0.5, 1.01, f"★ = Max value at {max_pos_shift}",
                transform=ax4.transAxes, fontsize=11, ha='center',
                color='orange', weight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])
        for spine in ax4.spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(4)

        st.pyplot(fig)
        plt.close()

        # Analysis with BIG, CLEAR message
        position_diff = int(np.abs(np.array(max_pos_orig) - np.array(max_pos_shift)).sum())

        st.markdown("---")
        st.markdown("## 📊 What Just Happened?")

        # Create three columns for clear comparison
        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            st.metric("Input Shift", f"({shift_x_2d}, {shift_y_2d}) pixels", delta=None)
            st.caption("How much we moved the bright feature")

        with comp_col2:
            st.metric("Star Position (Top)", f"{max_pos_orig}", delta=None)
            st.caption("Green star location")

        with comp_col3:
            if position_diff == 0:
                st.metric("Star Position (Bottom)", f"{max_pos_shift}", delta="SAME! ✓", delta_color="normal")
            else:
                st.metric("Star Position (Bottom)", f"{max_pos_shift}", delta=f"Moved {position_diff}", delta_color="inverse")
            st.caption("Orange star location")

        # BIG SIMPLE EXPLANATION
        st.markdown("---")
        if position_diff == 0:
            st.success(f"""
            ### 🎯 PERFECT INVARIANCE!

            **What you see:**
            - Top row: Star at position **{max_pos_orig}** (green)
            - Bottom row: Star STILL at position **{max_pos_orig}** (orange)

            **What this means:**
            Even though you shifted the bright feature by **{shift_x_2d}×{shift_y_2d} = {shift_x_2d*shift_y_2d + abs(shift_x_2d-shift_y_2d)} pixels**,
            the star (max activation) stayed in the **exact same spot**!

            **Why?** The bright feature stayed within the same pooling grid cell (the cyan squares).
            Max pooling takes the brightest pixel from each cell, so as long as the bright spot
            doesn't jump to a different cell, the output doesn't change.

            → **This is translation invariance!** Small movements don't affect the output.
            """)
        elif position_diff <= 1:
            st.info(f"""
            ### ✓ APPROXIMATE INVARIANCE

            **What you see:**
            - Top row: Star at position **{max_pos_orig}** (green)
            - Bottom row: Star at position **{max_pos_shift}** (orange)
            - Difference: Only **{position_diff} cell** apart!

            **What this means:**
            You shifted the input by **{shift_x_2d}×{shift_y_2d} pixels**, but the star barely moved!

            **Why?** The bright feature moved just enough to cross into a neighboring pooling cell.
            But it's still very close to where it started.

            → **This is why CNNs are robust!** Small shifts in input → tiny shifts in output.
            """)
        else:
            st.warning(f"""
            ### ⚠️ INVARIANCE BREAKING

            **What you see:**
            - Top row: Star at position **{max_pos_orig}** (green)
            - Bottom row: Star at position **{max_pos_shift}** (orange)
            - Difference: **{position_diff} cells** apart - that's significant!

            **What this means:**
            Your shift of **{shift_x_2d}×{shift_y_2d} pixels** was large enough that the bright feature
            jumped across multiple pooling grid cells.

            **Why?** When features move across many pooling boundaries, invariance breaks down.
            The output changes significantly.

            → **Try smaller shifts** to see invariance in action!
            """)

        # Simple takeaway
        st.markdown("---")
        st.markdown("""
        ### 💡 The Key Takeaway

        **Translation Invariance = Robustness to Small Movements**

        1. **Pooling divides the image into a grid** (cyan lines)
        2. **Each grid cell produces ONE value** (the maximum)
        3. **Small shifts within a cell** → Same output (invariance!)
        4. **Large shifts across cells** → Different output (invariance breaks)

        This is why CNNs can recognize a cat whether it's on the left or right side of the image!
        """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
### Summary (Based on Lecture by Bernhard Kainz)

**Invariance** (e.g., image classification):
- **Definition**: Output stays constant despite input transformation
- **Example**: "cat" label regardless of where cat appears in image
- **Formula**: `f(x) = f(S_v(x))` where S is shift operator

**Equivariance** (e.g., segmentation):
- **Definition**: Output transforms exactly like input
- **Example**: Segmentation mask shifts with object
- **Formula**: `f(S_v(x)) = S_v(f(x))` - f and S commute

**In CNNs:**
- **Convolutional layers**: Shift equivariant (in theory via Fourier transform)
- **Max pooling**: Approximately shift invariant BUT breaks equivariance
- **Problem**: Striding violates Nyquist sampling theorem → aliasing
- **Solution**: Blur before downsampling (anti-aliasing)

**Key Insight**: The 1D example shows max pooling can produce completely different outputs for shifted inputs!

**Beyond Translation:**
- Can extend to **group equivariance** (rotations, etc.)
- Rotation-invariant CNNs use spherical harmonics
- Deformation invariance for small local perturbations

**Reference**: Bernhard Kainz - Deep Learning: Equivariance and Invariance
""")
