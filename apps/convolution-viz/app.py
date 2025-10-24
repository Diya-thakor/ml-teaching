"""
Interactive Convolution Operations Visualizer
Step-by-step visualization of Conv2D, MaxPool, and AvgPool operations
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Set page config
st.set_page_config(
    page_title="Convolution Visualizer",
    page_icon="🔲",
    layout="wide",
)

# High DPI for sharp figures
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'


def conv2d_step(input_matrix, kernel, stride, padding, position):
    """
    Perform one step of convolution at given position.
    Returns the output value and the receptive field coordinates.
    """
    i, j = position
    k_h, k_w = kernel.shape

    # Extract receptive field
    receptive_field = input_matrix[i:i+k_h, j:j+k_w]

    # Element-wise multiplication and sum
    output_val = np.sum(receptive_field * kernel)

    return output_val, (i, j, i+k_h, j+j+k_w)


def get_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size after conv/pool operation"""
    return ((input_size + 2*padding - kernel_size) // stride) + 1


def apply_padding(matrix, padding):
    """Apply zero padding to matrix"""
    if padding == 0:
        return matrix
    return np.pad(matrix, padding, mode='constant', constant_values=0)


def visualize_conv2d_step(input_matrix, kernel, stride, padding, step_idx, total_steps):
    """
    Visualize one step of convolution operation
    """
    # Apply padding
    padded = apply_padding(input_matrix, padding)
    h_in, w_in = padded.shape
    k_h, k_w = kernel.shape

    # Calculate output dimensions
    h_out = get_output_size(h_in, k_h, stride, 0)
    w_out = get_output_size(w_in, k_w, stride, 0)

    # Calculate position for current step
    out_i = step_idx // w_out
    out_j = step_idx % w_out
    in_i = out_i * stride
    in_j = out_j * stride

    # Extract receptive field
    receptive_field = padded[in_i:in_i+k_h, in_j:in_j+k_w]

    # Compute output value
    output_val = np.sum(receptive_field * kernel)

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Input with highlighted receptive field
    ax = axes[0]
    im = ax.imshow(padded, cmap='Blues', alpha=0.6, interpolation='nearest')

    # Highlight receptive field
    rect = patches.Rectangle((in_j-0.5, in_i-0.5), k_w, k_h,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add grid
    for i in range(h_in + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(w_in + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    # Annotate values with better contrast
    for i in range(h_in):
        for j in range(w_in):
            # Add white background for better readability
            text = ax.text(j, i, f'{padded[i,j]:.1f}', ha='center', va='center',
                          fontsize=10, fontweight='bold', color='black')
            text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax.set_title(f'Input (with padding={padding})\nReceptive field at ({in_i},{in_j})', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, w_in - 0.5)
    ax.set_ylim(h_in - 0.5, -0.5)
    ax.axis('off')

    # 2. Kernel
    ax = axes[1]
    im = ax.imshow(kernel, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)

    for i in range(k_h + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(k_w + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    for i in range(k_h):
        for j in range(k_w):
            text = ax.text(j, i, f'{kernel[i,j]:.2f}', ha='center', va='center',
                          fontsize=11, fontweight='bold', color='black')
            text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax.set_title(f'Kernel ({k_h}×{k_w})', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, k_w - 0.5)
    ax.set_ylim(k_h - 0.5, -0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Element-wise multiplication
    ax = axes[2]
    product = receptive_field * kernel
    im = ax.imshow(product, cmap='Greens', interpolation='nearest')

    for i in range(k_h + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(k_w + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    for i in range(k_h):
        for j in range(k_w):
            val = product[i,j]
            text = ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                          fontsize=10, fontweight='bold', color='black')
            text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax.set_title(f'Element-wise Product\nSum = {output_val:.2f}', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, k_w - 0.5)
    ax.set_ylim(k_h - 0.5, -0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Output being built
    ax = axes[3]
    output_matrix = np.full((h_out, w_out), np.nan)

    # Fill in completed positions
    for s in range(step_idx + 1):
        o_i = s // w_out
        o_j = s % w_out
        i_i = o_i * stride
        i_j = o_j * stride
        rf = padded[i_i:i_i+k_h, i_j:i_j+k_w]
        output_matrix[o_i, o_j] = np.sum(rf * kernel)

    # Create masked array for visualization
    masked = np.ma.array(output_matrix, mask=np.isnan(output_matrix))
    im = ax.imshow(masked, cmap='viridis', interpolation='nearest')

    for i in range(h_out + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(w_out + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    for i in range(h_out):
        for j in range(w_out):
            if not np.isnan(output_matrix[i,j]):
                # Always use black text with white background for maximum contrast
                text = ax.text(j, i, f'{output_matrix[i,j]:.1f}',
                              ha='center', va='center', fontsize=11, color='black', fontweight='bold')
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black',
                                  alpha=0.9, linewidth=1.5))

    # Highlight current position
    rect = patches.Rectangle((out_j-0.5, out_i-0.5), 1, 1,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.set_title(f'Output ({h_out}×{w_out})\nStep {step_idx+1}/{total_steps}', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, w_out - 0.5)
    ax.set_ylim(h_out - 0.5, -0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig


def visualize_pooling_step(input_matrix, pool_size, stride, pool_type, step_idx, total_steps):
    """
    Visualize one step of pooling operation
    """
    h_in, w_in = input_matrix.shape
    p_h, p_w = pool_size

    # Calculate output dimensions
    h_out = get_output_size(h_in, p_h, stride, 0)
    w_out = get_output_size(w_in, p_w, stride, 0)

    # Calculate position for current step
    out_i = step_idx // w_out
    out_j = step_idx % w_out
    in_i = out_i * stride
    in_j = out_j * stride

    # Extract pooling region
    pool_region = input_matrix[in_i:in_i+p_h, in_j:in_j+p_w]

    # Compute output value
    if pool_type == "MaxPool":
        output_val = np.max(pool_region)
        operation = "max"
    else:  # AvgPool
        output_val = np.mean(pool_region)
        operation = "mean"

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Input with highlighted pool region
    ax = axes[0]
    im = ax.imshow(input_matrix, cmap='Blues', alpha=0.6, interpolation='nearest')

    # Highlight pool region
    rect = patches.Rectangle((in_j-0.5, in_i-0.5), p_w, p_h,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add grid
    for i in range(h_in + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(w_in + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    # Annotate values with better contrast
    for i in range(h_in):
        for j in range(w_in):
            text = ax.text(j, i, f'{input_matrix[i,j]:.1f}', ha='center', va='center',
                          fontsize=10, fontweight='bold', color='black')
            text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax.set_title(f'Input ({h_in}×{w_in})\nPool region at ({in_i},{in_j})', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, w_in - 0.5)
    ax.set_ylim(h_in - 0.5, -0.5)
    ax.axis('off')

    # 2. Pool region detail
    ax = axes[1]
    im = ax.imshow(pool_region, cmap='Greens', interpolation='nearest')

    for i in range(p_h + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(p_w + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    for i in range(p_h):
        for j in range(p_w):
            val = pool_region[i,j]
            # Highlight max value for MaxPool
            if pool_type == "MaxPool" and val == output_val:
                color = 'red'
                bgcolor = 'yellow'
                edgecolor = 'red'
                linewidth = 2
            else:
                color = 'black'
                bgcolor = 'white'
                edgecolor = 'none'
                linewidth = 0
            text = ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                          fontsize=11, color=color, fontweight='bold')
            text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor=bgcolor,
                              edgecolor=edgecolor, alpha=0.9, linewidth=linewidth))

    ax.set_title(f'Pool Region ({p_h}×{p_w})\n{operation}({pool_region.ravel()}) = {output_val:.2f}',
                fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, p_w - 0.5)
    ax.set_ylim(p_h - 0.5, -0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Output being built
    ax = axes[2]
    output_matrix = np.full((h_out, w_out), np.nan)

    # Fill in completed positions
    for s in range(step_idx + 1):
        o_i = s // w_out
        o_j = s % w_out
        i_i = o_i * stride
        i_j = o_j * stride
        region = input_matrix[i_i:i_i+p_h, i_j:i_j+p_w]
        if pool_type == "MaxPool":
            output_matrix[o_i, o_j] = np.max(region)
        else:
            output_matrix[o_i, o_j] = np.mean(region)

    # Create masked array for visualization
    masked = np.ma.array(output_matrix, mask=np.isnan(output_matrix))
    im = ax.imshow(masked, cmap='viridis', interpolation='nearest')

    for i in range(h_out + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(w_out + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    for i in range(h_out):
        for j in range(w_out):
            if not np.isnan(output_matrix[i,j]):
                text = ax.text(j, i, f'{output_matrix[i,j]:.1f}',
                       ha='center', va='center', fontsize=11, color='black', fontweight='bold')
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='black', alpha=0.9, linewidth=1))

    # Highlight current position
    rect = patches.Rectangle((out_j-0.5, out_i-0.5), 1, 1,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.set_title(f'Output ({h_out}×{w_out})\nStep {step_idx+1}/{total_steps}', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, w_out - 0.5)
    ax.set_ylim(h_out - 0.5, -0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig


def main():
    st.title("Convolution & Pooling Visualizer")
    st.markdown("""
    **Interactive step-by-step visualization** of convolution and pooling operations.
    Watch how the output is built one position at a time!
    """)

    # Sidebar controls
    st.sidebar.header("Operation Settings")

    # Operation type
    operation = st.sidebar.selectbox(
        "Operation Type",
        ["Conv2D", "MaxPool2D", "AvgPool2D"]
    )

    # Input matrix settings
    st.sidebar.subheader("Input Matrix")
    input_size = st.sidebar.slider("Input Size", 4, 10, 6)

    # Initialize or regenerate input matrix
    if 'input_matrix' not in st.session_state or st.sidebar.button("Randomize Input"):
        st.session_state.input_matrix = np.random.randint(0, 10, size=(input_size, input_size)).astype(float)

    input_matrix = st.session_state.input_matrix

    # Resize if input_size changed
    if input_matrix.shape[0] != input_size:
        st.session_state.input_matrix = np.random.randint(0, 10, size=(input_size, input_size)).astype(float)
        input_matrix = st.session_state.input_matrix

    if operation == "Conv2D":
        # Convolution parameters
        st.sidebar.subheader("Convolution Parameters")
        kernel_size = st.sidebar.slider("Kernel Size", 2, 5, 3)
        stride = st.sidebar.slider("Stride", 1, 3, 1)
        padding = st.sidebar.slider("Padding", 0, 2, 0)

        # Kernel initialization
        st.sidebar.subheader("Kernel Values")
        kernel_type = st.sidebar.selectbox("Kernel Type",
                                          ["Edge Detection (Horizontal)",
                                           "Edge Detection (Vertical)",
                                           "Sharpen",
                                           "Blur",
                                           "Random"])

        if kernel_type == "Edge Detection (Horizontal)":
            if kernel_size == 3:
                kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            else:
                kernel = np.random.randn(kernel_size, kernel_size)
        elif kernel_type == "Edge Detection (Vertical)":
            if kernel_size == 3:
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            else:
                kernel = np.random.randn(kernel_size, kernel_size)
        elif kernel_type == "Sharpen":
            if kernel_size == 3:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            else:
                kernel = np.random.randn(kernel_size, kernel_size)
        elif kernel_type == "Blur":
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        else:  # Random
            if 'kernel' not in st.session_state or st.sidebar.button("Randomize Kernel"):
                st.session_state.kernel = np.random.randn(kernel_size, kernel_size)
            kernel = st.session_state.kernel
            if kernel.shape[0] != kernel_size:
                st.session_state.kernel = np.random.randn(kernel_size, kernel_size)
                kernel = st.session_state.kernel

        # Calculate dimensions
        padded = apply_padding(input_matrix, padding)
        h_out = get_output_size(padded.shape[0], kernel_size, stride, 0)
        w_out = get_output_size(padded.shape[1], kernel_size, stride, 0)
        total_steps = h_out * w_out

        # Step selector
        st.sidebar.subheader("Step Control")
        step_idx = st.sidebar.slider("Step", 0, total_steps - 1, 0)

        # Auto-play
        auto_play = st.sidebar.checkbox("Auto-play", value=False)
        if auto_play:
            import time
            if step_idx < total_steps - 1:
                time.sleep(0.5)
                st.session_state.step_idx = step_idx + 1
                st.rerun()

        # Info
        st.sidebar.markdown(f"""
        **Dimensions:**
        - Input: {input_matrix.shape[0]}×{input_matrix.shape[1]}
        - Padded: {padded.shape[0]}×{padded.shape[1]}
        - Kernel: {kernel_size}×{kernel_size}
        - Output: {h_out}×{w_out}
        - Total steps: {total_steps}
        """)

        # Explanation
        st.markdown("""
        ### How Conv2D Works

        1. **Slide** the kernel over the input (with stride)
        2. **Multiply** element-wise at each position
        3. **Sum** all products to get one output value
        4. **Repeat** for all positions
        """)

        # Visualization
        fig = visualize_conv2d_step(input_matrix, kernel, stride, padding, step_idx, total_steps)
        st.pyplot(fig)
        plt.close()

    else:  # Pooling
        pool_type = "MaxPool" if operation == "MaxPool2D" else "AvgPool"

        # Pooling parameters
        st.sidebar.subheader("Pooling Parameters")
        pool_size = st.sidebar.slider("Pool Size", 2, 4, 2)
        stride = st.sidebar.slider("Stride", 1, 3, 2)

        # Calculate dimensions
        h_out = get_output_size(input_matrix.shape[0], pool_size, stride, 0)
        w_out = get_output_size(input_matrix.shape[1], pool_size, stride, 0)
        total_steps = h_out * w_out

        # Step selector
        st.sidebar.subheader("Step Control")
        step_idx = st.sidebar.slider("Step", 0, total_steps - 1, 0)

        # Info
        st.sidebar.markdown(f"""
        **Dimensions:**
        - Input: {input_matrix.shape[0]}×{input_matrix.shape[1]}
        - Pool: {pool_size}×{pool_size}
        - Output: {h_out}×{w_out}
        - Total steps: {total_steps}
        """)

        # Explanation
        if pool_type == "MaxPool":
            st.markdown("""
            ### How MaxPool Works

            1. **Slide** a window over the input
            2. **Take maximum** value in each window
            3. **Output** the max value at that position
            4. **Repeat** for all positions

            **Purpose:** Downsampling + keeping strongest activations
            """)
        else:
            st.markdown("""
            ### How AvgPool Works

            1. **Slide** a window over the input
            2. **Compute average** of values in each window
            3. **Output** the average at that position
            4. **Repeat** for all positions

            **Purpose:** Smooth downsampling
            """)

        # Visualization
        fig = visualize_pooling_step(input_matrix, (pool_size, pool_size), stride, pool_type, step_idx, total_steps)
        st.pyplot(fig)
        plt.close()

    # Formula reference
    with st.expander("Mathematical Formulas"):
        st.markdown("""
        ### Output Size Formula

        For both convolution and pooling:

        **Output Size = ⌊(Input Size + 2×Padding - Kernel Size) / Stride⌋ + 1**

        ### Conv2D Operation

        At position (i, j):

        **Output[i,j] = Σ Σ Input[i×stride+m, j×stride+n] × Kernel[m,n]**

        ### MaxPool Operation

        **Output[i,j] = max(Pool Region)**

        ### AvgPool Operation

        **Output[i,j] = mean(Pool Region)**
        """)


if __name__ == "__main__":
    main()
