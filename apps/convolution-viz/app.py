import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------
# Page config and style
# ----------------------------------
st.set_page_config(page_title="Multi-Channel Conv2D Visualizer", layout="wide")
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'

st.title("🎨 Multi-Channel Convolution Visualizer")
st.markdown("""
Step-by-step visualization of **multi-channel Conv2D**.  
At each step, you’ll see how the output cell is computed by combining all input-channel contributions.
""")

# ----------------------------------
# Helpers
# ----------------------------------
def get_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1

def apply_padding(matrix, padding):
    if padding == 0:
        return matrix
    return np.pad(matrix, padding, mode='constant', constant_values=0)

@st.cache_data
def generate_data(h, w, c_in, c_out, k):
    np.random.seed(0)
    x = np.random.randint(0, 9, (c_in, h, w)).astype(float)
    kx = np.random.randn(c_out, c_in, k, k)
    return x, kx

# ----------------------------------
# Visualization
# ----------------------------------
def visualize_step(input_tensor, kernels, stride, padding, c_out_idx, step_idx):
    C_in, H, W = input_tensor.shape
    C_out, _, K, _ = kernels.shape

    padded = np.stack([apply_padding(input_tensor[c], padding) for c in range(C_in)])
    _, Hp, Wp = padded.shape
    H_out = get_output_size(Hp, K, stride, 0)
    W_out = get_output_size(Wp, K, stride, 0)

    oi, oj = divmod(step_idx, W_out)
    ii, jj = oi * stride, oj * stride
    kernel = kernels[c_out_idx]

    # Compute per-channel pieces
    pieces = []
    val_sum = 0.0
    for c in range(C_in):
        rf = padded[c, ii:ii+K, jj:jj+K]
        prod = rf * kernel[c]
        val = prod.sum()
        val_sum += val
        pieces.append((rf, kernel[c], val, prod))

    # --- Figure layout ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap_in = 'Blues'
    cmap_k = 'RdBu_r'

    # (1) Input composite (stacked channels, each tinted differently)
    colors = ['Reds', 'Greens', 'Blues']
    ax = axes[0]
    for c in range(C_in):
        ax.imshow(padded[c], cmap=colors[c % len(colors)], alpha=0.4)
    rect = patches.Rectangle((jj-0.5, ii-0.5), K, K, linewidth=3,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"All Input Channels (padding={padding})", fontsize=10)
    for i in range(Hp + 1):
        ax.axhline(i - 0.5, color='gray', lw=0.5)
    for j in range(Wp + 1):
        ax.axvline(j - 0.5, color='gray', lw=0.5)
    for i in range(Hp):
        for j in range(Wp):
            t = ax.text(j, i, f'{padded[0,i,j]:.1f}', ha='center', va='center', fontsize=8)
            t.set_bbox(dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.axis('off')

    # (2) Kernel slice of chosen output channel
    ax = axes[1]
    im = ax.imshow(kernel.sum(0), cmap=cmap_k, interpolation='nearest', vmin=-1, vmax=1)
    ax.set_title(f"Kernel (summed over {C_in} channels)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(K + 1):
        ax.axhline(i - 0.5, color='gray', lw=0.5)
        ax.axvline(i - 0.5, color='gray', lw=0.5)
    ax.axis('off')

    # (3) Output matrix partially filled
    ax = axes[2]
    output_h, output_w = H_out, W_out
    output = np.full((output_h, output_w), np.nan)
    for s in range(step_idx + 1):
        oi2, oj2 = divmod(s, output_w)
        ii2, jj2 = oi2 * stride, oj2 * stride
        rf = padded[:, ii2:ii2+K, jj2:jj2+K]
        output[oi2, oj2] = np.sum(rf * kernel)
    masked = np.ma.array(output, mask=np.isnan(output))
    im = ax.imshow(masked, cmap='viridis', interpolation='nearest')
    for i in range(output_h + 1):
        ax.axhline(i - 0.5, color='gray', lw=0.5)
    for j in range(output_w + 1):
        ax.axvline(j - 0.5, color='gray', lw=0.5)
    # text boxes
    for i in range(output_h):
        for j in range(output_w):
            if not np.isnan(output[i, j]):
                val = output[i, j]
                text = ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10, color='black')
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='black', alpha=0.9, lw=1))
    rect = patches.Rectangle((oj-0.5, oi-0.5), 1, 1, lw=3, ec='red', fc='none')
    ax.add_patch(rect)
    ax.set_title(f'Output {output_h}×{output_w}\nStep {step_idx+1}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    # Expression text
    expr_terms = []
    for c in range(C_in):
        expr_terms.append(f"Σ(Ch {c})={pieces[c][2]:.2f}")
    expr = " + ".join(expr_terms) + f" → **{val_sum:.2f}**"
    st.markdown(f"### Output[{c_out_idx},{oi},{oj}] = {expr}")

    return fig

# ----------------------------------
# Sidebar controls
# ----------------------------------
st.sidebar.header("Configuration")
H = st.sidebar.slider("Input Height", 4, 8, 5)
W = st.sidebar.slider("Input Width", 4, 8, 5)
C_in = st.sidebar.selectbox("Input Channels", [1, 3], index=1)
C_out = st.sidebar.slider("Output Channels", 1, 3, 1)
K = st.sidebar.slider("Kernel Size", 2, 4, 3)
stride = st.sidebar.slider("Stride", 1, 2, 1)
padding = st.sidebar.slider("Padding", 0, 2, 0)

x, kx = generate_data(H, W, C_in, C_out, K)
Hp, Wp = H + 2*padding, W + 2*padding
H_out = get_output_size(Hp, K, stride, 0)
W_out = get_output_size(Wp, K, stride, 0)
total_steps = H_out * W_out

st.sidebar.markdown(f"**Output dims:** {C_out} × {H_out} × {W_out}")
c_out_idx = st.sidebar.selectbox("Output Channel", list(range(C_out)))
step_idx = st.sidebar.slider("Step", 0, total_steps - 1, 0)

# ----------------------------------
# Main view
# ----------------------------------
st.header(f"Computing Output Channel {c_out_idx}, Step {step_idx+1}/{total_steps}")
fig = visualize_step(x, kx, stride, padding, c_out_idx, step_idx)
st.pyplot(fig)
plt.close(fig)
