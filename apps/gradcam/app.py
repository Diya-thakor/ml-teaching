"""
Interactive Grad-CAM Visualization App
A blog-post style walkthrough of Gradient-weighted Class Activation Mapping
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# Set page config
st.set_page_config(
    page_title="Grad-CAM Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib DPI for high resolution figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# Define LeNet model (same as training)
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        self.features = x
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@st.cache_resource
def load_model():
    """Load the pre-trained LeNet model"""
    model = LeNet()
    model.load_state_dict(torch.load('lenet_mnist.pth', map_location='cpu'))
    model.eval()
    return model


@st.cache_data
def load_test_dataset():
    """Load MNIST test dataset"""
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_ds


def compute_gradcam_steps(model, img, target_class=None):
    """
    Compute Grad-CAM with intermediate steps for visualization
    Returns dict with all intermediate values
    """
    # Storage for feature maps
    features = []

    def hook_fn(module, input, output):
        features.append(output)

    # Register hook on conv2
    handle = model.conv2.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    img_batch = img.unsqueeze(0)
    output = model(img_batch)

    # Determine target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Get conv2 features (8x8)
    conv2_features = features[0]
    conv2_features.retain_grad()

    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()

    # Get gradients
    gradients = conv2_features.grad

    # Compute weights (global average pooling)
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted sum + ReLU
    feature_maps = conv2_features.detach()
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample (bicubic gives smoother heatmaps than bilinear)
    cam_upsampled = F.interpolate(cam, size=(28, 28), mode='bicubic', align_corners=False)
    cam_normalized = cam_upsampled.squeeze().cpu()
    cam_normalized = (cam_normalized - cam_normalized.min()) / (cam_normalized.max() - cam_normalized.min() + 1e-8)

    handle.remove()

    return {
        'output': output,
        'pred_class': target_class,
        'conv2_features': conv2_features.detach().cpu(),
        'gradients': gradients.cpu(),
        'weights': weights.cpu(),
        'cam_raw': cam.squeeze().cpu(),
        'cam_normalized': cam_normalized,
        'class_scores': output[0].detach().cpu()
    }


# Main app
def main():
    st.title("Grad-CAM: Understanding CNN Decisions")
    st.markdown("""
    **Gradient-weighted Class Activation Mapping** is a powerful visualization technique
    that shows *where* a CNN is looking when making predictions.

    This interactive app walks you through each step of the Grad-CAM algorithm!
    """)

    # Load model and data
    with st.spinner("Loading model..."):
        model = load_model()

    with st.spinner("Loading MNIST dataset..."):
        test_ds = load_test_dataset()

    st.success("Model and data loaded!")

    # Show model architecture
    with st.expander("Model Architecture", expanded=False):
        st.markdown("""
        **LeNet Architecture for MNIST:**

        | Layer | Operation | Input Shape | Output Shape |
        |-------|-----------|-------------|--------------|
        | Input | - | (1, 28, 28) | (1, 28, 28) |
        | Conv1 | 5×5, 6 filters | (1, 28, 28) | (6, 24, 24) |
        | ReLU + MaxPool | 2×2 | (6, 24, 24) | (6, 12, 12) |
        | Conv2 | 5×5, 16 filters | (6, 12, 12) | **(16, 8, 8)** ← Grad-CAM |
        | ReLU + MaxPool | 2×2 | (16, 8, 8) | (16, 4, 4) |
        | Flatten | - | (16, 4, 4) | (256,) |
        | FC1 + ReLU | 256 → 120 | (256,) | (120,) |
        | FC2 + ReLU | 120 → 84 | (120,) | (84,) |
        | FC3 (output) | 84 → 10 | (84,) | (10,) |

        **Total parameters:** ~61K

        We hook **Conv2** output (before pooling) to get 8×8 feature maps with spatial detail.
        """)

    # Sidebar for settings
    st.sidebar.header("Settings")

    # Image selection mode
    selection_mode = st.sidebar.radio("Image selection mode:", ["Grid", "Index"])

    if selection_mode == "Grid":
        st.sidebar.markdown("### Select an image from the grid below")
        # Show grid of images in main area (will be added later)
        img_idx = st.session_state.get('selected_img_idx', 0)
    else:
        img_idx = st.sidebar.number_input(
            "Select image index (0-9999)",
            min_value=0,
            max_value=len(test_ds)-1,
            value=0
        )

    # Show image grid if in grid mode
    if selection_mode == "Grid":
        st.subheader("Select an Image")
        start_idx = st.number_input("Starting index:", min_value=0, max_value=9980, value=0, step=20)

        # Show 20 images in 2 rows of 10
        for row in range(2):
            cols = st.columns(10)
            for col in range(10):
                i = row * 10 + col
                idx = start_idx + i
                if idx < len(test_ds):
                    img_preview, label_preview = test_ds[idx]
                    with cols[col]:
                        fig, ax = plt.subplots(figsize=(0.5, 0.5))
                        ax.imshow(img_preview.squeeze().numpy(), cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=False)
                        plt.close()
                        if st.button(f"{idx}", key=f"btn_{idx}", use_container_width=True):
                            st.session_state['selected_img_idx'] = idx
                            img_idx = idx
                            st.rerun()

        st.markdown("---")

    img, true_label = test_ds[img_idx]

    # Get prediction
    with torch.no_grad():
        pred_output = model(img.unsqueeze(0))
        pred_class = pred_output.argmax(dim=1).item()
        pred_prob = F.softmax(pred_output, dim=1)[0, pred_class].item()

    # Target class selection
    use_pred_class = st.sidebar.checkbox("Use predicted class", value=True)
    if use_pred_class:
        target_class = pred_class
    else:
        target_class = st.sidebar.selectbox("Choose target class", range(10), index=pred_class)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**True Label:** {true_label}")
    st.sidebar.markdown(f"**Predicted:** {pred_class} ({pred_prob*100:.1f}%)")
    st.sidebar.markdown(f"**Explaining:** Class {target_class}")

    # Compute Grad-CAM
    with st.spinner("Computing Grad-CAM..."):
        results = compute_gradcam_steps(model, img, target_class)

    # Display original image
    st.header("Input Image")
    st.markdown("""
    **Goal:** Understand where the model looks when predicting class {target_class}.

    **Grad-CAM Pipeline:** Forward pass → Gradients → Weight channels → Create heatmap
    """.format(target_class=target_class))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(img.squeeze().numpy(), cmap='gray', interpolation='nearest')
        ax.set_title(f"True: {true_label} | Pred: {pred_class}", fontsize=11)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Step 1: Show all 16 feature maps
    st.header("Step 1: Conv2 Feature Maps (8×8)")
    st.markdown("""
    Conv2 is the **last convolutional layer** with 16 channels. Each channel detects different patterns
    (edges, curves, digit-specific features). We hook before pooling to preserve 8×8 spatial detail.
    """)

    with st.expander("Show code for Step 1", expanded=False):
        st.code("""
# Forward pass through the network
features = []

def hook_fn(module, input, output):
    features.append(output)

# Hook conv2 layer (before pooling)
handle = model.conv2.register_forward_hook(hook_fn)
output = model(img.unsqueeze(0))

# features[0] contains conv2 output: shape [1, 16, 8, 8]
conv2_features = features[0][0]  # [16, 8, 8]
        """, language="python")

    conv2_features = results['conv2_features'][0]  # [16, 8, 8]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = i // 4, i % 4
        im = axes[row, col].imshow(conv2_features[i].numpy(), cmap='viridis', interpolation='nearest')
        axes[row, col].set_title(f"Ch {i}", fontsize=9)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Step 2: Show gradients
    st.header("Step 2: Gradients ∂(class {target_class})/∂(features)".format(target_class=target_class))
    st.markdown(f"""
    Gradients measure **sensitivity**: how much each feature pixel affects the class {target_class} score.

    **Red** = positive influence | **Blue** = negative | **White** = neutral
    """)

    with st.expander("Show code for Step 2", expanded=False):
        st.code(f"""
# Enable gradient retention
conv2_features.retain_grad()

# Backward pass for target class
model.zero_grad()
output[0, {target_class}].backward()

# Get gradients: shape [1, 16, 8, 8]
gradients = conv2_features.grad
        """, language="python")

    gradients = results['gradients'][0]  # [16, 8, 8]

    # Show all 16 channels
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = i // 4, i % 4
        grad = gradients[i].numpy()
        vmax = np.abs(grad).max()
        im = axes[row, col].imshow(grad, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[row, col].set_title(f"Grad Ch {i}", fontsize=9)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Step 2.5: Detailed example of averaging for channel 0
    st.header("Step 2.5: Example - Averaging Channel 0")
    st.markdown("""
    **Problem:** 1,024 gradient values (16 channels × 8×8) is too much!

    **Solution:** Average each channel's 64 values → 16 importance weights (α values)
    """)

    with st.expander("Show code for averaging", expanded=False):
        st.code("""
# Take channel 0 gradient map (8x8)
grad_channel_0 = gradients[0]  # shape: [8, 8]

# Compute mean across spatial dimensions
alpha_0 = grad_channel_0.mean()

# This gives us one importance weight for channel 0
        """, language="python")

    grad_ch0 = gradients[0].numpy()  # [8, 8]
    alpha_ch0 = grad_ch0.mean()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Channel 0 Gradients (8×8)")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(grad_ch0, cmap='RdBu_r', vmin=-np.abs(grad_ch0).max(), vmax=np.abs(grad_ch0).max(), interpolation='nearest')

        # Annotate with values
        for i in range(8):
            for j in range(8):
                text = ax.text(j, i, f'{grad_ch0[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=6.5)

        ax.set_title("64 gradient values", fontsize=10)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Computing the Average")
        st.markdown(f"""
        **Formula:**
        ```
        α_0 = (1/64) × Σ(all 64 gradient values)
        ```

        **Calculation:**
        - Sum of all values: `{grad_ch0.sum():.6f}`
        - Number of values: `64` (8×8)
        - **α_0 = {alpha_ch0:.6f}**
        """)

        st.markdown("---")
        st.markdown("**Interpretation:**")
        if alpha_ch0 > 0:
            st.markdown(f"✓ **Positive weight ({alpha_ch0:.6f})**: Channel 0 features **support** class {target_class}")
        else:
            st.markdown(f"✗ **Negative weight ({alpha_ch0:.6f})**: Channel 0 features **oppose** class {target_class}")

    st.markdown("---")

    # Step 3: Show all channel weights
    st.header("Step 3: Channel Importance Weights")
    st.markdown("""
    Repeat averaging for **all 16 channels**: [16, 8, 8] gradients → [16] weights (α values)

    **Green** = supports class {target_class} | **Red** = opposes | Channel 0 highlighted in blue
    """.format(target_class=target_class))

    with st.expander("Show code for Step 3", expanded=False):
        st.code("""
# Compute weights for all channels using global average pooling
# gradients shape: [16, 8, 8]
weights = gradients.mean(dim=(1, 2))  # Average over H and W dimensions

# Result: [16] - one weight per channel
# weights[0] = α_0, weights[1] = α_1, ..., weights[15] = α_15
        """, language="python")

    weights = results['weights'][0, :, 0, 0].numpy()  # [16]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ['green' if w > 0 else 'red' for w in weights]
        bars = ax.bar(range(16), weights, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Channel", fontsize=10)
        ax.set_ylabel("α_k", fontsize=10)
        ax.set_title(f"Channel Importance (Class {target_class})", fontsize=11, fontweight='bold')
        ax.set_xticks(range(16))
        ax.grid(axis='y', alpha=0.3)

        # Highlight channel 0
        bars[0].set_edgecolor('blue')
        bars[0].set_linewidth(2.5)
        ax.legend([bars[0]], ['Ch 0'], loc='best', fontsize=9)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### All Channel Weights")
        for i in range(16):
            st.markdown(f"**α_{i:2d}** = `{weights[i]:7.4f}`")

    st.markdown("---")

    # Show top channels
    st.markdown("### Most Important Channels")
    sorted_indices = np.argsort(np.abs(weights))[::-1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Positive Contributors:**")
        pos_sorted = np.argsort(weights)[::-1]
        for i in range(min(5, len(pos_sorted))):
            ch = pos_sorted[i]
            if weights[ch] > 0:
                st.markdown(f"{i+1}. Channel {ch}: `{weights[ch]:.4f}`")

    with col2:
        st.markdown("**Top 5 Negative Contributors:**")
        neg_sorted = np.argsort(weights)
        for i in range(min(5, len(neg_sorted))):
            ch = neg_sorted[i]
            if weights[ch] < 0:
                st.markdown(f"{i+1}. Channel {ch}: `{weights[ch]:.4f}`")

    st.markdown("---")

    # Step 4: Weighted sum
    st.header("Step 4: Weighted Sum → Heatmap")
    st.markdown("""
    Combine feature maps weighted by importance: **CAM = ReLU(Σ α_k × feature_k)**

    Channels with large |α_k| contribute more. ReLU keeps only positive contributions.
    """)

    with st.expander("Show code for Step 4", expanded=False):
        st.code("""
# Weighted sum across channels
# weights shape: [16, 1, 1]
# feature_maps shape: [16, 8, 8]
cam = (weights.unsqueeze(-1).unsqueeze(-1) * feature_maps).sum(dim=0)

# Apply ReLU to keep only positive contributions
cam = F.relu(cam)  # shape: [8, 8]
        """, language="python")

    cam_raw = results['cam_raw'].numpy()  # [8, 8]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cam_raw, cmap='jet', interpolation='nearest')
        ax.set_title("Raw Grad-CAM (8×8)", fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()

    with col2:
        cam_norm = results['cam_normalized'].numpy()  # [28, 28]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cam_norm, cmap='jet', interpolation='bilinear')
        ax.set_title("Upsampled Grad-CAM (28×28)", fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()
        st.caption("Bicubic interpolation for smooth heatmap")

    st.markdown("---")

    # Step 5: Final visualization
    st.header("Step 5: Final Visualization")
    st.markdown(f"""
    Overlay the heatmap on the original image. **Warm colors** = high importance for class {target_class}.
    """)


    with st.expander("Show code for Step 5", expanded=False):
        st.code("""
# Upsample CAM to input size
cam_upsampled = F.interpolate(
    cam.unsqueeze(0).unsqueeze(0),
    size=(28, 28),
    mode='bilinear',
    align_corners=False
)

# Normalize to [0, 1]
cam_normalized = (cam_upsampled - cam_upsampled.min()) / \\
                 (cam_upsampled.max() - cam_upsampled.min())

# Overlay on original image
plt.imshow(original_image, cmap='gray')
plt.imshow(cam_normalized, cmap='jet', alpha=0.5)
        """, language="python")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Original
    axes[0].imshow(img.squeeze().numpy(), cmap='gray', interpolation='nearest')
    axes[0].set_title(f"Original\nTrue: {true_label}, Pred: {pred_class}", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(results['cam_normalized'].numpy(), cmap='jet', interpolation='bilinear')
    axes[1].set_title(f"Grad-CAM for Class {target_class}", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img.squeeze().numpy(), cmap='gray', interpolation='nearest')
    axes[2].imshow(results['cam_normalized'].numpy(), cmap='jet', alpha=0.5, interpolation='bilinear')
    axes[2].set_title("Overlay", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Class scores
    st.header("All Class Scores (Logits)")
    st.markdown("""
    Model outputs 10 scores (higher = more confident). **Green** = predicted, **Orange** = target
    """)

    class_scores = results['class_scores'].numpy()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ['green' if i == pred_class else 'lightblue' for i in range(10)]
    bars = ax.bar(range(10), class_scores, color=colors, edgecolor='black', alpha=0.8)
    bars[target_class].set_color('orange')
    bars[target_class].set_label(f'Target: {target_class}')

    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Output", fontsize=11, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close()

    # Footer
    st.markdown("---")
    with st.expander("Summary & References"):
        st.markdown("""
        ### Algorithm Recap

        1. Forward pass → capture conv2 features (16 × 8×8)
        2. Backward pass → compute gradients
        3. Global average pooling → 16 channel weights
        4. Weighted sum + ReLU → 8×8 heatmap
        5. Upsample (bicubic) → overlay on input

        ### Why Grad-CAM?

        - **Interpretability**: See where models look
        - **Debugging**: Catch spurious correlations
        - **Trust**: Verify learned features

        ### References

        **Paper:** Selvaraju et al. (2017) - [Grad-CAM](https://arxiv.org/abs/1610.02391)

        **Variants:** Grad-CAM++, Score-CAM, Eigen-CAM

        **Try:** Different images, counterfactual classes!
        """)


if __name__ == "__main__":
    main()
