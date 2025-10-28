"""
Simple Image Clustering Demo
Progressive complexity: RGB → Position → Combined → ResNet18 features

Educational app showing different clustering approaches:
1. Color-based (RGB values)
2. Position-based (X, Y coordinates)
3. Combined (RGB + XY)
4. Deep features (ResNet18)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Page config
st.set_page_config(page_title="Simple Clustering Demo", layout="wide")

st.title("🎨 Image Clustering: From Simple to Deep")
st.markdown("""
**Learn clustering progressively!** Start with simple color/position clustering, then see how deep learning improves it.
""")

# Cache models
@st.cache_resource
def load_resnet18_backbone(layer='layer4'):
    """
    Load pretrained ResNet18 backbone up to specified layer

    layer options:
    - 'layer1': stride 4, 64 channels, high res but less semantic
    - 'layer2': stride 8, 128 channels
    - 'layer3': stride 16, 256 channels
    - 'layer4': stride 32, 512 channels, most semantic (default)
    """
    model = models.resnet18(pretrained=True)

    # Map layer names to indices
    layer_map = {
        'layer1': 5,  # up to layer1
        'layer2': 6,  # up to layer2
        'layer3': 7,  # up to layer3
        'layer4': 8,  # up to layer4 (before avgpool)
    }

    children = list(model.children())
    backbone = torch.nn.Sequential(*children[:layer_map[layer]])
    backbone.eval()

    # Get output channels
    channels_map = {
        'layer1': 64,
        'layer2': 128,
        'layer3': 256,
        'layer4': 512
    }

    return backbone, channels_map[layer]

@st.cache_data
def extract_dense_resnet_features(_model, img_array, target_size=None, upsample_mode='bilinear'):
    """
    Extract dense ResNet18 feature map from image

    Args:
        _model: ResNet backbone
        img_array: Input image as numpy array
        target_size: If provided, upsample feature map to this size (H, W)
        upsample_mode: 'bilinear' or 'nearest'

    Returns:
        feature_map: (H', W', C) where H', W' depend on upsampling
    """
    import torch.nn.functional as F

    # Prepare image
    img_pil = Image.fromarray(img_array)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        img_tensor = transform(img_pil).unsqueeze(0)  # (1, 3, H, W)
        feature_map = _model(img_tensor)  # (1, C, H', W')

        # Upsample if requested
        if target_size is not None:
            feature_map = F.interpolate(
                feature_map,
                size=target_size,
                mode=upsample_mode,
                align_corners=False if upsample_mode == 'bilinear' else None
            )

        # Convert to numpy and rearrange
        feature_map = feature_map.squeeze(0)  # (C, H', W')
        feature_map = feature_map.permute(1, 2, 0)  # (H', W', C)
        feature_map = feature_map.numpy()

    return feature_map

def get_pixel_features(img_array, feature_type='rgb'):
    """
    Extract features for each pixel

    feature_type:
    - 'rgb': RGB color values (3D)
    - 'xy': Normalized X,Y positions (2D)
    - 'rgbxy': Combined RGB + XY (5D)
    """
    h, w = img_array.shape[:2]

    if feature_type == 'rgb':
        # Just RGB values
        features = img_array.reshape(-1, 3) / 255.0

    elif feature_type == 'xy':
        # Just positions
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xx = xx / w  # Normalize to [0, 1]
        yy = yy / h
        features = np.stack([xx.flatten(), yy.flatten()], axis=1)

    elif feature_type == 'rgbxy':
        # Combined
        rgb = img_array.reshape(-1, 3) / 255.0
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xx = xx / w
        yy = yy / h
        xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
        features = np.concatenate([rgb, xy], axis=1)

    return features

def cluster_and_visualize(img_array, features, n_clusters, title, alpha=0.5):
    """Cluster pixels and create visualization"""
    h, w = img_array.shape[:2]

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    labels = labels.reshape(h, w)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=14, weight='bold')
    axes[0].axis('off')

    # Cluster labels
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_img = colors[labels][:, :, :3]  # Remove alpha
    axes[1].imshow(cluster_img)
    axes[1].set_title(f"Clusters (K={n_clusters})", fontsize=14, weight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(cluster_img, alpha=alpha)
    axes[2].set_title(f"Overlay ({int(alpha*100)}% transparent)", fontsize=14, weight='bold')
    axes[2].axis('off')

    # Add legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}')
                     for i in range(n_clusters)]
    axes[2].legend(handles=legend_patches, loc='upper right', fontsize=10)

    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()

    return fig, labels

def get_cluster_stats(labels, img_array, n_clusters):
    """Get statistics for each cluster"""
    stats = []
    for i in range(n_clusters):
        mask = labels == i
        pixel_count = mask.sum()
        percentage = 100 * pixel_count / labels.size

        # Average color in this cluster
        cluster_pixels = img_array.reshape(-1, 3)[labels.flatten() == i]
        avg_color = cluster_pixels.mean(axis=0) / 255.0

        stats.append({
            'cluster': i,
            'pixels': pixel_count,
            'percentage': percentage,
            'avg_color': avg_color
        })

    return stats

def compute_elbow_curve(features, max_k=10):
    """
    Compute elbow curve (inertia vs K) for K-Means clustering

    Returns:
        k_values: list of K values tested
        inertias: list of within-cluster sum of squares (WCSS) for each K
    """
    max_k = min(max_k, len(features) - 1)
    k_values = range(2, max_k + 1)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    return list(k_values), inertias

# Sidebar: Image selection
st.sidebar.header("📁 Image Selection")

sample_dir = Path(__file__).parent / "sample_images"
if sample_dir.exists():
    sample_files = sorted(list(sample_dir.glob("*.jpg")))

    if sample_files:
        selected_file = st.sidebar.selectbox(
            "Choose an image",
            sample_files,
            format_func=lambda x: x.name
        )

        # Load image
        img = Image.open(selected_file).convert('RGB')

        # Resize for faster processing
        max_size = st.sidebar.slider("Max image dimension (px)", 200, 800, 400, 50,
                                     help="Smaller = faster")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        st.sidebar.image(img, caption=selected_file.name, use_container_width=True)
        st.sidebar.success(f"Image size: {img_array.shape[1]}×{img_array.shape[0]}")
    else:
        st.error("No sample images found! Run download_samples.py first.")
        st.stop()
else:
    st.error("Sample images directory not found! Run download_samples.py first.")
    st.stop()

# Number of clusters
st.sidebar.markdown("---")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4,
                               help="How many groups to create")

# Opacity slider (onion skin!)
overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.6, 0.05,
                                   help="Adjust cluster overlay transparency (onion skin effect)")

# ============ ELBOW CURVE SECTION ============
st.markdown("---")
st.subheader("📈 Elbow Method: Finding Optimal Number of Clusters")

with st.expander("🔍 Show Elbow Curve (RGB Features)", expanded=False):
    st.markdown("""
    **The elbow method helps determine the ideal number of clusters (K).**

    - **X-axis**: Number of clusters (K)
    - **Y-axis**: Within-cluster sum of squares (WCSS / Inertia)
    - **Elbow point**: Where the curve bends sharply - often the optimal K

    Note: This example uses RGB features. Different features (position, combined, ResNet) may have different optimal K values.
    """)

    max_k_elbow = st.slider("Max K to test", 3, 15, 10,
                           help="Test K-Means with K from 2 to this value", key="elbow_max_k")

    with st.spinner("Computing elbow curve..."):
        features_rgb_sample = get_pixel_features(img_array, 'rgb')
        # Sample pixels for faster computation (use max 10k pixels)
        if len(features_rgb_sample) > 10000:
            sample_indices = np.random.choice(len(features_rgb_sample), 10000, replace=False)
            features_rgb_sample = features_rgb_sample[sample_indices]

        k_values, inertias = compute_elbow_curve(features_rgb_sample, max_k=max_k_elbow)

    # Plot elbow curve
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
    ax_elbow.plot(k_values, inertias, marker='o', markersize=8, linewidth=2, color='royalblue')
    ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=12, weight='bold')
    ax_elbow.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12, weight='bold')
    ax_elbow.set_title("Elbow Curve for K-Means Clustering (RGB Features)", fontsize=14, weight='bold')
    ax_elbow.grid(True, alpha=0.3)
    ax_elbow.set_xticks(k_values)

    # Highlight the "elbow" point using derivative
    if len(inertias) > 2:
        # Compute second derivative to find elbow
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff and 0-indexing
        elbow_k = k_values[elbow_idx]

        # Mark elbow point
        ax_elbow.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Suggested elbow: K={elbow_k}')
        ax_elbow.scatter([elbow_k], [inertias[elbow_idx]], color='red', s=200, zorder=5,
                       marker='*', edgecolors='black', linewidths=2)
        ax_elbow.legend(fontsize=11)

        st.info(f"💡 **Suggested K = {elbow_k}** (based on maximum curvature)")

    plt.tight_layout()
    st.pyplot(fig_elbow)
    plt.close()

    st.markdown("""
    **How to read this**:
    - **Steep drop**: Adding more clusters significantly reduces WCSS
    - **Flattening curve**: Adding more clusters doesn't help much
    - **Elbow**: The point where diminishing returns start - often the best K
    - **Note**: This is a heuristic, not a strict rule. Consider domain knowledge too!
    """)

st.markdown("---")

# Main content - 4 tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Method 1: RGB Colors",
    "📍 Method 2: Position (X,Y)",
    "🔗 Method 3: RGB + Position",
    "🧠 Method 4: ResNet18 Features"
])

# ============ TAB 1: RGB CLUSTERING ============
with tab1:
    st.header("Method 1: Clustering by Color (RGB)")
    st.markdown("""
    **Simplest approach**: Cluster pixels based only on their RGB color values.

    - **Feature**: 3D vector `[R, G, B]` for each pixel
    - **Pro**: Simple, fast, intuitive
    - **Con**: Ignores spatial information - similar colors anywhere get same cluster
    """)

    with st.spinner("Clustering by RGB..."):
        features_rgb = get_pixel_features(img_array, 'rgb')
        fig_rgb, labels_rgb = cluster_and_visualize(
            img_array, features_rgb, n_clusters,
            "Clustering by RGB Color Only",
            alpha=overlay_opacity
        )

    st.pyplot(fig_rgb)
    plt.close()

    # Stats
    st.subheader("📊 Cluster Statistics")
    stats_rgb = get_cluster_stats(labels_rgb, img_array, n_clusters)

    cols = st.columns(min(5, n_clusters))
    for i, stat in enumerate(stats_rgb):
        with cols[i % len(cols)]:
            # Show color swatch
            fig_swatch, ax = plt.subplots(figsize=(2, 1))
            ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                      facecolor=stat['avg_color']))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            st.pyplot(fig_swatch)
            plt.close()

            st.metric(f"Cluster {i}",
                     f"{stat['percentage']:.1f}%",
                     f"{stat['pixels']:,} px")

    with st.expander("💡 What's happening?"):
        st.markdown("""
        **RGB clustering groups pixels by color similarity.**

        Example: In a beach photo:
        - All blue pixels (sky, ocean) → same cluster
        - All yellow/beige pixels (sand) → same cluster
        - All green pixels (trees) → same cluster

        **Problem**: Blue sky and blue ocean get same cluster even though they're in different locations!
        """)

# ============ TAB 2: POSITION CLUSTERING ============
with tab2:
    st.header("Method 2: Clustering by Position (X, Y)")
    st.markdown("""
    **Spatial approach**: Cluster pixels based only on their location in the image.

    - **Feature**: 2D vector `[X, Y]` for each pixel (normalized to 0-1)
    - **Pro**: Creates spatially coherent regions
    - **Con**: Ignores color - might group sky and ocean together just because they're nearby
    """)

    with st.spinner("Clustering by position..."):
        features_xy = get_pixel_features(img_array, 'xy')
        fig_xy, labels_xy = cluster_and_visualize(
            img_array, features_xy, n_clusters,
            "Clustering by Position (X, Y) Only",
            alpha=overlay_opacity
        )

    st.pyplot(fig_xy)
    plt.close()

    # Stats
    st.subheader("📊 Cluster Statistics")
    stats_xy = get_cluster_stats(labels_xy, img_array, n_clusters)

    cols = st.columns(min(5, n_clusters))
    for i, stat in enumerate(stats_xy):
        with cols[i % len(cols)]:
            fig_swatch, ax = plt.subplots(figsize=(2, 1))
            ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                      facecolor=stat['avg_color']))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            st.pyplot(fig_swatch)
            plt.close()

            st.metric(f"Cluster {i}",
                     f"{stat['percentage']:.1f}%",
                     f"{stat['pixels']:,} px")

    with st.expander("💡 What's happening?"):
        st.markdown("""
        **Position clustering divides the image into spatial regions.**

        Think of it like a grid:
        - Top-left region → Cluster 0
        - Top-right region → Cluster 1
        - Bottom-left region → Cluster 2
        - etc.

        **Problem**: Might put sky and trees together if they're both on the left side!

        **When useful**: If you want to analyze different parts of the image separately (top vs bottom, left vs right).
        """)

# ============ TAB 3: COMBINED CLUSTERING ============
with tab3:
    st.header("Method 3: Clustering by RGB + Position")
    st.markdown("""
    **Best of both worlds**: Combine color AND position information.

    - **Feature**: 5D vector `[R, G, B, X, Y]` for each pixel
    - **Pro**: Spatially coherent regions with similar colors
    - **Con**: Need to balance importance of color vs position
    """)

    # Weight sliders
    col1, col2 = st.columns(2)
    with col1:
        rgb_weight = st.slider("RGB weight", 0.0, 2.0, 1.0, 0.1,
                              help="Higher = color matters more")
    with col2:
        xy_weight = st.slider("Position weight", 0.0, 2.0, 0.5, 0.1,
                             help="Higher = position matters more")

    with st.spinner("Clustering by RGB + Position..."):
        features_rgbxy = get_pixel_features(img_array, 'rgbxy')
        # Apply weights
        features_rgbxy[:, :3] *= rgb_weight
        features_rgbxy[:, 3:] *= xy_weight

        fig_rgbxy, labels_rgbxy = cluster_and_visualize(
            img_array, features_rgbxy, n_clusters,
            f"RGB + Position (RGB={rgb_weight:.1f}, XY={xy_weight:.1f})",
            alpha=overlay_opacity
        )

    st.pyplot(fig_rgbxy)
    plt.close()

    # Stats
    st.subheader("📊 Cluster Statistics")
    stats_rgbxy = get_cluster_stats(labels_rgbxy, img_array, n_clusters)

    cols = st.columns(min(5, n_clusters))
    for i, stat in enumerate(stats_rgbxy):
        with cols[i % len(cols)]:
            fig_swatch, ax = plt.subplots(figsize=(2, 1))
            ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                      facecolor=stat['avg_color']))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            st.pyplot(fig_swatch)
            plt.close()

            st.metric(f"Cluster {i}",
                     f"{stat['percentage']:.1f}%",
                     f"{stat['pixels']:,} px")

    with st.expander("💡 What's happening?"):
        st.markdown("""
        **Combined clustering considers both color AND location.**

        **Experiment with weights**:
        - **High RGB weight, low position**: Acts like Tab 1 (color-dominant)
        - **Low RGB weight, high position**: Acts like Tab 2 (position-dominant)
        - **Balanced weights**: Groups regions that are both similar in color AND nearby

        Example in a beach photo with balanced weights:
        - Blue pixels in top half → sky cluster
        - Blue pixels in middle → ocean cluster (different from sky!)
        - Yellow pixels in bottom → sand cluster

        **This is better than pure RGB or pure position!**
        """)

# ============ TAB 4: RESNET18 FEATURES ============
with tab4:
    st.header("Method 4: Deep Learning Features (ResNet18)")
    st.markdown("""
    **Semantic understanding**: Use a pretrained neural network to extract meaningful features **at each location**.

    - **Model**: ResNet18 (18-layer CNN, pretrained on ImageNet)
    - **Dense feature map**: Gets 512-dim features for each spatial location!
    - **Much simpler than DINOv3!** Only 512-dim features vs 768-dim
    - **Pro**: Understands semantic content (recognizes objects, textures, patterns)
    - **Con**: Spatial resolution is reduced (32x downsampling)
    """)

    st.info("💡 Key idea: Extract a **dense feature map** (features at each location) then **upsample** for finer resolution!")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        resnet_layer = st.selectbox(
            "ResNet Layer",
            ['layer1', 'layer2', 'layer3', 'layer4'],
            index=2,  # layer3 default
            help="Earlier layers = higher resolution but less semantic"
        )

    with col2:
        resolution_mode = st.selectbox(
            "Feature Resolution",
            ['Native (fast)', 'Upsampled to image size (better)'],
            index=1,
            help="Upsample features for finer segmentation"
        )

    with col3:
        upsample_mode = st.selectbox(
            "Upsampling Mode",
            ['bilinear', 'nearest'],
            index=0,
            help="Bilinear = smooth, Nearest = blocky"
        ) if 'Upsampled' in resolution_mode else 'nearest'

    # Layer info
    layer_info = {
        'layer1': {'stride': 4, 'channels': 64, 'desc': 'High resolution, less semantic'},
        'layer2': {'stride': 8, 'channels': 128, 'desc': 'Good balance'},
        'layer3': {'stride': 16, 'channels': 256, 'desc': 'Semantic, reasonable resolution'},
        'layer4': {'stride': 32, 'channels': 512, 'desc': 'Most semantic, low resolution'}
    }

    info = layer_info[resnet_layer]
    st.info(f"📊 **{resnet_layer}**: Stride {info['stride']}× (≈ {info['stride']}×{info['stride']}px per patch), {info['channels']} channels - {info['desc']}")

    # Extract features
    with st.spinner(f"Extracting dense ResNet18 features from {resnet_layer}..."):
        model, n_channels = load_resnet18_backbone(resnet_layer)

        # Determine target size for upsampling
        if 'Upsampled' in resolution_mode:
            target_size = (img_array.shape[0], img_array.shape[1])
        else:
            target_size = None

        feature_map = extract_dense_resnet_features(
            model, img_array,
            target_size=target_size,
            upsample_mode=upsample_mode
        )

    h_feat, w_feat, n_channels = feature_map.shape
    st.success(f"✓ Extracted dense feature map: {h_feat}×{w_feat} locations with {n_channels}-dim features each")

    # Cluster the features
    with st.spinner("Clustering feature locations..."):
        # Reshape to (H'*W', C) for clustering
        features_flat = feature_map.reshape(-1, n_channels)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_flat = kmeans.fit_predict(features_flat)

        # Reshape back to (H', W')
        labels_resnet = labels_flat.reshape(h_feat, w_feat)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=14, weight='bold')
    axes[0].axis('off')

    # Cluster labels
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_img = colors[labels_resnet][:, :, :3]  # Remove alpha
    axes[1].imshow(cluster_img)
    axes[1].set_title(f"Clusters (K={n_clusters})", fontsize=14, weight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(cluster_img, alpha=overlay_opacity)
    axes[2].set_title(f"Overlay ({int(overlay_opacity*100)}% transparent)", fontsize=14, weight='bold')
    axes[2].axis('off')

    # Add legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}')
                     for i in range(n_clusters)]
    axes[2].legend(handles=legend_patches, loc='upper right', fontsize=10)

    fig.suptitle(f"ResNet18 {resnet_layer} ({upsample_mode} upsample)", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    # Statistics
    st.subheader("📊 Cluster Statistics")
    stats_resnet = get_cluster_stats(labels_resnet, img_array, n_clusters)

    cols = st.columns(min(5, n_clusters))
    for i, stat in enumerate(stats_resnet):
        with cols[i % len(cols)]:
            # Show color swatch
            fig_swatch, ax = plt.subplots(figsize=(2, 1))
            ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                      facecolor=stat['avg_color']))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            st.pyplot(fig_swatch)
            plt.close()

            st.metric(f"Cluster {i}",
                     f"{stat['percentage']:.1f}%",
                     f"{stat['pixels']:,} px")

    with st.expander("💡 What's happening?"):
        st.markdown(f"""
        **ResNet18 dense features + bilinear upsampling = high-quality segmentation!**

        **How it works**:
        1. Pass entire image through ResNet18 up to **{resnet_layer}**
        2. Get dense feature map (NOT a single vector!)
        3. **Bilinear upsample** features to full image size → {h_feat}×{w_feat} grid
        4. Cluster the {h_feat*w_feat:,} feature vectors
        5. Result: Pixel-level segmentation!

        **Key advantages**:
        - ✅ **Single forward pass** (not hundreds of patches!)
        - ✅ **Bilinear upsampling** → smooth, high-resolution segmentation
        - ✅ **Layer selection** → trade resolution vs semantics
        - ✅ **Semantic understanding** → recognizes textures, objects, patterns

        **Layer comparison**:
        - **layer1** (stride 4): {img_array.shape[0]//4}×{img_array.shape[1]//4} → High res, less semantic (edges, textures)
        - **layer2** (stride 8): {img_array.shape[0]//8}×{img_array.shape[1]//8} → Good balance
        - **layer3** (stride 16): {img_array.shape[0]//16}×{img_array.shape[1]//16} → **Recommended!** Semantic + reasonable resolution
        - **layer4** (stride 32): {img_array.shape[0]//32}×{img_array.shape[1]//32} → Most semantic, lowest resolution

        **Upsampling modes**:
        - **Bilinear**: Smooth interpolation, better boundaries
        - **Nearest**: Blocky but faster

        **Example in a beach photo**:
        - Sky regions → similar features → same cluster (even different shades!)
        - Water regions → similar features → same cluster (different from sky!)
        - Sand regions → different texture → different cluster
        - Objects/people → distinct patterns → separate clusters

        **This is the standard approach in semantic segmentation!**
        Similar to FCN, U-Net, DeepLab - extract dense features then upsample.

        **ResNet18 vs RGB+Position**:
        - RGB+Position: Hand-crafted, fast, but struggles with similar colors
        - ResNet18: Learned features, understands SEMANTICS not just colors!
        """)

    # Comparison with different layers
    with st.expander("📊 Try Different Layers!"):
        st.markdown("""
        **Experiment guide**:

        1. **For fine details** (edges, small objects):
           - Use **layer1** or **layer2**
           - Higher resolution, captures fine structure
           - Less semantic (might group visually similar but semantically different regions)

        2. **For semantic regions** (sky, water, objects):
           - Use **layer3** (recommended!) or **layer4**
           - More semantic understanding
           - Smoother boundaries

        3. **Upsampling**:
           - **Bilinear** almost always better
           - Creates smooth transitions between clusters
           - Looks more natural

        **Try it**: Switch between layer1 and layer3 on the same image - see the difference!
        """)

    # Show feature map visualization
    with st.expander("🔬 Visualize Feature Map (Advanced)"):
        st.markdown(f"**Feature map shape**: {h_feat}×{w_feat}×{n_channels}")

        # Show first 3 feature channels as RGB
        if n_channels >= 3:
            feat_viz = feature_map[:, :, :3]
            # Normalize to [0, 1]
            feat_viz = (feat_viz - feat_viz.min()) / (feat_viz.max() - feat_viz.min() + 1e-8)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(feat_viz)
            ax.set_title("First 3 Feature Channels (as RGB)", fontsize=14, weight='bold')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            st.caption(f"This shows the first 3 (out of {n_channels}) feature channels. Each channel captures different patterns!")

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Summary: Clustering Approaches

| Method | Features | Dimensions | Speed | Resolution | Best For |
|--------|----------|------------|-------|-----------|----------|
| **RGB** | Color values | 3D (R,G,B) | ⚡⚡⚡ | Pixel-level | Distinct colors |
| **Position** | Coordinates | 2D (X,Y) | ⚡⚡⚡ | Pixel-level | Spatial regions |
| **RGB + Position** | Combined | 5D | ⚡⚡⚡ | Pixel-level | Balanced, general use |
| **ResNet18 layer1** | Dense CNN | 64D per 4×4px | ⚡⚡ | High (4× down) | Fine details |
| **ResNet18 layer3** | Dense CNN | 256D per 16×16px | ⚡⚡ | Good (16× down) | **Semantic (recommended!)** |
| **ResNet18 layer4** | Dense CNN | 512D per 32×32px | ⚡ | Lower (32× down) | Most semantic |

**Key insights**:
- ✅ **Dense features + bilinear upsampling** = high-quality segmentation!
- ✅ **layer3 recommended** - best balance of semantics and resolution
- ✅ **Much better than coarse patches** - single forward pass, smooth results
- ✅ **Standard approach** used in FCN, U-Net, DeepLab

**Try different layers and images to see the trade-offs!** 🎉
""")
