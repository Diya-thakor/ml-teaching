"""
Image Clustering & Similarity Visualization
Educational app demonstrating:
- ResNet18 feature extraction (simpler than DINOv2)
- Clustering across multiple images
- Similarity search
- Interactive visualizations with PCA/t-SNE/UMAP
"""

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import umap
from pathlib import Path

# Page config
st.set_page_config(page_title="Image Clustering Demo", layout="wide")

st.title("🔬 Image Clustering & Similarity (ResNet18)")
st.markdown("""
Upload multiple images to see clustering, similarity search, and embedding visualization.
Uses **ResNet18** pretrained on ImageNet to extract image-level features.
""")

# Cache model loading
@st.cache_resource
def load_resnet_model():
    """Load pretrained ResNet18 (without final classification layer)"""
    model = models.resnet18(pretrained=True)
    # Remove the final FC layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

@st.cache_data
def extract_features(_model, images_list):
    """Extract ResNet18 features from images"""
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    features = []
    with torch.no_grad():
        for img in images_list:
            img_tensor = transform(img).unsqueeze(0)
            feat = _model(img_tensor)
            feat = feat.squeeze().numpy()  # Remove batch and spatial dims (512-D vector)
            features.append(feat)

    return np.array(features)

def compute_pca(embeddings, n_components=2):
    """Compute PCA reduction"""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_
    return reduced, explained_var, pca

def compute_tsne(embeddings, n_components=2):
    """Compute t-SNE reduction"""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced = tsne.fit_transform(embeddings)
    return reduced

def compute_umap(embeddings, n_components=2):
    """Compute UMAP reduction"""
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(embeddings)-1))
    reduced = reducer.fit_transform(embeddings)
    return reduced

def compute_elbow_curve(embeddings, max_k=10):
    """
    Compute elbow curve (inertia vs K) for K-Means clustering

    Returns:
        k_values: list of K values tested
        inertias: list of within-cluster sum of squares (WCSS) for each K
    """
    max_k = min(max_k, len(embeddings) - 1)
    k_values = range(2, max_k + 1)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    return list(k_values), inertias

# Sidebar: Upload images or use samples
st.sidebar.header("📁 Image Selection")

use_samples = st.sidebar.checkbox("Use sample images (45 images)", value=True)

if use_samples:
    # Load sample images from disk
    sample_dir = Path(__file__).parent / "sample_images"
    if sample_dir.exists():
        sample_files = sorted(list(sample_dir.glob("*.jpg")))
        st.sidebar.success(f"✓ Loaded {len(sample_files)} sample images")

        # Load images
        images = []
        image_names = []
        for filepath in sample_files:
            img = Image.open(filepath).convert('RGB')
            images.append(img)
            image_names.append(filepath.name)

        uploaded_files = sample_files  # For consistent logic below
    else:
        st.sidebar.error("Sample images directory not found. Please upload your own images.")
        use_samples = False
        uploaded_files = None
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 1:
        st.sidebar.success(f"✓ Loaded {len(uploaded_files)} images")

        # Load images
        images = []
        image_names = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
            image_names.append(file.name)

if (use_samples and sample_dir.exists()) or (not use_samples and uploaded_files and len(uploaded_files) > 1):
    # Show image grid
    st.subheader(f"📸 Loaded {len(images)} Images")

    # Display images in grid
    cols_per_row = 10
    num_rows = (len(images) + cols_per_row - 1) // cols_per_row

    with st.expander("👁️ View All Images (Grid)", expanded=False):
        for row_idx in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row_idx * cols_per_row + col_idx
                if img_idx < len(images):
                    with cols[col_idx]:
                        st.image(images[img_idx], caption=image_names[img_idx], use_container_width=True)

    # Extract features
    with st.spinner("🔄 Extracting ResNet18 features..."):
        model = load_resnet_model()
        embeddings = extract_features(model, images)

    st.success(f"✓ Extracted embeddings: shape {embeddings.shape} (images × 512 features)")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎨 Clustering Visualization", "🔍 Similarity Search", "📊 Raw Embeddings"])

    # ============ TAB 1: CLUSTERING ============
    with tab1:
        st.header("Clustering Visualization")
        st.markdown("""
        **Cluster images by their ResNet18 features** - images with similar content will cluster together.
        """)

        # Elbow curve to determine optimal K
        st.subheader("📈 Elbow Method: Finding Optimal Number of Clusters")

        with st.expander("🔍 Show Elbow Curve", expanded=False):
            st.markdown("""
            **The elbow method helps determine the ideal number of clusters (K).**

            - **X-axis**: Number of clusters (K)
            - **Y-axis**: Within-cluster sum of squares (WCSS / Inertia)
            - **Elbow point**: Where the curve bends sharply - often the optimal K
            """)

            max_k_elbow = st.slider("Max K to test", 3, min(15, len(images)), min(10, len(images)),
                                   help="Test K-Means with K from 2 to this value")

            with st.spinner("Computing elbow curve..."):
                k_values, inertias = compute_elbow_curve(embeddings, max_k=max_k_elbow)

            # Plot elbow curve
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
            ax_elbow.plot(k_values, inertias, marker='o', markersize=8, linewidth=2, color='royalblue')
            ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=12, weight='bold')
            ax_elbow.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12, weight='bold')
            ax_elbow.set_title("Elbow Curve for K-Means Clustering", fontsize=14, weight='bold')
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

        col1, col2, col3 = st.columns(3)

        with col1:
            reduction_method = st.selectbox(
                "Dimensionality Reduction",
                ["PCA", "t-SNE", "UMAP"],
                help="Method to reduce 512-D embeddings to 2D"
            )

        with col2:
            cluster_method = st.selectbox(
                "Clustering Algorithm",
                ["K-Means", "DBSCAN", "Hierarchical"],
                help="Algorithm to group similar images"
            )

        with col3:
            if cluster_method == "K-Means":
                n_clusters = st.slider("Number of Clusters", 2, min(10, len(images)), 3)
            elif cluster_method == "DBSCAN":
                eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.0, 0.1)
            else:
                n_clusters = st.slider("Number of Clusters", 2, min(10, len(images)), 3)

        # Dimensionality reduction
        with st.spinner(f"Computing {reduction_method}..."):
            if reduction_method == "PCA":
                coords_2d, explained_var, pca_model = compute_pca(embeddings, 2)
                st.info(f"PCA: Explained variance = {explained_var[0]:.1%} (PC1) + {explained_var[1]:.1%} (PC2) = {sum(explained_var):.1%} total")
            elif reduction_method == "t-SNE":
                coords_2d = compute_tsne(embeddings, 2)
            else:  # UMAP
                coords_2d = compute_umap(embeddings, 2)

        # Clustering
        with st.spinner(f"Running {cluster_method} clustering..."):
            if cluster_method == "K-Means":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(embeddings)
            elif cluster_method == "DBSCAN":
                clusterer = DBSCAN(eps=eps, min_samples=2)
                labels = clusterer.fit_predict(embeddings)
            else:  # Hierarchical
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(embeddings)

        # Create static matplotlib plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

        for i, cluster_id in enumerate(np.unique(labels)):
            mask = labels == cluster_id
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"

            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                      c=[colors[i]], label=cluster_name, s=100, alpha=0.7, edgecolors='black')

            # Add image names as text
            for j, is_in_cluster in enumerate(mask):
                if is_in_cluster:
                    # Shorten name for display
                    short_name = image_names[j][:15]
                    ax.annotate(short_name, (coords_2d[j, 0], coords_2d[j, 1]),
                               fontsize=8, alpha=0.8, ha='center')

        ax.set_title(f"{reduction_method} Projection with {cluster_method} Clustering", fontsize=14, weight='bold')
        ax.set_xlabel(f'{reduction_method} Component 1', fontsize=12)
        ax.set_ylabel(f'{reduction_method} Component 2', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show cluster summary
        st.subheader("📋 Cluster Summary")
        cluster_cols = st.columns(min(5, len(np.unique(labels))))
        for i, cluster_id in enumerate(np.unique(labels)):
            with cluster_cols[i % len(cluster_cols)]:
                cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
                cluster_images = [img for j, img in enumerate(images) if labels[j] == cluster_id]
                cluster_names = [name for j, name in enumerate(image_names) if labels[j] == cluster_id]

                st.markdown(f"**{cluster_name}** ({len(cluster_images)} images)")

                # Show first 3 always
                for img, name in zip(cluster_images[:3], cluster_names[:3]):
                    st.image(img, caption=name, width=150)

                # If more than 3, add expandable section
                if len(cluster_images) > 3:
                    with st.expander(f"➕ Show {len(cluster_images) - 3} more images"):
                        for img, name in zip(cluster_images[3:], cluster_names[3:]):
                            st.image(img, caption=name, width=150)

        with st.expander("💡 What's happening?"):
            st.markdown("""
            **ResNet18 Image-Level Clustering**

            **How it works**:
            1. Each image is passed through pretrained ResNet18 (trained on ImageNet)
            2. We extract features from the second-to-last layer → **512-dimensional vector** per image
            3. These features capture semantic content (objects, textures, patterns)
            4. Clustering algorithm groups images with similar features

            **Why ResNet18?**
            - ✅ **Simpler than DINOv2**: 11M parameters vs 86M+ parameters
            - ✅ **Faster inference**: No attention mechanism overhead
            - ✅ **Well-understood**: Classic CNN architecture
            - ✅ **Strong features**: Trained on 1.2M ImageNet images
            - ✅ **Fixed-size output**: Always 512-D vector per image

            **What clusters together?**
            - Images with similar objects (e.g., all cats, all cars)
            - Images with similar scenes (e.g., all beaches, all forests)
            - Images with similar textures/patterns

            **Dimensionality reduction**:
            - 512-D vectors are hard to visualize → reduce to 2D
            - **PCA**: Linear projection, preserves global structure
            - **t-SNE**: Non-linear, preserves local neighborhoods
            - **UMAP**: Non-linear, balances local and global structure
            """)

    # ============ TAB 2: SIMILARITY SEARCH ============
    with tab2:
        st.header("🔍 Similarity Search")
        st.markdown("Select a reference image to find the most similar images based on ResNet18 embeddings.")

        # Select reference image
        col1, col2 = st.columns([1, 2])

        with col1:
            ref_idx = st.selectbox(
                "Reference Image",
                range(len(images)),
                format_func=lambda i: image_names[i]
            )
            st.image(images[ref_idx], caption=f"Reference: {image_names[ref_idx]}", width=300)

            top_k = st.slider("Show top K similar", 1, min(10, len(images)-1), min(5, len(images)-1))

        with col2:
            # Compute cosine similarities
            ref_embedding = embeddings[ref_idx].reshape(1, -1)
            similarities = cosine_similarity(ref_embedding, embeddings)[0]

            # Get top K (excluding self)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_indices = [i for i in sorted_indices if i != ref_idx][:top_k]

            st.subheader(f"Top {top_k} Most Similar Images")

            # Display in grid
            cols = st.columns(min(3, top_k))
            for i, idx in enumerate(sorted_indices):
                with cols[i % len(cols)]:
                    similarity_score = similarities[idx]
                    st.image(images[idx], caption=f"{image_names[idx]}", width=200)
                    st.metric("Cosine Similarity", f"{similarity_score:.4f}")

                    # Show embedding snippet
                    with st.expander("View embedding (first 10 dims)"):
                        st.code(np.array2string(embeddings[idx][:10], precision=3, separator=', '))

        # Similarity matrix heatmap
        st.subheader("📊 Pairwise Similarity Matrix")
        sim_matrix = cosine_similarity(embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(image_names)))
        ax.set_yticks(range(len(image_names)))
        ax.set_xticklabels(image_names, rotation=90, fontsize=8)
        ax.set_yticklabels(image_names, fontsize=8)
        ax.set_title("Cosine Similarity Between All Images", fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax, label='Similarity')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("💡 What's happening?"):
            st.markdown("""
            **Semantic Similarity with ResNet18**

            **Cosine similarity** measures the angle between two embedding vectors:
            - **1.0** = Identical (same direction)
            - **0.0** = Unrelated (perpendicular)
            - **-1.0** = Opposite (rare in practice)

            **Why it works**:
            - ResNet18 learns to map similar images to nearby points in 512-D space
            - Images with similar content → similar embeddings → high cosine similarity
            - Use cases: Reverse image search, duplicate detection, recommendation systems

            **Example**:
            - Two photos of cats → high similarity (0.85+)
            - Cat and dog → medium similarity (0.6-0.8)
            - Cat and car → low similarity (0.3-0.5)
            """)

    # ============ TAB 3: RAW EMBEDDINGS ============
    with tab3:
        st.header("📊 Raw Embeddings Inspector")

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_img_idx = st.selectbox(
                "Select Image",
                range(len(images)),
                format_func=lambda i: image_names[i],
                key="raw_embed_select"
            )
            st.image(images[selected_img_idx], caption=image_names[selected_img_idx], width=300)

        with col2:
            st.subheader(f"Embedding for: {image_names[selected_img_idx]}")

            # Show stats
            embedding_vec = embeddings[selected_img_idx]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Dimensions", len(embedding_vec))
            with col_b:
                st.metric("Mean", f"{embedding_vec.mean():.4f}")
            with col_c:
                st.metric("Std Dev", f"{embedding_vec.std():.4f}")

            # Visualization options
            viz_option = st.radio(
                "Visualization",
                ["Line Plot", "Histogram", "Top/Bottom Features", "Raw Values"],
                horizontal=True
            )

            if viz_option == "Line Plot":
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(embedding_vec, color='royalblue', linewidth=1)
                ax.set_title("Embedding Feature Values", fontsize=12, weight='bold')
                ax.set_xlabel("Feature Index")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            elif viz_option == "Histogram":
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(embedding_vec, bins=50, color='lightblue', edgecolor='black')
                ax.set_title("Distribution of Embedding Values", fontsize=12, weight='bold')
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
                plt.close()

            elif viz_option == "Top/Bottom Features":
                n_show = st.slider("Number of features to show", 5, 50, 20)

                # Top features
                top_indices = np.argsort(embedding_vec)[-n_show:][::-1]
                top_values = embedding_vec[top_indices]

                # Bottom features
                bottom_indices = np.argsort(embedding_vec)[:n_show]
                bottom_values = embedding_vec[bottom_indices]

                col_top, col_bottom = st.columns(2)
                with col_top:
                    st.markdown(f"**Top {n_show} Features**")
                    fig, ax = plt.subplots(figsize=(5, 6))
                    ax.barh(range(len(top_values)), top_values, color='green')
                    ax.set_yticks(range(len(top_values)))
                    ax.set_yticklabels([f"Feat {i}" for i in top_indices])
                    ax.set_xlabel("Value")
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                    plt.close()

                with col_bottom:
                    st.markdown(f"**Bottom {n_show} Features**")
                    fig, ax = plt.subplots(figsize=(5, 6))
                    ax.barh(range(len(bottom_values)), bottom_values, color='red')
                    ax.set_yticks(range(len(bottom_values)))
                    ax.set_yticklabels([f"Feat {i}" for i in bottom_indices])
                    ax.set_xlabel("Value")
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                    plt.close()

            else:  # Raw Values
                st.markdown("**Raw Embedding Vector (512 dimensions)**")
                st.text(np.array2string(embedding_vec, precision=4, separator=', ', max_line_width=100))

        # PCA components for all images
        st.subheader("🔬 PCA Analysis of All Embeddings")
        n_components = st.slider("Number of PCA components", 2, 10, 5)

        pca_all = PCA(n_components=n_components)
        pca_transformed = pca_all.fit_transform(embeddings)

        col1, col2 = st.columns(2)
        with col1:
            # Explained variance
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(n_components), pca_all.explained_variance_ratio_, color='lightcoral')
            ax.set_xticks(range(n_components))
            ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
            ax.set_title("Explained Variance Ratio", fontsize=12, weight='bold')
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Variance Ratio")
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()

        with col2:
            # Cumulative variance
            cumsum = np.cumsum(pca_all.explained_variance_ratio_)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(n_components), cumsum, marker='o', markersize=8, color='green', linewidth=2)
            ax.set_xticks(range(n_components))
            ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
            ax.set_title("Cumulative Explained Variance", fontsize=12, weight='bold')
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Cumulative Variance")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        # PCA component table
        st.markdown("**PCA Components for Each Image**")
        import pandas as pd
        pca_df = pd.DataFrame(
            pca_transformed,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=image_names
        )
        st.dataframe(pca_df.style.format("{:.4f}"), use_container_width=True)

        with st.expander("💡 What's happening?"):
            st.markdown("""
            **Understanding ResNet18 Embeddings**

            **Raw embeddings** (512-D):
            - Each dimension is a learned feature from ImageNet training
            - Some dimensions activate for specific patterns (edges, textures, objects)
            - No human-interpretable meaning for individual dimensions

            **PCA Analysis**:
            - Finds the principal directions of variation in embedding space
            - PC1 = direction with most variance (e.g., "outdoor vs indoor")
            - PC2 = second most variance (e.g., "animals vs objects")
            - First 5-10 PCs typically capture 70-90% of variance

            **Why 512 dimensions?**
            - ResNet18's last pooling layer outputs 512 channels
            - Each channel is a different "detector" for visual patterns
            - 512-D is enough to represent diverse ImageNet categories (1000 classes)

            **Compared to other models**:
            - ResNet50: 2048-D (more capacity, slower)
            - EfficientNet: varies (512-2048D)
            - Vision Transformers (ViT, DINOv2): 384-1024D (attention-based)
            """)

else:
    st.info("👆 Please upload at least 2 images in the sidebar to begin.")

    st.markdown("""
    ### 📖 What This App Does

    **ResNet18** is a convolutional neural network trained on ImageNet (1.2M images, 1000 categories).
    We use it to extract rich visual features from images for clustering and similarity search.

    **Features:**
    - 🎨 **Clustering**: Group similar images using K-Means, DBSCAN, or Hierarchical clustering
    - 🔍 **Similarity Search**: Find images similar to a reference based on cosine similarity
    - 📊 **Embedding Visualization**: Explore raw embeddings, PCA components, and distributions

    **Try it with:**
    - Photos from different categories (animals, vehicles, landscapes)
    - Images with subtle variations
    - Similar objects from different angles

    **Learn more:** [ResNet Paper](https://arxiv.org/abs/1512.03385)
    """)
