"""
DINOv3 Clustering & Similarity Visualization
Educational app demonstrating:
- DINOv3 feature extraction (from timm)
- Clustering with various algorithms
- Similarity search
- Interactive visualizations with PCA/t-SNE/UMAP
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import umap
from pathlib import Path
import io

# Page config
st.set_page_config(page_title="DINOv3 Clustering Demo", layout="wide")

st.title("🔬 DINOv3 Feature Clustering & Similarity")
st.markdown("""
This app demonstrates **self-supervised learning** with Meta's DINOv3 model (via timm).
Upload images to see clustering, similarity search, and embedding visualization.
""")

# Cache model loading
@st.cache_resource
def load_dino_model(model_name='vit_base_patch14_reg4_dinov2.lvd142m'):
    """Load DINOv3 model from timm (DINOv2 with registers)"""
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 for features
    model.eval()
    return model

@st.cache_data
def extract_features(_model, images_list):
    """Extract DINOv3 features from images"""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    # Get the model's data config for proper preprocessing
    config = resolve_data_config({}, model=_model)
    transform = create_transform(**config)

    features = []
    with torch.no_grad():
        for img in images_list:
            img_tensor = transform(img).unsqueeze(0)
            feat = _model(img_tensor)
            features.append(feat.squeeze().numpy())

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
    with st.spinner("🔄 Extracting DINOv3 features..."):
        model = load_dino_model()
        embeddings = extract_features(model, images)

    st.success(f"✓ Extracted embeddings: shape {embeddings.shape} (images × features)")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Clustering Visualization", "🔍 Similarity Search", "📊 Raw Embeddings", "🖼️ Within-Image Clustering"])

    # ============ TAB 1: CLUSTERING ============
    with tab1:
        st.header("Clustering Visualization")

        col1, col2, col3 = st.columns(3)

        with col1:
            reduction_method = st.selectbox(
                "Dimensionality Reduction",
                ["PCA", "t-SNE", "UMAP"],
                help="Method to reduce embeddings to 2D"
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
                eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.1)
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

        # Create interactive plot
        fig = go.Figure()

        # Add points colored by cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"

            # Prepare hover text
            hover_texts = []
            for idx in np.where(mask)[0]:
                # Get top 5 PCA components if using PCA
                if reduction_method == "PCA":
                    pca_full = pca_model.transform([embeddings[idx]])[0][:5]
                    pca_str = ", ".join([f"{v:.3f}" for v in pca_full])
                    hover_text = f"<b>{image_names[idx]}</b><br>Cluster: {cluster_id}<br>Top 5 PCA: [{pca_str}]"
                else:
                    hover_text = f"<b>{image_names[idx]}</b><br>Cluster: {cluster_id}"
                hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode='markers+text',
                name=cluster_name,
                text=[image_names[i] for i in np.where(mask)[0]],
                textposition="top center",
                textfont=dict(size=10),
                hovertext=hover_texts,
                hoverinfo='text',
                marker=dict(
                    size=15,
                    line=dict(width=2, color='white')
                )
            ))

        fig.update_layout(
            title=f"{reduction_method} Projection with {cluster_method} Clustering",
            xaxis_title=f"{reduction_method} Component 1",
            yaxis_title=f"{reduction_method} Component 2",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

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

    # ============ TAB 2: SIMILARITY SEARCH ============
    with tab2:
        st.header("🔍 Similarity Search")
        st.markdown("Select a reference image to find the most similar images based on DINOv2 embeddings.")

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

        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=image_names,
            y=image_names,
            colorscale='Viridis',
            hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="Cosine Similarity Between All Images",
            xaxis_title="Images",
            yaxis_title="Images",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

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
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=embedding_vec,
                    mode='lines',
                    name='Embedding',
                    line=dict(color='royalblue')
                ))
                fig.update_layout(
                    title="Embedding Feature Values",
                    xaxis_title="Feature Index",
                    yaxis_title="Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Histogram":
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=embedding_vec,
                    nbinsx=50,
                    marker=dict(color='lightblue', line=dict(color='black', width=1))
                ))
                fig.update_layout(
                    title="Distribution of Embedding Values",
                    xaxis_title="Value",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

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
                    fig_top = go.Figure()
                    fig_top.add_trace(go.Bar(
                        x=top_values,
                        y=[f"Feat {i}" for i in top_indices],
                        orientation='h',
                        marker=dict(color='green')
                    ))
                    fig_top.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top, use_container_width=True)

                with col_bottom:
                    st.markdown(f"**Bottom {n_show} Features**")
                    fig_bottom = go.Figure()
                    fig_bottom.add_trace(go.Bar(
                        x=bottom_values,
                        y=[f"Feat {i}" for i in bottom_indices],
                        orientation='h',
                        marker=dict(color='red')
                    ))
                    fig_bottom.update_layout(height=400, yaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig_bottom, use_container_width=True)

            else:  # Raw Values
                st.markdown("**Raw Embedding Vector**")
                st.text(np.array2string(embedding_vec, precision=4, separator=', ', max_line_width=100))

                # Download option
                st.download_button(
                    label="📥 Download as NumPy array",
                    data=embedding_vec.tobytes(),
                    file_name=f"{image_names[selected_img_idx]}_embedding.npy",
                    mime="application/octet-stream"
                )

        # PCA components for all images
        st.subheader("🔬 PCA Analysis of All Embeddings")
        n_components = st.slider("Number of PCA components", 2, 10, 5)

        pca_all = PCA(n_components=n_components)
        pca_transformed = pca_all.fit_transform(embeddings)

        col1, col2 = st.columns(2)
        with col1:
            # Explained variance
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=pca_all.explained_variance_ratio_,
                marker=dict(color='lightcoral')
            ))
            fig.update_layout(
                title="Explained Variance Ratio",
                xaxis_title="Principal Component",
                yaxis_title="Variance Ratio",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cumulative variance
            cumsum = np.cumsum(pca_all.explained_variance_ratio_)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=cumsum,
                mode='lines+markers',
                marker=dict(size=10, color='green'),
                line=dict(width=3)
            ))
            fig.update_layout(
                title="Cumulative Explained Variance",
                xaxis_title="Principal Component",
                yaxis_title="Cumulative Variance",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # PCA component table
        st.markdown("**PCA Components for Each Image**")
        import pandas as pd
        pca_df = pd.DataFrame(
            pca_transformed,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=image_names
        )
        st.dataframe(pca_df.style.format("{:.4f}"), use_container_width=True)

    # ============ TAB 4: WITHIN-IMAGE CLUSTERING ============
    with tab4:
        st.header("🖼️ Within-Image Clustering")
        st.markdown("""
        **Cluster regions/patches within a single image!**
        This extracts features from different parts of an image and clusters them to find similar regions.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Select image
            selected_img_idx = st.selectbox(
                "Select Image to Analyze",
                range(len(images)),
                format_func=lambda i: image_names[i],
                key="within_img_select"
            )

            st.image(images[selected_img_idx], caption=image_names[selected_img_idx], use_container_width=True)

            # Parameters
            st.markdown("**Parameters**")
            patch_size = st.slider("Patch Size (pixels)", 32, 128, 56, 8, help="Size of square patches to extract")
            stride = st.slider("Stride (pixels)", 16, 64, 28, 4, help="Distance between patch centers")
            n_clusters_patch = st.slider("Number of Clusters", 2, 10, 4)

        with col2:
            st.subheader("Clustering Patches")

            # Extract patches from the selected image
            selected_img = images[selected_img_idx]
            img_array = np.array(selected_img)

            with st.spinner("🔄 Extracting patch features..."):
                # Extract overlapping patches
                patches = []
                patch_positions = []

                h, w = img_array.shape[:2]
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        patch = img_array[y:y+patch_size, x:x+patch_size]
                        patches.append(Image.fromarray(patch))
                        patch_positions.append((x, y))

                st.info(f"✓ Extracted {len(patches)} patches ({patch_size}×{patch_size} with stride {stride})")

                # Get features for patches
                patch_embeddings = extract_features(model, patches)

                # Cluster patches
                if len(patches) >= n_clusters_patch:
                    clusterer = KMeans(n_clusters=n_clusters_patch, random_state=42)
                    patch_labels = clusterer.fit_predict(patch_embeddings)

                    st.success(f"✓ Clustered {len(patches)} patches into {n_clusters_patch} clusters")

                    # Create overlay visualization
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    from matplotlib.colors import ListedColormap

                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                    # Original image
                    axes[0].imshow(img_array)
                    axes[0].set_title("Original Image", fontsize=16, weight='bold')
                    axes[0].axis('off')

                    # Clustered patches overlay
                    axes[1].imshow(img_array)

                    # Color map for clusters
                    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters_patch))

                    # Draw rectangles for each patch colored by cluster
                    for (x, y), label in zip(patch_positions, patch_labels):
                        rect = mpatches.Rectangle(
                            (x, y), patch_size, patch_size,
                            linewidth=2,
                            edgecolor=colors[label],
                            facecolor=colors[label],
                            alpha=0.4
                        )
                        axes[1].add_patch(rect)

                    axes[1].set_title("Clustered Regions", fontsize=16, weight='bold')
                    axes[1].axis('off')

                    # Create legend
                    legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}')
                                    for i in range(n_clusters_patch)]
                    axes[1].legend(handles=legend_patches, loc='upper right', fontsize=12)

                    st.pyplot(fig)
                    plt.close()

                    # Show cluster statistics
                    st.subheader("📊 Cluster Statistics")
                    cluster_stats_cols = st.columns(n_clusters_patch)

                    for cluster_id in range(n_clusters_patch):
                        with cluster_stats_cols[cluster_id]:
                            cluster_count = np.sum(patch_labels == cluster_id)
                            cluster_pct = 100 * cluster_count / len(patch_labels)
                            st.metric(f"Cluster {cluster_id}", f"{cluster_count} patches", f"{cluster_pct:.1f}%")

                    # Show example patches from each cluster
                    st.subheader("🔍 Example Patches from Each Cluster")

                    for cluster_id in range(n_clusters_patch):
                        with st.expander(f"Cluster {cluster_id} - Show example patches"):
                            cluster_indices = np.where(patch_labels == cluster_id)[0]
                            n_examples = min(10, len(cluster_indices))

                            # Show up to 10 examples
                            example_cols = st.columns(min(5, n_examples))
                            for i, idx in enumerate(cluster_indices[:n_examples]):
                                with example_cols[i % 5]:
                                    st.image(patches[idx], caption=f"Patch {idx}\n@{patch_positions[idx]}", width=100)

                else:
                    st.warning(f"Not enough patches ({len(patches)}) for {n_clusters_patch} clusters. Reduce patch size or stride.")

else:
    st.info("👆 Please upload at least 2 images in the sidebar to begin.")

    st.markdown("""
    ### 📖 What This App Does

    **DINOv3** (DINOv2 with registers) is a self-supervised vision transformer from Meta AI that learns rich visual representations without labels.

    **Features:**
    - 🎨 **Clustering**: Group similar images using K-Means, DBSCAN, or Hierarchical clustering
    - 🔍 **Similarity Search**: Find images similar to a reference based on cosine similarity
    - 📊 **Embedding Visualization**: Explore raw embeddings, PCA components, and distributions
    - 🖼️ **Within-Image Clustering**: Cluster regions/patches within a single image to find similar areas
    - 🖱️ **Interactive Plots**: Hover over points to see image names and embedding snippets

    **Try it with:**
    - Photos from different categories (animals, vehicles, landscapes)
    - Images with subtle variations
    - Similar objects from different angles

    **Learn more:** [DINOv2 Paper](https://arxiv.org/abs/2304.07193) | [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
    """)
