# DINOv3 Clustering & Similarity Visualization

Interactive educational app demonstrating clustering and similarity search using DINOv3 (DINOv2 with registers) embeddings from Meta AI.

## Features

### 🎨 Clustering Visualization
- **Multiple algorithms**: K-Means, DBSCAN, Hierarchical
- **Dimensionality reduction**: PCA, t-SNE, UMAP
- **Interactive plots**: Hover to see image names and PCA components
- **Visual cluster summary**: See grouped images side-by-side

### 🔍 Similarity Search
- **Cosine similarity**: Find images most similar to a reference
- **Pairwise heatmap**: Explore all image-to-image similarities
- **Embedding preview**: View embedding snippets for each result

### 📊 Raw Embeddings Inspector
- **Feature visualization**: Line plots, histograms, top/bottom features
- **PCA analysis**: Explained variance and cumulative variance plots
- **Component table**: See PCA coordinates for each image
- **Download**: Export embeddings as NumPy arrays

### 🖼️ Within-Image Clustering
- **Patch extraction**: Extract overlapping patches from a single image
- **Cluster regions**: Group similar regions/areas within the image
- **Visual overlay**: See clustered regions with colored overlays
- **Patch examples**: View example patches from each cluster
- **Statistics**: See cluster sizes and percentages

## What is DINOv3?

**DINOv3** refers to DINOv2 models enhanced with **vision transformer registers** (see [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)).

**DINOv2** (Distillation with NO labels, version 2) is a self-supervised learning method from Meta AI that:
- Learns visual representations without requiring labeled data
- Uses vision transformers (ViT) trained on massive image collections
- Produces features that work well for many downstream tasks
- Captures semantic similarity between images

The "registers" enhancement improves feature quality by adding learnable tokens that absorb artifacts.

## How to Use

### 1. Download sample images (optional but recommended)
```bash
python create_samples.py
```
This downloads 45 sample images from Unsplash covering:
- Animals (cats, dogs, birds)
- Fruits (apples, bananas, oranges, strawberries, grapes, pineapples, mangos)
- Vehicles (cars, bicycles)
- Landscapes (mountains, beaches, forests)
- Food (pizza, burgers, sushi)
- Buildings
- Flowers

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Select images
**Option A (Default)**: Check "Use sample images" in the sidebar to load 45 pre-downloaded images

**Option B**: Uncheck the box and upload your own images (JPG or PNG)
- Try images from different categories for clear clustering
- Or upload variations of similar objects to explore fine-grained similarities

### 5. View image grid
Click "View All Images (Grid)" to see all loaded images in a 10-column grid

### 6. Explore the tabs

**Tab 1: Clustering**
- Choose dimensionality reduction method (PCA, t-SNE, UMAP)
- Select clustering algorithm and parameters
- Hover over points to see image names and embeddings
- Scroll down to see cluster summaries with image thumbnails

**Tab 2: Similarity Search**
- Select a reference image
- View top K most similar images with cosine similarity scores
- Explore the pairwise similarity heatmap

**Tab 3: Raw Embeddings**
- Select an image to inspect its embedding vector
- View statistics (dimensions, mean, std)
- Visualize with line plots, histograms, or top/bottom features
- See PCA analysis for all images

**Tab 4: Within-Image Clustering**
- Select any image to analyze
- Adjust patch size and stride to control granularity
- Choose number of clusters (2-10)
- View side-by-side: original image and clustered regions overlay
- Explore example patches from each cluster
- See statistics: how many patches per cluster

## Model Details

**Model**: `vit_base_patch14_reg4_dinov2.lvd142m` from timm
- Architecture: Vision Transformer Base
- Patch size: 14×14
- Registers: 4 register tokens
- Training: Self-supervised on LVD-142M dataset
- Feature dimension: 768

## Educational Use Cases

- **Teach clustering algorithms**: Compare K-Means, DBSCAN, and Hierarchical clustering
- **Explain dimensionality reduction**: Show PCA vs t-SNE vs UMAP differences
- **Demonstrate embeddings**: Visualize high-dimensional feature spaces
- **Self-supervised learning**: Show how models learn without labels
- **Similarity metrics**: Explore cosine similarity in practice
- **Within-image analysis**: Show how different parts of an image relate to each other

## References

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) - Oquab et al., 2023
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) - Darcet et al., 2023
- [timm Documentation](https://huggingface.co/docs/timm/index)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

## Tips

- **For clear clusters**: Use images from 3-5 distinct categories
- **For similarity search**: Upload variations of the same object (different angles, lighting, etc.)
- **PCA**: Fast, interpretable, but linear
- **t-SNE**: Good for visualization, preserves local structure
- **UMAP**: Balances local and global structure, faster than t-SNE
- **K-Means**: Need to specify number of clusters
- **DBSCAN**: Automatically finds clusters, can detect outliers (noise)
- **Hierarchical**: Creates a tree of clusters, intuitive for nested categories
