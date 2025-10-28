# DINOv3 Clustering App - Features Summary

## 🎯 What You Get

### Instant Demo with 45 Sample Images
- **No upload needed!** Check "Use sample images" (enabled by default)
- Diverse categories: fruits (lots!), animals, vehicles, landscapes, food, buildings, flowers
- View all in a collapsible 10-column grid

### Three Powerful Tabs

#### Tab 1: Clustering Visualization 🎨
- **3 Reduction Methods**: PCA, t-SNE, UMAP
- **3 Clustering Algorithms**: K-Means, DBSCAN, Hierarchical
- **Interactive Plotly plots**: Hover to see image names and PCA components
- **Visual cluster summary**: Thumbnails grouped by cluster

#### Tab 2: Similarity Search 🔍
- Select any image as reference
- Find top K most similar images with cosine similarity scores
- Pairwise similarity heatmap for all images
- View embedding snippets for each result

#### Tab 3: Raw Embeddings Inspector 📊
- **4 visualization modes**: Line plots, histograms, top/bottom features, raw values
- **Statistics**: Dimensions (768), mean, std dev
- **PCA analysis**: Explained variance, cumulative variance, component table
- **Download**: Export embeddings as NumPy arrays

#### Tab 4: Within-Image Clustering 🖼️
- **Patch extraction**: Extract overlapping patches from any image
- **Adjustable parameters**: Patch size (32-128px), stride (16-64px), clusters (2-10)
- **Visual overlay**: Side-by-side comparison with colored cluster regions
- **Patch examples**: View up to 10 example patches per cluster (expandable)
- **Statistics**: Cluster sizes and percentages

## 🚀 Quick Start

```bash
# Download sample images (45 images from Unsplash)
python create_samples.py

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 and start exploring!

## 📸 Sample Images Breakdown

| Category | Count | Examples |
|----------|-------|----------|
| Fruits | 19 | Apples (3), Bananas (3), Oranges (3), Strawberries (3), Grapes, Pineapples (2), Mangos (2), Watermelons |
| Animals | 8 | Cats (3), Dogs (3), Birds (2) |
| Vehicles | 5 | Cars (3), Bicycles (2) |
| Landscapes | 6 | Mountains (2), Beaches (2), Forests (2) |
| Food | 6 | Pizza (2), Burgers (2), Sushi (2) |
| Buildings | 2 | Architecture |
| Flowers | 3 | Various flowers |
| **Total** | **45** | |

## 🎓 Educational Use Cases

1. **Teach clustering**: Compare K-Means vs DBSCAN vs Hierarchical
   - K-Means: See how fruits cluster together vs animals
   - DBSCAN: Find outliers automatically
   - Hierarchical: Natural tree structure

2. **Explain dimensionality reduction**:
   - PCA: Fast, linear, interpretable (explained variance shown)
   - t-SNE: Preserves local structure, great for visualization
   - UMAP: Balance of local and global structure

3. **Demonstrate embeddings**:
   - 768-dimensional vectors per image
   - Hover tooltips show PCA components
   - Raw embedding inspector with multiple views

4. **Self-supervised learning**:
   - No labels needed!
   - DINOv3 learns semantic similarity automatically
   - Show how similar objects cluster together

5. **Similarity metrics**:
   - Cosine similarity in practice
   - Pairwise similarity heatmap
   - Find similar objects (e.g., all apples, all dogs)

## 🔬 Model Details

- **Model**: `vit_base_patch14_reg4_dinov2.lvd142m`
- **Architecture**: Vision Transformer Base with 4 register tokens
- **Input**: 224×224 images (auto-resized)
- **Output**: 768-dimensional embeddings
- **Training**: Self-supervised on LVD-142M dataset

## 💡 Tips

- **For clear fruit clusters**: Use K-Means with 5-7 clusters
- **For similar object search**: Select an apple, find other apples!
- **For outlier detection**: Use DBSCAN with eps=0.5
- **PCA interpretation**: First 2 components usually capture 30-50% variance
- **t-SNE**: Good for final visualization, but slower
- **UMAP**: Best balance of speed and quality

## 🎉 Try These Experiments

### Between-Images (Tabs 1-3)
1. **Fruit clustering**: See if apples, bananas, oranges form separate clusters
2. **Animal vs vehicles**: Do animals and vehicles cluster separately?
3. **Similar search**: Select a cat, see if other cats are most similar
4. **Outlier detection**: Use DBSCAN to find unique images
5. **PCA analysis**: How many components needed for 90% variance?
6. **Compare algorithms**: Try K-Means, DBSCAN, Hierarchical on same data

### Within-Image (Tab 4)
7. **Food segmentation**: Select a pizza or burger image - see if toppings cluster separately from background
8. **Landscape regions**: Select a beach/mountain image - see sky, land, water cluster separately
9. **Animal features**: Select a cat/dog - see if face, body, background form different clusters
10. **Building patterns**: Select an architecture image - see windows, walls, details cluster
11. **Adjust granularity**: Try different patch sizes (small=fine details, large=broad regions)
12. **Stride experiments**: Small stride=dense overlapping, large stride=sparse coverage
