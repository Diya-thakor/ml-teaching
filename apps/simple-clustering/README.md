# Simple Image Clustering Demo

Educational app teaching image clustering with **progressive complexity**: from simple RGB/position clustering to deep learning features.

## 🎯 What You'll Learn

This app demonstrates 4 different ways to cluster pixels in an image:

1. **RGB Colors** - Cluster by color similarity only
2. **Position (X,Y)** - Cluster by spatial location only
3. **RGB + Position** - Combine color and location (adjustable weights!)
4. **ResNet18 Features** - Use deep learning for semantic understanding

Each method teaches important concepts while being visual and interactive!

## 🚀 Quick Start

### 1. Download sample images
```bash
python download_samples.py
```

Downloads 7-8 carefully selected images with clear color regions:
- Sunset beach
- Colorful houses
- Autumn forest
- Tulip field
- Mountain lake
- City skyline
- etc.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## 📚 Educational Value

### Tab 1: RGB Clustering 🎨
**Concept**: Cluster pixels by color similarity

**What students learn**:
- K-Means clustering basics
- Feature vectors (3D: R, G, B)
- Limitation: ignores spatial information

**Example**: In a beach photo, all blue pixels (sky + ocean) get same cluster

**Best for**: Understanding color-based segmentation

### Tab 2: Position Clustering 📍
**Concept**: Cluster pixels by location

**What students learn**:
- Spatial features (2D: X, Y normalized)
- Spatial coherence
- Limitation: ignores color information

**Example**: Divides image into grid-like regions (top, bottom, left, right)

**Best for**: Understanding spatial segmentation

### Tab 3: Combined (RGB + Position) 🔗
**Concept**: Balance color and position

**What students learn**:
- Multi-dimensional features (5D: R, G, B, X, Y)
- Feature weighting and importance
- Trade-offs between different features

**Interactive**:
- Adjust RGB weight slider → see color become more/less important
- Adjust position weight → see spatial coherence change
- Experiment to find best balance!

**Example**: Blue sky and blue ocean now get DIFFERENT clusters (same color, different position)

**Best for**: Understanding feature engineering and balancing multiple cues

### Tab 4: ResNet18 Features 🧠
**Concept**: Use pretrained CNN for semantic features

**What students learn**:
- Transfer learning (using pretrained models)
- Deep features vs hand-crafted features
- Patch-based processing
- ResNet18 architecture (simpler than DINOv3!)

**Two modes**:
1. **Whole image**: Extract single 512-dim vector (shows why this doesn't work for clustering)
2. **Patch-based**: Extract features from patches, then cluster (works great!)

**Example**: ResNet recognizes "sky texture", "water texture", "sand texture" semantically

**Best for**: Introduction to deep learning for vision, simpler than DINOv3

## 🎓 Teaching Flow

### Beginner Path
1. Start with **Tab 1 (RGB)** - most intuitive
2. Show limitation with multi-colored objects
3. Try **Tab 2 (Position)** - see spatial segmentation
4. Show limitation (groups unrelated things just because they're nearby)
5. **Tab 3 (Combined)** - shows how to fix both limitations!

### Advanced Path
1. After showing Tabs 1-3, ask: "Can we do better?"
2. **Tab 4 (ResNet18)** - introduce deep learning
3. Compare patch-based ResNet18 to RGB+Position
4. Show how CNN learns semantic features automatically
5. Discuss: simpler than DINOv3, but still powerful!

## 🎯 Suggested Experiments

### Experiment 1: Color Segmentation
**Image**: Tulip field or colorful houses
**Method**: RGB clustering
**Expected**: Each color (red, yellow, blue) forms a cluster
**Lesson**: RGB works great when colors are distinct!

### Experiment 2: Spatial Regions
**Image**: Mountain lake or sunset beach
**Method**: Position clustering
**Expected**: Horizontal bands (sky, mountains, water, shore)
**Lesson**: Position creates coherent regions!

### Experiment 3: The Sky-Ocean Problem
**Image**: Sunset beach
**Method**: RGB → see sky and ocean merge (both blue)
**Method**: RGB+Position (balanced weights) → see sky and ocean separate!
**Lesson**: Combined features solve the problem!

### Experiment 4: Semantic Understanding
**Image**: Any complex scene
**Method**: RGB+Position → struggles with semantic boundaries
**Method**: ResNet18 patches → groups semantic regions
**Lesson**: Deep features understand meaning, not just color/position!

### Experiment 5: Weight Sensitivity
**Image**: City skyline
**Tab 3**: Adjust RGB weight from 0.1 to 2.0
**Observe**: How clusters change from position-dominant to color-dominant
**Lesson**: Feature engineering requires tuning!

## 🔧 Technical Details

### Clustering Algorithm
- **K-Means** used throughout (for consistency)
- `n_init=10` for reproducibility
- `random_state=42` for deterministic results

### Feature Dimensions
| Method | Dimensions | Description |
|--------|------------|-------------|
| RGB | 3D | `[R/255, G/255, B/255]` normalized to [0,1] |
| Position | 2D | `[X/width, Y/height]` normalized to [0,1] |
| RGB+Position | 5D | Concatenation, then weighted |
| ResNet18 | 512D | Output of avgpool layer |

### ResNet18 Details
- Pretrained on ImageNet (1000 classes)
- Remove final FC layer → get 512-dim features
- Input: 224×224 images (resized automatically)
- Output: Global average pooled features

**Why ResNet18 and not ResNet50?**
- Lighter: 11M params vs 25M params
- Faster: Processes patches quicker
- Simpler: Easier to understand (18 layers vs 50)
- Still powerful: 512-dim features capture semantics well

**ResNet18 vs DINOv3**:
| | ResNet18 | DINOv3 |
|---|---|---|
| Training | Supervised (ImageNet labels) | Self-supervised (no labels) |
| Features | 512-dim | 768-dim |
| Speed | Fast | Slower |
| Use case | General purpose | State-of-the-art embeddings |

## 💡 Tips

### For Teaching
- Start with colorful images (tulips, houses) for RGB demo
- Use landscape images (beach, mountain) for position demo
- Complex scenes (city, forest) show ResNet18's advantage

### For Performance
- Use "Max image dimension" slider in sidebar
- Smaller images = faster clustering
- Recommended: 400px for real-time interaction

### For Best Results
- **RGB weight**: 1.0 (default) usually good
- **Position weight**: 0.3-0.7 for most images
- **Patch size**: 56px balances detail and speed
- **Stride**: Half of patch size for good coverage

## 📊 Comparison with DINOv3 App

| Feature | Simple Clustering App | DINOv3 App |
|---------|----------------------|------------|
| **Complexity** | Beginner-friendly | Advanced |
| **Methods** | 4 progressive methods | DINOv3 only |
| **Features** | RGB, XY, ResNet18 (512D) | DINOv3 (768D) |
| **Speed** | Fast | Slower |
| **Clustering** | Pixel-level + patches | Between-images + within-image |
| **Best for** | Teaching fundamentals | State-of-the-art demos |

**Use simple-clustering when**:
- Teaching beginners
- Explaining feature engineering
- Comparing different approaches
- Want fast, interactive demos

**Use dinov2-clustering when**:
- Demonstrating SOTA self-supervised learning
- Clustering multiple images
- Advanced students
- Research-oriented

## 🎉 Sample Images

The `download_samples.py` script downloads:

1. **Sunset beach** - Sky, ocean, sand (great for all methods)
2. **Colorful houses** - Distinct colors (perfect for RGB)
3. **Autumn forest** - Mixed colors and textures (challenging)
4. **Tulip field** - Clear color segmentation
5. **Mountain lake** - Clear spatial regions
6. **City skyline** - Architecture patterns
7. **Colorful birds** - Complex textures

All images carefully selected for teaching different clustering concepts!

## 🐛 Troubleshooting

**"No sample images found"**
- Run `python download_samples.py` first

**"ResNet18 is slow"**
- Reduce image size with slider
- Use larger stride (fewer patches)

**"Clusters look bad"**
- Try adjusting number of clusters
- Experiment with RGB/position weights
- Some images work better with certain methods!

## 📚 References

- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., 2015
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
