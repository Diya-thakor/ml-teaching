# Interactive ML Teaching Apps

A collection of educational Streamlit apps demonstrating machine learning concepts through interactive visualizations and progressive learning approaches.

## 🎯 Overview

These apps are designed for teaching ML/AI concepts in a hands-on, interactive way. Each app follows a **progressive complexity** approach - starting with simple, interpretable methods and gradually introducing more sophisticated techniques.

## 📱 Available Apps

### 1. Image Clustering (ResNet18)
**Path**: [`image-clustering/`](./image-clustering/)
**Concepts**: Image embeddings, clustering algorithms, similarity search

**What it teaches**:
- Extract semantic features from images using ResNet18
- Cluster images by visual content (K-Means, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Similarity search with cosine distance
- Elbow method to find optimal K

**Features**:
- 3 tabs: Clustering | Similarity Search | Raw Embeddings
- 512-D ResNet18 embeddings
- 45 sample images included
- Interactive controls and visualizations

**Running**:
```bash
cd image-clustering
streamlit run app.py
```

---

### 2. Simple Image Clustering (Progressive)
**Path**: [`simple-clustering/`](./simple-clustering/)
**Concepts**: Feature engineering, pixel-level clustering, deep features

**What it teaches**:
- Progressive feature complexity: RGB → Position → Combined → ResNet18
- How features affect clustering results
- Dense feature maps vs global embeddings
- Bilinear upsampling for segmentation
- Elbow method for optimal K

**Features**:
- 4 tabs showing progression of sophistication
- Tab 1: RGB colors (3D) - clusters by color
- Tab 2: Position XY (2D) - clusters by location
- Tab 3: RGB+Position (5D) - combined features
- Tab 4: ResNet18 dense features (64-512D) - semantic understanding
- 7 sample images included

**Running**:
```bash
cd simple-clustering
streamlit run app.py
```

---

### 3. Document Clustering (Progressive)
**Path**: [`document-clustering/`](./document-clustering/)
**Concepts**: NLP, text embeddings, semantic understanding

**What it teaches**:
- Progressive NLP techniques: Simple → TF-IDF → BERT
- How representation affects clustering quality
- Semantic similarity vs keyword matching
- Document-level embeddings
- Elbow method for optimal K

**Features**:
- 3 tabs showing progression of sophistication
- Tab 1: Simple features (6D) - word count, sentences, punctuation
- Tab 2: TF-IDF (50-200D) - keyword importance
- Tab 3: BERT embeddings (384D) - semantic meaning
- 19 sample documents across 4 categories

**Running**:
```bash
cd document-clustering
streamlit run app.py
```

---

## 🎓 Educational Philosophy

### Progressive Complexity
All clustering apps follow a **simple → sophisticated** learning path:

1. **Start Simple**: Interpretable features (RGB, word counts)
2. **Add Context**: Combined features (RGB+Position, TF-IDF)
3. **Go Deep**: Neural network features (ResNet18, BERT)

### Key Insight
**Good features matter more than complex algorithms!**
- Simple K-Means works great with good features (ResNet18, BERT)
- Poor features lead to poor clustering, regardless of algorithm
- Deep learning provides semantic understanding

## 📊 Common Features

### Elbow Method
All clustering apps include elbow curve visualization:
- WCSS vs K plot
- Automatic elbow detection (second derivative)
- Visual marker at suggested K

### Clustering Algorithms
- **K-Means**: Fast, assumes spherical clusters
- **DBSCAN**: Density-based (image-clustering only)
- **Hierarchical**: Agglomerative (image-clustering only)

## 📁 Repository Structure

```
apps/
├── image-clustering/              # ResNet18 image clustering
│   ├── app.py
│   ├── requirements.txt
│   └── sample_images/ (45 images)
│
├── simple-clustering/            # Progressive image clustering
│   ├── app.py
│   ├── requirements.txt
│   └── sample_images/ (7 images)
│
├── document-clustering/          # Progressive document clustering
│   ├── app.py
│   ├── requirements.txt
│   └── sample_documents/ (19 files)
│
└── README.md                     # This file
```

## 🚀 Deployment

### Local Development
```bash
cd <app-directory>
pip install -r requirements.txt
streamlit run app.py
```

### Production
**Recommended**: Hugging Face Spaces or Streamlit Cloud

## 🔗 Course Website
[https://nipunbatra.github.io/ml-teaching/](https://nipunbatra.github.io/ml-teaching/)
