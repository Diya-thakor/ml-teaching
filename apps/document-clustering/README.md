# Document Clustering Demo

**Progressive document clustering: From simple features to BERT embeddings**

An educational Streamlit app that demonstrates different approaches to clustering text documents, from basic statistical features to deep semantic understanding.

## 🎯 Overview

This app teaches document clustering through three progressive methods:

1. **Simple Statistical Features** (6D) - Like RGB for images
   - Word count, sentence count, average word length
   - Punctuation, digits, uppercase letters
   - Clusters by document *style*, not meaning

2. **TF-IDF Vectors** (50-200D) - Keyword-based representation
   - Term Frequency × Inverse Document Frequency
   - Captures important words in each document
   - Clusters by shared *vocabulary*

3. **BERT Embeddings** (384D) - Deep semantic understanding
   - Sentence-BERT pretrained model
   - Understands context, synonyms, paraphrasing
   - Clusters by actual *meaning*

## 📁 Sample Documents

The app includes 19 sample documents across 4 categories:
- **Technology/AI** (5 docs): Machine learning, quantum computing, blockchain, AI, cybersecurity
- **Health/Medicine** (5 docs): Cardiovascular health, nutrition, mental health, immunology, diabetes
- **Environment/Climate** (5 docs): Climate change, renewable energy, ocean conservation, biodiversity, sustainable agriculture
- **Sports/Fitness** (4 docs): Sports training, marathon running, strength conditioning, yoga

## 🚀 Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or specify a port
streamlit run app.py --server.port 8520
```

The app will be available at `http://localhost:8501` (or your specified port).

## 📊 Features

### Interactive Controls
- **Document selection**: Choose subset or use all documents
- **Number of clusters**: Adjust K for K-Means clustering
- **Visualization method**: PCA or t-SNE for 2D projection
- **TF-IDF settings**: Adjust vocabulary size and min document frequency

### Visualizations
- **2D scatter plots**: PCA/t-SNE projection with document labels
- **Cluster membership**: See which documents are in each cluster
- **Feature statistics**: View raw features for each document
- **Top words per cluster**: For TF-IDF, see most important words
- **Semantic similarity**: For BERT, measure cluster cohesion

## 🎓 Educational Value

### Key Learning Outcomes

1. **Simple features cluster by style, not meaning**
   - Long technical documents group together regardless of topic
   - Similar to clustering images by pixel count rather than content

2. **TF-IDF captures topic keywords**
   - Documents sharing vocabulary cluster together
   - Better than simple features, but treats words independently
   - Misses synonyms and context

3. **BERT understands semantic meaning**
   - Clusters documents by actual topic, not just word overlap
   - Handles synonyms ("ML" = "machine learning" = "AI")
   - Understands context (same words, different meanings)

### Analogy to Image Clustering

| Documents | Images |
|-----------|--------|
| Simple features | Position (X, Y) |
| TF-IDF | RGB colors |
| BERT | ResNet18 deep features |

## 🔧 Technical Details

### Models Used
- **TF-IDF**: scikit-learn's `TfidfVectorizer`
- **BERT**: `sentence-transformers` with `all-MiniLM-L6-v2` model
  - 384-dimensional embeddings
  - Lightweight (22M parameters)
  - Fast inference

### Clustering
- **Algorithm**: K-Means clustering
- **Dimensionality reduction**: PCA or t-SNE for visualization
- **Normalization**: StandardScaler for simple features

### Dependencies
```
streamlit          # Web app framework
numpy              # Numerical computing
matplotlib         # Plotting
scikit-learn       # ML algorithms (K-Means, TF-IDF, PCA, t-SNE)
sentence-transformers  # BERT embeddings
pandas             # Data display
```

## 📝 Files

- `app.py` - Main Streamlit application
- `create_samples.py` - Script to generate sample documents
- `sample_documents/` - 19 text documents across 4 topics
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎯 Use Cases

### In the Classroom
- Demonstrate progression from simple to sophisticated NLP
- Compare feature engineering vs deep learning
- Explore trade-offs (speed vs semantic understanding)

### Experimentation
- Try different K values and see how clusters change
- Compare PCA vs t-SNE visualizations
- Adjust TF-IDF parameters (vocabulary size, min DF)
- Add your own documents to cluster

## 💡 Key Insights

1. **Simple features are fast but shallow**
   - 6D vectors, instant computation
   - Clusters by style metrics (length, complexity)
   - Ignores semantic content

2. **TF-IDF balances speed and semantics**
   - 50-200D vectors, very fast
   - Captures topic keywords
   - But treats "good" and "excellent" as unrelated

3. **BERT is slow but deeply semantic**
   - 384D vectors, requires GPU for large corpora
   - Understands meaning, context, synonyms
   - State-of-the-art for document clustering/search

4. **Trade-offs matter**
   - Simple/TF-IDF: Fast, interpretable, limited
   - BERT: Slow, black-box, powerful

## 🚀 Extensions

Ideas for extending this app:

1. **Hierarchical clustering** - Show dendrogram instead of K-Means
2. **Topic modeling** - Add LDA (Latent Dirichlet Allocation)
3. **User document upload** - Let users cluster their own texts
4. **Comparison metrics** - Silhouette score, Davies-Bouldin index
5. **Other embeddings** - Word2Vec, GloVe, GPT embeddings
6. **Named Entity Recognition** - Show entities extracted from each cluster

## 📚 Related Concepts

- **Information Retrieval**: BERT embeddings power semantic search
- **Recommendation Systems**: Cluster users/items by embeddings
- **Topic Modeling**: Discover themes in document collections
- **Text Classification**: Supervised learning from labeled documents

## 🎉 Summary

This app demonstrates the evolution of document clustering from basic statistics to deep learning. Students can see firsthand how BERT's semantic understanding produces superior clusters compared to simple features or TF-IDF, and understand the trade-offs involved.

**Try it with different document sets and see which method works best!**
