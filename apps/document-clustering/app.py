"""
Document Clustering Demo
Progressive complexity: Simple Features → TF-IDF → BERT Embeddings

Educational app showing different document clustering approaches:
1. Simple features (word count, sentence count, etc.)
2. TF-IDF vectors (term frequency - inverse document frequency)
3. BERT embeddings (deep semantic understanding)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import re
from collections import Counter

# Page config
st.set_page_config(page_title="Document Clustering Demo", layout="wide")

st.title("📄 Document Clustering: From Simple to Semantic")
st.markdown("""
**Learn document clustering progressively!** Start with simple text statistics,
then TF-IDF, and finally deep semantic embeddings with BERT.
""")

# ==================== FEATURE EXTRACTION ====================

def get_simple_features(text):
    """
    Extract simple statistical features from text (like RGB for images)

    Returns 6D feature vector:
    - Word count
    - Sentence count
    - Average word length
    - Punctuation count
    - Digit count
    - Uppercase letter count
    """
    # Word count
    words = text.split()
    word_count = len(words)

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Average word length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    # Punctuation count
    punctuation_count = sum(1 for c in text if c in '.,!?;:')

    # Digit count
    digit_count = sum(1 for c in text if c.isdigit())

    # Uppercase letter count
    uppercase_count = sum(1 for c in text if c.isupper())

    return np.array([
        word_count,
        sentence_count,
        avg_word_length,
        punctuation_count,
        digit_count,
        uppercase_count
    ])

def get_keyword_features(texts, top_n=20):
    """
    Extract keyword-based features (word presence/frequency)
    Similar to RGB but for common words

    Returns matrix where each row is a document and each column is a keyword count
    """
    # Get all words from all documents
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())  # Words with 4+ letters
        all_words.extend(words)

    # Get top N most common words (excluding very common stop words)
    common_stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were',
                        'they', 'their', 'will', 'would', 'could', 'about', 'into'}
    word_counts = Counter(all_words)

    # Remove stopwords
    for word in common_stopwords:
        word_counts.pop(word, None)

    # Get top keywords
    top_keywords = [word for word, _ in word_counts.most_common(top_n)]

    # Create feature vectors
    features = []
    for text in texts:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        word_count = Counter(words)
        feature_vec = np.array([word_count.get(kw, 0) for kw in top_keywords])
        features.append(feature_vec)

    return np.array(features), top_keywords

@st.cache_resource
def load_sentence_transformer():
    """Load sentence-transformer model (lightweight BERT-based model)"""
    from sentence_transformers import SentenceTransformer
    # Use a smaller, faster model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
    return model

@st.cache_data
def get_bert_embeddings(_model, texts):
    """Get BERT embeddings for documents"""
    embeddings = _model.encode(texts, show_progress_bar=False)
    return embeddings

# ==================== VISUALIZATION ====================

def visualize_clusters_2d(features, labels, doc_names, n_clusters, title, method='PCA'):
    """
    Visualize high-dimensional clusters in 2D using PCA or t-SNE
    """
    # Reduce to 2D
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:  # t-SNE
        # Use PCA first if features are very high dimensional
        if features.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            features = pca.fit_transform(features)
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))

    coords_2d = reducer.fit_transform(features)

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=[colors[i]], label=f'Cluster {i}', s=100, alpha=0.7, edgecolors='black')

        # Add document names as text
        for j, is_in_cluster in enumerate(mask):
            if is_in_cluster:
                # Shorten name for display
                short_name = doc_names[j].replace('.txt', '').replace('_', ' ')[:20]
                ax.annotate(short_name, (coords_2d[j, 0], coords_2d[j, 1]),
                           fontsize=8, alpha=0.8, ha='center')

    ax.set_title(f"{title} ({method} visualization)", fontsize=14, weight='bold')
    ax.set_xlabel(f'{method} Component 1', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, coords_2d

def show_cluster_members(doc_names, doc_texts, labels, n_clusters):
    """Show which documents are in each cluster"""
    st.subheader("📋 Cluster Membership")

    cols = st.columns(min(n_clusters, 3))

    for i in range(n_clusters):
        with cols[i % len(cols)]:
            st.markdown(f"**Cluster {i}**")

            # Get documents in this cluster
            cluster_docs = [(name, text) for name, text, label in zip(doc_names, doc_texts, labels) if label == i]

            st.markdown(f"*{len(cluster_docs)} documents*")

            for name, text in cluster_docs:
                short_name = name.replace('.txt', '').replace('_', ' ')
                # Show preview and full text in tabs
                preview = text[:150].replace('\n', ' ') + '...'
                with st.expander(f"📄 {short_name}"):
                    tab_preview, tab_full = st.tabs(["Preview", "Full Text"])
                    with tab_preview:
                        st.text(preview)
                    with tab_full:
                        st.text_area("", value=text, height=300, disabled=True, label_visibility="collapsed")

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

# ==================== MAIN APP ====================

# Sidebar: Load documents
st.sidebar.header("📁 Document Selection")

sample_dir = Path(__file__).parent / "sample_documents"
if not sample_dir.exists():
    st.error("Sample documents directory not found! Run create_samples.py first.")
    st.stop()

# Load all documents
doc_files = sorted(list(sample_dir.glob("*.txt")))
if not doc_files:
    st.error("No sample documents found! Run create_samples.py first.")
    st.stop()

# Allow user to select subset or use all
use_all = st.sidebar.checkbox("Use all documents", value=True)

if use_all:
    selected_files = doc_files
else:
    selected_files = st.sidebar.multiselect(
        "Select documents to cluster",
        doc_files,
        default=doc_files[:10],
        format_func=lambda x: x.name
    )

if len(selected_files) < 3:
    st.warning("⚠️ Please select at least 3 documents for meaningful clustering.")
    st.stop()

# Load document texts
doc_names = [f.name for f in selected_files]
doc_texts = [f.read_text() for f in selected_files]

st.sidebar.success(f"✓ Loaded {len(doc_texts)} documents")

# Number of clusters
st.sidebar.markdown("---")
n_clusters = st.sidebar.slider("Number of Clusters", 2, min(10, len(doc_texts)-1),
                               min(4, len(doc_texts)//2),
                               help="How many groups to create")

# Visualization method
viz_method = st.sidebar.selectbox("Dimensionality Reduction",
                                  ['PCA', 't-SNE'],
                                  help="Method to visualize high-dim clusters in 2D")

# ==================== ELBOW CURVE SECTION ====================
st.markdown("---")
st.subheader("📈 Elbow Method: Finding Optimal Number of Clusters")

with st.expander("🔍 Show Elbow Curve (Simple Features)", expanded=False):
    st.markdown("""
    **The elbow method helps determine the ideal number of clusters (K).**

    - **X-axis**: Number of clusters (K)
    - **Y-axis**: Within-cluster sum of squares (WCSS / Inertia)
    - **Elbow point**: Where the curve bends sharply - often the optimal K

    Note: This example uses simple features. Different features (TF-IDF, BERT) may have different optimal K values.
    """)

    max_k_elbow = st.slider("Max K to test", 3, min(15, len(doc_texts)), min(10, len(doc_texts)),
                           help="Test K-Means with K from 2 to this value", key="elbow_max_k")

    with st.spinner("Computing elbow curve..."):
        # Use simple features for elbow curve
        simple_features = np.array([get_simple_features(text) for text in doc_texts])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        simple_features_scaled = scaler.fit_transform(simple_features)

        k_values, inertias = compute_elbow_curve(simple_features_scaled, max_k=max_k_elbow)

    # Plot elbow curve
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
    ax_elbow.plot(k_values, inertias, marker='o', markersize=8, linewidth=2, color='royalblue')
    ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=12, weight='bold')
    ax_elbow.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12, weight='bold')
    ax_elbow.set_title("Elbow Curve for K-Means Clustering (Simple Features)", fontsize=14, weight='bold')
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

# ==================== TABS ====================

tab1, tab2, tab3 = st.tabs([
    "📊 Method 1: Simple Features",
    "🔤 Method 2: TF-IDF Vectors",
    "🧠 Method 3: BERT Embeddings"
])

# ============ TAB 1: SIMPLE FEATURES ============
with tab1:
    st.header("Method 1: Simple Statistical Features")
    st.markdown("""
    **Basic approach**: Extract simple statistical features from each document (like RGB for images).

    **Features** (6 dimensions):
    - Word count
    - Sentence count
    - Average word length
    - Punctuation count
    - Digit count
    - Uppercase letter count

    **Pro**: Simple, fast, interpretable
    **Con**: Ignores semantic meaning - documents about different topics might cluster together if they have similar statistics
    """)

    with st.spinner("Extracting simple features..."):
        # Extract features for each document
        simple_features = np.array([get_simple_features(text) for text in doc_texts])

        # Normalize features (important for K-Means)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        simple_features_scaled = scaler.fit_transform(simple_features)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_simple = kmeans.fit_predict(simple_features_scaled)

    # Visualize
    fig_simple, _ = visualize_clusters_2d(
        simple_features_scaled, labels_simple, doc_names, n_clusters,
        "Simple Features Clustering", method=viz_method
    )
    st.pyplot(fig_simple)
    plt.close()

    # Show cluster members
    show_cluster_members(doc_names, doc_texts, labels_simple, n_clusters)

    # Show feature statistics
    with st.expander("📊 Feature Statistics"):
        st.markdown("**Feature names**: Word count, Sentence count, Avg word length, Punctuation, Digits, Uppercase")

        import pandas as pd
        feature_names = ['Words', 'Sentences', 'Avg Word Len', 'Punctuation', 'Digits', 'Uppercase']
        df_features = pd.DataFrame(simple_features,
                                   columns=feature_names,
                                   index=[n.replace('.txt', '') for n in doc_names])
        st.dataframe(df_features, use_container_width=True)

    with st.expander("💡 What's happening?"):
        st.markdown("""
        **Simple features cluster by document style, not content!**

        Example patterns:
        - Long technical documents → same cluster (high word count)
        - Short simple documents → same cluster (low word count)
        - Documents with lots of numbers → same cluster (high digit count)

        **Problem**: A technical document about AI and another about medicine might cluster together
        just because they're both long and technical - the semantic meaning is ignored!

        **Similar to RGB clustering**: Groups by surface-level statistics, not deeper meaning.
        """)

# ============ TAB 2: TF-IDF ============
with tab2:
    st.header("Method 2: TF-IDF (Term Frequency - Inverse Document Frequency)")
    st.markdown("""
    **Intermediate approach**: Represent each document as a vector of word importance scores.

    **How it works**:
    - **TF (Term Frequency)**: How often a word appears in a document
    - **IDF (Inverse Document Frequency)**: Penalizes common words that appear in many documents
    - **TF-IDF = TF × IDF**: Highlights words that are frequent in a document but rare overall

    **Pro**: Captures word usage patterns, handles different topics
    **Con**: Treats words as independent tokens, ignores word order and context
    """)

    # TF-IDF settings
    col1, col2 = st.columns(2)
    with col1:
        max_features = st.slider("Max features (vocabulary size)", 10, 200, 50, 10,
                                help="Number of most important words to consider")
    with col2:
        min_df = st.slider("Min document frequency", 1, 5, 1, 1,
                          help="Ignore words appearing in fewer than this many docs")

    with st.spinner("Computing TF-IDF vectors..."):
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english',  # Remove common English stop words
            lowercase=True
        )

        # Transform documents to TF-IDF vectors
        tfidf_features = vectorizer.fit_transform(doc_texts).toarray()

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_tfidf = kmeans.fit_predict(tfidf_features)

    st.success(f"✓ Created {tfidf_features.shape[1]}-dimensional TF-IDF vectors")

    # Visualize
    fig_tfidf, _ = visualize_clusters_2d(
        tfidf_features, labels_tfidf, doc_names, n_clusters,
        "TF-IDF Clustering", method=viz_method
    )
    st.pyplot(fig_tfidf)
    plt.close()

    # Show cluster members
    show_cluster_members(doc_names, doc_texts, labels_tfidf, n_clusters)

    # Show top words per cluster
    with st.expander("🔤 Top Words per Cluster"):
        st.markdown("**Most important words (by TF-IDF score) in each cluster:**")

        for i in range(n_clusters):
            st.markdown(f"**Cluster {i}:**")

            # Get documents in this cluster
            cluster_mask = labels_tfidf == i
            cluster_tfidf = tfidf_features[cluster_mask]

            # Average TF-IDF scores for this cluster
            avg_tfidf = cluster_tfidf.mean(axis=0)

            # Get top 10 words
            top_indices = avg_tfidf.argsort()[-10:][::-1]
            top_words = [(feature_names[idx], avg_tfidf[idx]) for idx in top_indices]

            words_str = ", ".join([f"{word} ({score:.3f})" for word, score in top_words])
            st.text(words_str)

    with st.expander("💡 What's happening?"):
        st.markdown("""
        **TF-IDF captures word importance and helps cluster by topic!**

        **How TF-IDF works**:
        1. **Term Frequency**: Count how often each word appears in each document
        2. **Inverse Document Frequency**: Give lower weight to words that appear in many documents
        3. **Multiply**: TF × IDF gives importance score for each word in each document

        **Example**:
        - Word "health" appears often in health documents → high TF
        - Word "health" is rare in other documents → high IDF
        - Result: "health" gets high TF-IDF score in health documents

        **Common words like "the", "and", "is"**:
        - Appear in ALL documents → low IDF → low TF-IDF (filtered out)

        **Result**: Documents cluster by topic because they share distinctive vocabulary!

        **Better than simple features** but still treats words independently (ignores context).
        """)

# ============ TAB 3: BERT EMBEDDINGS ============
with tab3:
    st.header("Method 3: BERT Embeddings (Deep Semantic Understanding)")
    st.markdown("""
    **Advanced approach**: Use a pretrained language model to create semantic embeddings.

    **Model**: Sentence-BERT (all-MiniLM-L6-v2)
    - 384-dimensional dense embeddings
    - Trained on millions of text pairs to understand semantic similarity
    - Captures meaning, context, and relationships between words

    **Pro**: Deep semantic understanding - clusters by actual meaning!
    **Con**: Requires pretrained model, more computationally expensive
    """)

    try:
        with st.spinner("Loading BERT model..."):
            bert_model = load_sentence_transformer()

        with st.spinner("Computing BERT embeddings..."):
            bert_embeddings = get_bert_embeddings(bert_model, doc_texts)

            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_bert = kmeans.fit_predict(bert_embeddings)

        st.success(f"✓ Created {bert_embeddings.shape[1]}-dimensional BERT embeddings")

        # Visualize
        fig_bert, coords_bert = visualize_clusters_2d(
            bert_embeddings, labels_bert, doc_names, n_clusters,
            "BERT Embeddings Clustering", method=viz_method
        )
        st.pyplot(fig_bert)
        plt.close()

        # Show cluster members
        show_cluster_members(doc_names, doc_texts, labels_bert, n_clusters)

        # Semantic similarity within clusters
        with st.expander("🔍 Cluster Cohesion (Semantic Similarity)"):
            st.markdown("**Average cosine similarity within each cluster** (higher = more similar):")

            from sklearn.metrics.pairwise import cosine_similarity

            for i in range(n_clusters):
                cluster_mask = labels_bert == i
                cluster_embeddings = bert_embeddings[cluster_mask]

                if len(cluster_embeddings) > 1:
                    # Compute pairwise similarities
                    similarities = cosine_similarity(cluster_embeddings)
                    # Get upper triangle (excluding diagonal)
                    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
                    avg_sim = upper_tri.mean()

                    st.metric(f"Cluster {i}", f"{avg_sim:.3f}",
                             help="1.0 = identical, 0.0 = unrelated")
                else:
                    st.metric(f"Cluster {i}", "N/A (single doc)")

        with st.expander("💡 What's happening?"):
            st.markdown("""
            **BERT understands semantic meaning, not just word statistics!**

            **How BERT embeddings work**:
            1. **Pretrained on massive text corpus**: BERT learned language patterns from Wikipedia, books, etc.
            2. **Contextual understanding**: Same word has different embeddings based on context
               - "bank" in "river bank" vs "bank account" → different embeddings
            3. **Sentence-level meaning**: Entire document mapped to fixed-size vector (384 dims)
            4. **Semantic similarity**: Documents with similar meanings have similar embeddings

            **Example**:
            - "Machine learning uses neural networks"
            - "AI systems employ deep learning"

            These have NO words in common, but BERT knows they're semantically similar!

            **Why it's better**:
            - ✅ Captures **meaning**, not just word counts
            - ✅ Understands **synonyms** (ML = machine learning = AI)
            - ✅ Handles **context** (same words in different contexts)
            - ✅ **Robust to paraphrasing** - same idea, different words

            **Result**: Clusters truly reflect document topics and semantic content!

            **This is the state-of-the-art approach** for document clustering and search engines.
            """)

    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        st.info("💡 Install required package: `pip install sentence-transformers`")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
### 🎓 Summary: Document Clustering Approaches

| Method | Features | Dimensions | Speed | Semantic Understanding |
|--------|----------|------------|-------|----------------------|
| **Simple Features** | Statistics | 6D | ⚡⚡⚡ | ❌ Style only |
| **TF-IDF** | Word importance | 50-200D | ⚡⚡ | ⚠️ Keywords only |
| **BERT** | Deep embeddings | 384D | ⚡ | ✅ Full semantics |

**Key insights**:
- 📊 **Simple features**: Cluster by document style (length, complexity) - NOT meaning
- 🔤 **TF-IDF**: Cluster by shared vocabulary - better but treats words independently
- 🧠 **BERT**: Cluster by actual semantic meaning - understands context and synonyms

**Analogy to image clustering**:
- Simple features ← → Position (X, Y)
- TF-IDF ← → RGB colors
- BERT ← → ResNet18 deep features

**Try clustering different document sets and compare the methods!** 🎉
""")
