# Unsupervised Learning: Clustering Lecture

**Comprehensive 120-minute lecture on clustering algorithms with beautiful Python-generated visualizations**

## 📊 What's Included

- **Duration**: ~120 minutes
- **Format**: Typst slides (modern LaTeX alternative)
- **Visualizations**: 17 publication-quality figures generated from Python
- **Topics**: K-Means, Hierarchical Clustering, DBSCAN, practical considerations

## 🚀 Quick Start

### Build Everything

```bash
cd /Users/nipun/git/ml-teaching/unsupervised/slides
./build.sh
```

This will:
1. Generate all figures from Python scripts
2. Copy figures to slides directory
3. Compile Typst slides to PDF

### Manual Steps

#### 1. Generate Figures

```bash
cd ../assets/clustering
python3 generate_figures.py
```

#### 2. Compile Slides

```bash
cd ../../slides
typst compile clustering-lecture.typ
```

## 📂 File Structure

```
unsupervised/
├── slides/
│   ├── clustering-lecture.typ       # Main slide deck (with figures)
│   ├── clustering-lecture.pdf       # Compiled presentation
│   ├── build.sh                     # Build script
│   ├── README.md                    # This file
│   └── figures/                     # Generated visualizations (copied)
└── assets/
    └── clustering/
        ├── generate_figures.py      # Python script for all visualizations
        └── figures/                 # Generated PDFs (original location)
```

## 🎨 Generated Visualizations

The Python script generates 17 beautiful figures:

1. **supervised_vs_unsupervised.pdf** - Side-by-side comparison
2. **kmeans_different_k.pdf** - Effect of K (K=3,4,5,6)
3. **kmeans_iteration_0-5.pdf** - Step-by-step algorithm (6 frames)
4. **elbow_curve.pdf** - Optimal K selection
5. **feature_scaling.pdf** - Scaled vs unscaled comparison
6. **nonconvex_failure.pdf** - K-Means failure on moons/circles
7. **dbscan_success.pdf** - DBSCAN succeeds where K-Means fails
8. **dendrogram.pdf** - Hierarchical clustering tree
9. **linkage_comparison.pdf** - Single/Complete/Average linkage
10. **kmeans_pp_comparison.pdf** - Random vs K-Means++ init
11. **voronoi_diagram.pdf** - Decision boundaries

## 📖 Lecture Structure

### Section 1: Motivation (10 min)
- Supervised vs Unsupervised learning
- Why unsupervised learning?
- Real-world applications
- Clustering fundamentals

### Section 2: K-Means Deep Dive (55 min)
- Problem setup and objective function
- Derivation of centroid formula
- Algorithm (E-step & M-step)
- Step-by-step example with 6 points
- Convergence proof
- Voronoi decision boundaries

### Section 3: Practical Issues (40 min)
- Poor initialization → K-Means++
- Feature scaling importance
- Time complexity analysis
- Mini-Batch K-Means for large datasets
- Choosing K (Elbow method, Silhouette)
- Non-convex shapes failure
- DBSCAN as alternative

### Section 4: Hierarchical Clustering (15 min)
- When K-Means fails
- Dendrograms and linkage criteria
- Comparison with K-Means
- When to use each method

## 🔧 Dependencies

### Python Requirements

```bash
pip install numpy matplotlib scikit-learn scipy
```

### Typst Installation

```bash
# macOS (Homebrew)
brew install typst

# Or download from https://github.com/typst/typst/releases
```

### Touying Package

The slides use `@preview/touying:0.5.5` which is automatically downloaded by Typst on first compile.

## 🎓 Teaching Notes

### Suggested Flow

1. **Motivation** (10 min) - Get students excited, show real applications
2. **K-Means Intuition** (15 min) - Visual before math, show iterations
3. **K-Means Math** (20 min) - Derive objective, prove convergence
4. **K-Means Practice** (20 min) - Work through 6-point example
5. **Break** (5 min)
6. **Practical Issues** (30 min) - Live demo with scikit-learn
7. **Advanced Topics** (10 min) - DBSCAN, hierarchical
8. **Hierarchical** (15 min) - Dendrogram interpretation
9. **Q&A** (5 min)

### Interactive Elements

- Show K-Means iterations one by one (use `#pause` in slides)
- Ask students to predict next centroid positions
- Discuss when K-Means might fail (before showing failure modes)
- Live coding demo with scikit-learn (use Google Colab)

### Common Student Questions

**Q: Why not always use K-Means++?**
A: You should! It's the default in scikit-learn. No downside.

**Q: How to choose K in practice?**
A: Combine elbow + silhouette + domain knowledge. No single answer.

**Q: Why does K-Means fail on moons/circles?**
A: Linear decision boundaries (Voronoi). Use DBSCAN/Spectral for non-convex.

**Q: When to use hierarchical vs K-Means?**
A: Hierarchical for n<10k and need hierarchy. K-Means for large data.

## 🔄 Updating Figures

To regenerate all figures with different styling:

1. Edit `../assets/clustering/generate_figures.py`
2. Modify colors, sizes, DPI in the script header
3. Run `./build.sh` to regenerate everything

## 📚 Additional Resources

- **Google Colab Notebook**: [Add link to your notebook]
- **scikit-learn docs**: https://scikit-learn.org/stable/modules/clustering.html
- **ISLR Chapter 12**: Unsupervised Learning
- **Original K-Means++ paper**: Arthur & Vassilvitskii, 2007

## 📝 License

Part of the ML Teaching repository by Nipun Batra, IIT Gandhinagar.

## 🤝 Contributing

To add new visualizations:

1. Add generation function to `generate_figures.py`
2. Call it in `main()`
3. Use the figure in `clustering-lecture.typ`
4. Run `./build.sh` to test

## 💡 Tips

- **High DPI**: Figures are generated at 300 DPI for projection
- **Colors**: Use the COLORS palette defined in the script for consistency
- **Font**: Serif fonts for publication quality
- **Size**: Keep figures under 1MB each for fast compilation

## 📧 Contact

Nipun Batra - nipun.batra@iitgn.ac.in

IIT Gandhinagar
