#!/usr/bin/env python3
"""
Generate all clustering visualization figures for the lecture slides.
Produces beautiful, publication-quality PDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

OUTPUT_DIR = 'figures/'

# Color palette
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

def save_fig(filename):
    """Save figure as PDF with tight layout."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Generated {filename}")

# ==============================================================================
# 1. SUPERVISED VS UNSUPERVISED COMPARISON
# ==============================================================================

def generate_supervised_vs_unsupervised():
    """Generate side-by-side comparison of supervised vs unsupervised."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Supervised: labeled data
    np.random.seed(42)
    X1 = np.random.randn(30, 2) + [0, 0]
    X2 = np.random.randn(30, 2) + [4, 4]

    ax1.scatter(X1[:, 0], X1[:, 1], c=COLORS[0], s=80, alpha=0.7,
                edgecolors='black', linewidth=0.5, label='Class A')
    ax1.scatter(X2[:, 0], X2[:, 1], c=COLORS[1], s=80, alpha=0.7,
                edgecolors='black', linewidth=0.5, label='Class B')
    ax1.set_title('Supervised Learning\n(Labels Known)', fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Unsupervised: unlabeled data
    X_all = np.vstack([X1, X2])
    ax2.scatter(X_all[:, 0], X_all[:, 1], c='gray', s=80, alpha=0.7,
                edgecolors='black', linewidth=0.5)
    ax2.set_title('Unsupervised Learning\n(Labels Unknown - Find Structure!)', fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(alpha=0.3)

    save_fig('supervised_vs_unsupervised.pdf')

# ==============================================================================
# 2. K-MEANS WITH DIFFERENT K VALUES
# ==============================================================================

def generate_kmeans_different_k():
    """Generate K-Means clustering with K=3,4,5,6."""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=5, n_features=2,
                      cluster_std=0.8, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, k in enumerate([3, 4, 5, 6]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        ax = axes[idx]
        for i in range(k):
            cluster_points = X[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=COLORS[i % len(COLORS)], s=50, alpha=0.6,
                      edgecolors='black', linewidth=0.5)

        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200,
                  marker='*', edgecolors='white', linewidth=2,
                  label='Centroids', zorder=10)

        title_suffix = ['(Very Under)', '(Under)', '(Optimal)', '(Over)'][idx]
        ax.set_title(f'K = {k} {title_suffix}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Effect of K on Clustering Quality', fontsize=16, fontweight='bold', y=1.00)
    save_fig('kmeans_different_k.pdf')

# ==============================================================================
# 3. K-MEANS ITERATION ANIMATION (6 STEPS)
# ==============================================================================

def generate_kmeans_iterations():
    """Generate K-Means algorithm step-by-step with clear visual changes."""
    np.random.seed(42)
    # Create 9 points in 3 clear clusters to show dramatic changes
    X = np.array([
        [1, 1], [1.2, 1.5], [1.5, 1],      # Bottom-left cluster
        [5, 1], [5.3, 1.2], [5.5, 1.5],    # Bottom-right cluster
        [3, 5], [3.2, 5.3], [3.5, 5]       # Top cluster
    ])

    # Start with POOR initial centroids (far from true clusters)
    mu = np.array([[2.0, 2.0], [4.0, 2.0], [2.5, 4.0]])  # Suboptimal initialization

    iterations = []
    for iteration in range(3):
        # Assignment step
        distances = cdist(X, mu, 'euclidean')
        labels = np.argmin(distances, axis=1)
        iterations.append(('assign', labels.copy(), mu.copy()))

        # Update step
        mu_new = np.array([X[labels == k].mean(axis=0) for k in range(3)])
        iterations.append(('update', labels.copy(), mu_new.copy()))
        mu = mu_new

    # Generate figures for each step
    for idx, (step_type, labels, centroids) in enumerate(iterations):
        fig, ax = plt.subplots(figsize=(9, 7))

        # Plot points with larger markers
        for k in range(3):
            cluster_points = X[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=COLORS[k], s=250, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=f'Cluster {k+1}')

        # Plot centroids with emphasis
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black',
                  s=500, marker='*', edgecolors='yellow',
                  linewidth=3, label='Centroids', zorder=10)

        # Add lines from points to their centroids (for assignment step)
        if step_type == 'assign':
            for i, point in enumerate(X):
                centroid = centroids[labels[i]]
                ax.plot([point[0], centroid[0]], [point[1], centroid[1]],
                       'k--', alpha=0.3, linewidth=1)

        # Annotations for points
        for i, (x, y) in enumerate(X):
            ax.annotate(f'$x_{{{i+1}}}$', (x, y), xytext=(7, 7),
                       textcoords='offset points', fontsize=12, fontweight='bold')

        # Annotations for centroids
        for k, (cx, cy) in enumerate(centroids):
            ax.annotate(f'$\\mu_{{{k+1}}}$', (cx, cy), xytext=(-15, -20),
                       textcoords='offset points', fontsize=13,
                       fontweight='bold', color='yellow',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        iteration_num = idx // 2
        step_name = 'Assignment' if step_type == 'assign' else 'Update'
        ax.set_title(f'Iteration {iteration_num}: {step_name} Step',
                    fontweight='bold', fontsize=15)
        ax.set_xlabel('Feature 1', fontsize=13)
        ax.set_ylabel('Feature 2', fontsize=13)
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 6)

        save_fig(f'kmeans_iteration_{idx}.pdf')

# ==============================================================================
# 4. ELBOW CURVE
# ==============================================================================

def generate_elbow_curve():
    """Generate elbow curve for choosing K."""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=5, n_features=2,
                      cluster_std=0.8, random_state=42)

    k_values = range(2, 11)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Find elbow using second derivative
    inertias_arr = np.array(inertias)
    diffs = np.diff(inertias_arr)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2
    elbow_k = list(k_values)[elbow_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curve
    ax.plot(k_values, inertias, 'o-', linewidth=2.5, markersize=10,
            color=COLORS[1], label='WCSS (Φ)')

    # Highlight elbow
    ax.axvline(x=elbow_k, color=COLORS[0], linestyle='--', linewidth=2.5,
              alpha=0.7, label=f'Suggested K = {elbow_k}')
    ax.scatter([elbow_k], [inertias[elbow_idx]], color=COLORS[0], s=400,
              marker='*', edgecolors='black', linewidth=2, zorder=10)

    ax.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=13)
    ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontweight='bold', fontsize=13)
    ax.set_title('Elbow Method for Optimal K Selection', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)

    # Add annotation for the elbow point
    ax.annotate(f'Best K = {elbow_k}\n(Elbow point)',
               xy=(elbow_k, inertias[elbow_idx]),
               xytext=(elbow_k + 1.5, inertias[elbow_idx] + 100),
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    save_fig('elbow_curve.pdf')

# ==============================================================================
# 5. FEATURE SCALING COMPARISON
# ==============================================================================

def generate_feature_scaling():
    """Show impact of feature scaling on clustering."""
    np.random.seed(42)

    # Create data with different scales
    X = np.random.randn(100, 2)
    X[:, 0] = X[:, 0] * 1  # Age: small range
    X[:, 1] = X[:, 1] * 100 + 50000  # Income: large range

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Unscaled
    kmeans1 = KMeans(n_clusters=3, random_state=42)
    labels1 = kmeans1.fit_predict(X)

    for k in range(3):
        ax1.scatter(X[labels1 == k, 0], X[labels1 == k, 1],
                   c=COLORS[k], s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1],
               c='black', s=200, marker='*', edgecolors='white', linewidth=2)
    ax1.set_xlabel('Age (years)', fontweight='bold')
    ax1.set_ylabel('Income ($)', fontweight='bold')
    ax1.set_title('Unscaled Features\n(Income Dominates!)', fontweight='bold', color=COLORS[0])
    ax1.grid(alpha=0.3)

    # Scaled
    X_scaled = StandardScaler().fit_transform(X)
    kmeans2 = KMeans(n_clusters=3, random_state=42)
    labels2 = kmeans2.fit_predict(X_scaled)

    for k in range(3):
        ax2.scatter(X_scaled[labels2 == k, 0], X_scaled[labels2 == k, 1],
                   c=COLORS[k], s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],
               c='black', s=200, marker='*', edgecolors='white', linewidth=2)
    ax2.set_xlabel('Age (standardized)', fontweight='bold')
    ax2.set_ylabel('Income (standardized)', fontweight='bold')
    ax2.set_title('Scaled Features\n(Balanced Influence)', fontweight='bold', color=COLORS[2])
    ax2.grid(alpha=0.3)

    plt.suptitle('Impact of Feature Scaling on Clustering', fontsize=16, fontweight='bold')
    save_fig('feature_scaling.pdf')

# ==============================================================================
# 6. NON-CONVEX SHAPES (K-MEANS FAILURE)
# ==============================================================================

def generate_nonconvex_failure():
    """Show K-Means failure on non-convex shapes."""
    np.random.seed(42)

    # Create non-convex data
    X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    X_circles, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Moons - True structure
    axes[0, 0].scatter(X_moons[:100, 0], X_moons[:100, 1], c=COLORS[0], s=50, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='Moon 1')
    axes[0, 0].scatter(X_moons[100:, 0], X_moons[100:, 1], c=COLORS[1], s=50, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='Moon 2')
    axes[0, 0].set_title('Two Moons: True Structure', fontweight='bold', color=COLORS[2])
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_aspect('equal')

    # Moons - K-Means result
    kmeans_moons = KMeans(n_clusters=2, random_state=42)
    labels_moons = kmeans_moons.fit_predict(X_moons)

    for k in range(2):
        axes[0, 1].scatter(X_moons[labels_moons == k, 0], X_moons[labels_moons == k, 1],
                          c=COLORS[k], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0, 1].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
                      c='black', s=200, marker='*', edgecolors='white', linewidth=2)
    axes[0, 1].set_title('Two Moons: K-Means Result (FAILS!)', fontweight='bold', color=COLORS[0])
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_aspect('equal')

    # Circles - True structure
    axes[1, 0].scatter(X_circles[:100, 0], X_circles[:100, 1], c=COLORS[0], s=50, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='Outer')
    axes[1, 0].scatter(X_circles[100:, 0], X_circles[100:, 1], c=COLORS[1], s=50, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='Inner')
    axes[1, 0].set_title('Concentric Circles: True Structure', fontweight='bold', color=COLORS[2])
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Circles - K-Means result
    kmeans_circles = KMeans(n_clusters=2, random_state=42)
    labels_circles = kmeans_circles.fit_predict(X_circles)

    for k in range(2):
        axes[1, 1].scatter(X_circles[labels_circles == k, 0], X_circles[labels_circles == k, 1],
                          c=COLORS[k], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1, 1].scatter(kmeans_circles.cluster_centers_[:, 0], kmeans_circles.cluster_centers_[:, 1],
                      c='black', s=200, marker='*', edgecolors='white', linewidth=2)
    axes[1, 1].set_title('Concentric Circles: K-Means Result (FAILS!)', fontweight='bold', color=COLORS[0])
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_aspect('equal')

    plt.suptitle('K-Means Limitations: Non-Convex Shapes', fontsize=16, fontweight='bold')
    save_fig('nonconvex_failure.pdf')

# ==============================================================================
# 7. DBSCAN SUCCESS ON NON-CONVEX
# ==============================================================================

def generate_dbscan_success():
    """Show DBSCAN success on non-convex shapes."""
    np.random.seed(42)
    X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_moons)

    for k in range(2):
        ax1.scatter(X_moons[labels_kmeans == k, 0], X_moons[labels_kmeans == k, 1],
                   c=COLORS[k], s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='black', s=250, marker='*', edgecolors='white', linewidth=2)
    ax1.set_title('K-Means (Linear Boundaries)\nFAILS!', fontweight='bold',
                 color=COLORS[0], fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.set_aspect('equal')

    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_moons)

    for k in range(2):
        ax2.scatter(X_moons[labels_dbscan == k, 0], X_moons[labels_dbscan == k, 1],
                   c=COLORS[k], s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Mark noise points
    noise = X_moons[labels_dbscan == -1]
    if len(noise) > 0:
        ax2.scatter(noise[:, 0], noise[:, 1], c='gray', s=60, alpha=0.5,
                   edgecolors='black', linewidth=0.5, marker='x')

    ax2.set_title('DBSCAN (Density-Based)\nSUCCEEDS!', fontweight='bold',
                 color=COLORS[2], fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal')

    plt.suptitle('Solution: Use DBSCAN for Non-Convex Shapes', fontsize=16, fontweight='bold')
    save_fig('dbscan_success.pdf')

# ==============================================================================
# 8. HIERARCHICAL CLUSTERING DENDROGRAM
# ==============================================================================

def generate_dendrogram_progressive():
    """Generate progressive dendrogram showing step-by-step building."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage

    np.random.seed(42)
    X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [5, 5], [6, 5]])

    # Compute full linkage
    linkage_matrix = linkage(X, method='average')

    # Create 5 subplots showing progressive building
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    steps = [
        (1, 'Step 1: Merge x₁,x₂\n(height=1.0)'),
        (2, 'Step 2: Merge x₄,x₅\n(height=1.0)'),
        (3, 'Step 3: Merge {x₁,x₂},x₃\n(height=1.21)'),
        (4, 'Step 4: Merge {x₄,x₅},x₆\n(height=1.27)'),
        (5, 'Step 5: Final merge\n(height≈5.2)'),
    ]

    for idx, (num_merges, title) in enumerate(steps):
        ax = axes[idx]

        # Truncate linkage matrix to show only up to this step
        Z_truncated = linkage_matrix[:num_merges]

        # Plot dendrogram
        scipy_dendrogram(Z_truncated, ax=ax,
                        labels=[f'x{i+1}' for i in range(6)],
                        color_threshold=0, above_threshold_color='black')

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel('Distance' if idx == 0 else '', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 6)

    plt.tight_layout()
    save_fig('dendrogram_progressive.pdf')

def generate_dendrogram():
    """Generate beautiful dendrogram."""
    np.random.seed(42)
    X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [5, 5], [6, 5]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1.scatter(X[:, 0], X[:, 1], c=COLORS[1], s=200, alpha=0.7,
               edgecolors='black', linewidth=1.5)
    for i, (x, y) in enumerate(X):
        ax1.annotate(f'$x_{i+1}$', (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')
    ax1.set_title('Data Points', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(alpha=0.3)

    # Dendrogram
    linkage_matrix = linkage(X, method='average')
    dendrogram(linkage_matrix, ax=ax2, labels=[f'$x_{i+1}$' for i in range(6)],
              color_threshold=0, above_threshold_color='black')
    ax2.set_title('Hierarchical Clustering Dendrogram\n(Average Linkage)',
                 fontweight='bold', fontsize=14)
    ax2.set_xlabel('Data Points', fontweight='bold')
    ax2.set_ylabel('Distance', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add horizontal lines to show cuts
    for h in [2.5, 4.5]:
        ax2.axhline(y=h, color=COLORS[0], linestyle='--', linewidth=1.5, alpha=0.5)

    save_fig('dendrogram.pdf')

def generate_convergence_graphic():
    """Generate simple graphic showing Φ decreasing over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulate convergence curve
    iterations = np.arange(0, 11)
    # Exponential decay with some noise
    phi_values = 1000 * np.exp(-0.5 * iterations) + 200

    # Plot the convergence curve
    ax.plot(iterations, phi_values, 'o-', linewidth=3, markersize=12,
            color=COLORS[1], label='Objective Φ')

    # Highlight E-steps and M-steps with annotations
    for i in range(1, 4):
        # E-step arrow (assignment)
        ax.annotate('', xy=(i-0.4, phi_values[i]), xytext=(i-0.4, phi_values[i-1]),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS[0]))
        ax.text(i-0.6, (phi_values[i] + phi_values[i-1])/2, 'E-step\n(assign)',
               fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[0], alpha=0.3))

        # M-step arrow (update)
        if i < len(iterations) - 1:
            ax.annotate('', xy=(i+0.4, phi_values[i]), xytext=(i+0.4, phi_values[i]),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS[2]))
            ax.text(i+0.6, phi_values[i] - 30, 'M-step\n(update)',
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[2], alpha=0.3))

    # Add decreasing arrows to emphasize monotone decrease
    for i in range(1, 6):
        ax.annotate('', xy=(i, phi_values[i] + 15), xytext=(i-1, phi_values[i-1] - 15),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black',
                                 linestyle='--', alpha=0.4))

    # Convergence region
    ax.axhspan(phi_values[-1] - 10, phi_values[-1] + 10,
              alpha=0.2, color='green', label='Converged region')
    ax.text(8, phi_values[-1], 'Converged!\n(Φ stops changing)',
           fontsize=12, ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Iteration Number', fontweight='bold', fontsize=13)
    ax.set_ylabel('Objective Φ (WCSS)', fontweight='bold', fontsize=13)
    ax.set_title('K-Means Convergence: Φ Decreases Monotonically',
                fontweight='bold', fontsize=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(150, phi_values[0] + 100)

    # Add text box with key insight
    textstr = 'Key: Each step can only decrease (or maintain) Φ\nΦ is bounded below by 0 → Must converge!'
    ax.text(0.98, 0.65, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_fig('convergence_graphic.pdf')

def generate_hierarchical_steps():
    """Generate step-by-step hierarchical clustering visualization."""
    np.random.seed(42)
    X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [5, 5], [6, 5]])

    # Manual hierarchical clustering to track merges
    # Step 0: Initial state (all separate)
    # Step 1: Merge x1 and x2 (distance 1.0)
    # Step 2: Merge x4 and x5 (distance 1.0)
    # Step 3: Merge {x1,x2} with x3 (distance ~1.21)
    # Step 4: Merge {x4,x5} with x6 (distance ~1.27)
    # Step 5: Merge {x1,x2,x3} with {x4,x5,x6} (distance ~5.2)

    steps = [
        {
            'title': 'Step 0: Initial State',
            'clusters': [[0], [1], [2], [3], [4], [5]],
            'description': '6 clusters (each point separate)',
        },
        {
            'title': 'Step 1: Merge x₁ and x₂',
            'clusters': [[0, 1], [2], [3], [4], [5]],
            'description': 'Merge at distance 1.0\n5 clusters remain',
            'highlight': [0, 1],
        },
        {
            'title': 'Step 2: Merge x₄ and x₅',
            'clusters': [[0, 1], [2], [3, 4], [5]],
            'description': 'Merge at distance 1.0\n4 clusters remain',
            'highlight': [3, 4],
        },
        {
            'title': 'Step 3: Merge {x₁,x₂} with x₃',
            'clusters': [[0, 1, 2], [3, 4], [5]],
            'description': 'Merge at distance 1.21\n3 clusters remain',
            'highlight': [0, 1, 2],
        },
        {
            'title': 'Step 4: Merge {x₄,x₅} with x₆',
            'clusters': [[0, 1, 2], [3, 4, 5]],
            'description': 'Merge at distance 1.27\n2 clusters remain',
            'highlight': [3, 4, 5],
        },
    ]

    for step_idx, step_info in enumerate(steps):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot points colored by current cluster
        for cluster_idx, cluster in enumerate(step_info['clusters']):
            cluster_points = X[cluster]
            color = COLORS[cluster_idx % len(COLORS)]

            # Draw convex hull or circle around cluster if it has multiple points
            if len(cluster) > 1:
                if len(cluster) == 2:
                    # Draw line connecting two points
                    ax.plot(cluster_points[:, 0], cluster_points[:, 1],
                           'k--', linewidth=2, alpha=0.5)
                else:
                    # Draw convex hull
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        ax.plot(cluster_points[simplex, 0],
                               cluster_points[simplex, 1],
                               'k--', linewidth=2, alpha=0.5)

            # Check if this cluster should be highlighted
            highlight = 'highlight' in step_info and all(i in step_info['highlight'] for i in cluster)
            alpha = 1.0 if highlight else 0.5
            edgewidth = 3 if highlight else 1.5

            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[color], s=300, alpha=alpha,
                      edgecolors='black', linewidth=edgewidth,
                      label=f'Cluster {cluster_idx + 1}')

        # Annotate points
        for i, (x, y) in enumerate(X):
            ax.annotate(f'$x_{{{i+1}}}$', (x, y), xytext=(7, 7),
                       textcoords='offset points', fontsize=13, fontweight='bold')

        ax.set_title(f"{step_info['title']}\n{step_info['description']}",
                    fontweight='bold', fontsize=15)
        ax.set_xlabel('Feature 1', fontsize=13)
        ax.set_ylabel('Feature 2', fontsize=13)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 6)

        save_fig(f'hierarchical_step_{step_idx}.pdf')

# ==============================================================================
# 9. LINKAGE COMPARISON
# ==============================================================================

def generate_linkage_comparison():
    """Compare different linkage criteria."""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=50, centers=3, n_features=2,
                      cluster_std=0.5, random_state=42)

    linkages = ['single', 'complete', 'average']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, linkage_method in zip(axes, linkages):
        clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
        labels = clustering.fit_predict(X)

        for k in range(3):
            cluster_points = X[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=COLORS[k], s=70, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

        ax.set_title(f'{linkage_method.capitalize()} Linkage',
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(alpha=0.3)

    plt.suptitle('Hierarchical Clustering: Different Linkage Criteria',
                fontsize=16, fontweight='bold')
    save_fig('linkage_comparison.pdf')

# ==============================================================================
# 10. K-MEANS++ INITIALIZATION
# ==============================================================================

def generate_kmeans_pp():
    """Visualize K-Means++ vs poor initialization with WCSS comparison."""
    np.random.seed(42)
    # Create well-separated clusters
    centers = np.array([[-4, -4], [0, 4], [5, -2]])
    X, true_labels = make_blobs(n_samples=150, centers=centers,
                                cluster_std=0.8, random_state=42)

    # Try multiple random initializations to find a truly bad one
    worst_wcss = 0
    worst_seed = None
    worst_result = None

    for seed in range(100, 200):
        kmeans_test = KMeans(n_clusters=3, init='random', n_init=1, random_state=seed, max_iter=100)
        kmeans_test.fit(X)
        if kmeans_test.inertia_ > worst_wcss:
            worst_wcss = kmeans_test.inertia_
            worst_seed = seed
            worst_result = (kmeans_test.labels_, kmeans_test.cluster_centers_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Poor initialization: use the worst random seed found
    labels_poor, centers_poor = worst_result
    wcss_poor = worst_wcss

    for k in range(3):
        ax1.scatter(X[labels_poor == k, 0], X[labels_poor == k, 1],
                   c=COLORS[k], s=60, alpha=0.7, edgecolors='black', linewidth=0.8)
    ax1.scatter(centers_poor[:, 0], centers_poor[:, 1],
               c='black', s=300, marker='*', edgecolors='yellow', linewidth=3,
               label='Centroids', zorder=10)
    ax1.set_title(f'Poor Initialization\nWCSS (Φ) = {wcss_poor:.1f}',
                 fontweight='bold', fontsize=14, color=COLORS[0])
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # K-Means++ initialization
    kmeans_pp = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    wcss_pp = kmeans_pp.inertia_

    for k in range(3):
        ax2.scatter(X[labels_pp == k, 0], X[labels_pp == k, 1],
                   c=COLORS[k], s=60, alpha=0.7, edgecolors='black', linewidth=0.8)
    ax2.scatter(kmeans_pp.cluster_centers_[:, 0], kmeans_pp.cluster_centers_[:, 1],
               c='black', s=300, marker='*', edgecolors='yellow', linewidth=3,
               label='Final centroids', zorder=10)
    ax2.set_title(f'K-Means++ Initialization\nWCSS (Φ) = {wcss_pp:.1f}',
                 fontweight='bold', fontsize=14, color=COLORS[2])
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # Add comparison text
    improvement = ((wcss_poor - wcss_pp) / wcss_poor) * 100
    plt.suptitle(f'Importance of Smart Initialization\nK-Means++ reduces WCSS by {improvement:.1f}%',
                fontsize=16, fontweight='bold')
    save_fig('kmeans_pp_comparison.pdf')

# ==============================================================================
# 11. VORONOI DIAGRAM
# ==============================================================================

def generate_voronoi():
    """Generate Voronoi diagram showing K-Means decision boundaries."""
    np.random.seed(42)
    centers = np.array([[1, 1], [5, 1], [3, 4]])
    X, _ = make_blobs(n_samples=180, centers=centers,
                      cluster_std=0.6, random_state=42)

    # Perform K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a mesh to show decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict cluster for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot Voronoi regions with light colors
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[COLORS[0], COLORS[1], COLORS[2]],
                alpha=0.2)

    # Plot decision boundaries (where regions meet)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5],
               colors='black', linewidths=3, linestyles='solid',
               alpha=0.8)

    # Plot data points
    for k in range(3):
        cluster_points = X[kmeans.labels_ == k]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  c=COLORS[k], s=60, alpha=0.7,
                  edgecolors='black', linewidth=0.8,
                  label=f'Cluster {k+1}')

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1],
              c='black', s=500, marker='*',
              edgecolors='yellow', linewidth=3,
              label='Centroids', zorder=10)

    # Add centroid labels
    for k, (cx, cy) in enumerate(centroids):
        ax.annotate(f'$\\mu_{{{k+1}}}$', (cx, cy), xytext=(10, 10),
                   textcoords='offset points', fontsize=14,
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='yellow', alpha=0.8))

    ax.set_title('K-Means Decision Boundaries (Voronoi Diagram)\nLinear boundaries: equidistant from centroids',
                fontweight='bold', fontsize=15)
    ax.set_xlabel('Feature 1', fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature 2', fontweight='bold', fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    save_fig('voronoi_diagram.pdf')

# ==============================================================================
# 12. CLUSTER ASSIGNMENT ILLUSTRATION
# ==============================================================================

def generate_cluster_assignment():
    """Illustrate cluster assignment with union and intersection."""
    np.random.seed(42)

    # Create 8 points in 2 clusters
    X = np.array([[1, 1], [1.5, 1.2], [1.2, 1.5], [1.8, 1.3],
                  [5, 4], [5.5, 4.2], [5.2, 4.5], [5.8, 4.3]])

    # Manual cluster assignment
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot points with different colors and markers for each cluster
    for k in range(2):
        cluster_points = X[labels == k]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  c=COLORS[k], s=300, alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label=f'Cluster $C_{k+1}$', zorder=5)

    # Annotate each point with its index
    for i, (x, y) in enumerate(X):
        ax.annotate(f'$x_{i+1}$', (x, y), xytext=(0, 8),
                   textcoords='offset points', fontsize=14,
                   fontweight='bold', ha='center')

    # Draw circles around clusters to visualize them
    from matplotlib.patches import Ellipse

    # Cluster 1
    c1_center = X[labels == 0].mean(axis=0)
    ell1 = Ellipse(c1_center, width=1.5, height=1.0,
                   angle=20, alpha=0.2, color=COLORS[0], linewidth=2,
                   edgecolor=COLORS[0], linestyle='--')
    ax.add_patch(ell1)

    # Cluster 2
    c2_center = X[labels == 1].mean(axis=0)
    ell2 = Ellipse(c2_center, width=1.5, height=1.0,
                   angle=20, alpha=0.2, color=COLORS[1], linewidth=2,
                   edgecolor=COLORS[1], linestyle='--')
    ax.add_patch(ell2)

    # Add text annotations for set notation
    ax.text(1.4, 2.8, r'$C_1 = \{x_1, x_2, x_3, x_4\}$',
            fontsize=16, bbox=dict(boxstyle='round', facecolor=COLORS[0], alpha=0.3),
            fontweight='bold')
    ax.text(5.3, 5.7, r'$C_2 = \{x_5, x_6, x_7, x_8\}$',
            fontsize=16, bbox=dict(boxstyle='round', facecolor=COLORS[1], alpha=0.3),
            fontweight='bold')

    # Add mathematical properties at the bottom
    props_text = (
        r'$C_1 \cup C_2 = \{x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8\}$ (All points)'
        '\n\n'
        r'$C_1 \cap C_2 = \emptyset$ (No overlap - hard assignment)'
    )
    ax.text(3.5, 0.2, props_text,
            fontsize=15, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     edgecolor='black', linewidth=2, alpha=0.9),
            fontweight='bold')

    ax.set_title('Cluster Assignment: Union and Intersection Properties',
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Feature 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=13, fontweight='bold')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6.5)

    save_fig('cluster_assignment.pdf')

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("GENERATING CLUSTERING VISUALIZATION FIGURES")
    print("=" * 70)

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/11] Supervised vs Unsupervised...")
    generate_supervised_vs_unsupervised()

    print("[2/11] K-Means with different K values...")
    generate_kmeans_different_k()

    print("[3/11] K-Means iterations (step-by-step)...")
    generate_kmeans_iterations()

    print("[4/11] Elbow curve...")
    generate_elbow_curve()

    print("[5/11] Feature scaling comparison...")
    generate_feature_scaling()

    print("[6/11] Non-convex failure cases...")
    generate_nonconvex_failure()

    print("[7/11] DBSCAN success on non-convex...")
    generate_dbscan_success()

    print("[8/15] Hierarchical clustering dendrogram...")
    generate_dendrogram()

    print("[9/15] Progressive dendrogram...")
    generate_dendrogram_progressive()

    print("[10/15] Convergence graphic...")
    generate_convergence_graphic()

    print("[11/15] Hierarchical clustering step-by-step...")
    generate_hierarchical_steps()

    print("[12/15] Linkage comparison...")
    generate_linkage_comparison()

    print("[13/15] K-Means++ initialization...")
    generate_kmeans_pp()

    print("[14/15] Voronoi diagram...")
    generate_voronoi()

    print("[15/15] Cluster assignment visualization...")
    generate_cluster_assignment()

    print("\n" + "=" * 70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()

