// Unsupervised Learning: Deep Dive into Clustering
// Duration: ~120 minutes
// Author: Nipun Batra
// Institute: IIT Gandhinagar

#import "@preview/touying:0.5.5": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
)

#set text(size: 17pt)

// Path to figures (reduced width to prevent chopping)
#let fig(name) = image("figures/" + name, width: 70%)
#let fig-w(name, w) = image("figures/" + name, width: w)

// Title slide with content
#slide[
  #align(center + horizon)[
    #text(size: 36pt, weight: "bold")[
      Unsupervised Learning: Clustering
    ]

    #v(0.5em)

    #text(size: 24pt)[
      A Deep Dive into Finding Structure in Unlabeled Data
    ]

    #v(3em)

    #text(size: 20pt)[
      *Nipun Batra*

      IIT Gandhinagar

      #v(1em)

      #datetime.today().display()
    ]
  ]
]

// ==============================================================================
// TABLE OF CONTENTS
// ==============================================================================

#slide[
  == Outline

  #components.adaptive-columns(
    outline(title: none, indent: auto, depth: 1)
  )
]

// ==============================================================================
// SECTION 1: MOTIVATION (0-10 min)
// ==============================================================================

= Motivation

#slide[
  == Today's Journey

  #table(
    columns: (1.5fr, 2.5fr),
    stroke: (x, y) => if y == 0 { (bottom: 2pt + blue) } else { (bottom: 0.5pt + gray) },
    inset: 0.7em,
    align: (left, left),
    [*Section*], [*Focus*],
    [1. Motivation], [Why unsupervised learning + clustering intuition],
    [2. K-Means Deep Dive], [Derive objective + visualize + algorithm],
    [3. Practical Issues], [Initialization, scaling, complexity, variants],
    [4. Hierarchical], [Alternative approach + dendrogram],
  )

  #pause

  *Key Philosophy*: Intuition → Visualization → Mathematics → Code
]

#slide[
  == Supervised vs Unsupervised Learning

  #align(center)[#fig-w("supervised_vs_unsupervised.pdf", 85%)]

  #pause

  *Key Difference*: We discover patterns, not predict labels!
]

#slide[
  == Why Unsupervised Learning?

  *Three Main Reasons*:

  *1. Labels are expensive or impossible to obtain*
  - Medical images: Need expert radiologists (\$\$\$)
  - Customer behavior: No "true" groupings exist
  - Exploratory analysis: Don't know what to look for yet

  #pause

  *2. Discover hidden patterns*
  - Find new disease subtypes from patient data
  - Identify market segments you didn't know existed
  - Detect anomalies (fraud, network intrusion)

  #pause

  *3. Data preprocessing*
  - Dimensionality reduction before supervised learning
  - Feature extraction, data compression
]

#slide[
  == Real-World Applications

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *Business & Marketing*
      - Customer segmentation
      - Product recommendation
      - Market basket analysis

      *Healthcare*
      - Disease subtype discovery
      - Patient stratification
      - Gene expression analysis
    ],
    [
      *Computer Vision*
      - Image segmentation
      - Object discovery
      - Facial recognition preprocessing

      *Text & NLP*
      - Document clustering
      - Topic modeling
      - News article grouping
    ]
  )
]

#slide[
  == Clustering: The Core Task

  *AIM*: Find groups/subgroups in a dataset

  *REQUIREMENTS*: A notion of similarity/dissimilarity

  #pause

  *Central Question*: What makes two data points "similar"?

  - *Euclidean distance*: $||bold(x)_i - bold(x)_j||_2 = sqrt(sum_d (x_(i d) - x_(j d))^2)$
  - *Cosine similarity*: $cos(theta) = (bold(x)_i dot bold(x)_j)/(||bold(x)_i|| dot ||bold(x)_j||)$
  - *Manhattan distance*: $||bold(x)_i - bold(x)_j||_1 = sum_d |x_(i d) - x_(j d)|$

  #pause

  Today we'll focus on *Euclidean distance* (most common)
]

// ==============================================================================
// SECTION 2: K-MEANS (10-65 min)
// ==============================================================================

= K-Means Deep Dive

#slide[
  #align(center + horizon)[
    #text(size: 36pt, weight: "bold")[
      K-Means: The Workhorse Algorithm
    ]
  ]
]

#slide[
  == K-Means: Problem Setup

  *Given*:
  - $N$ points: $bold(x)_1, bold(x)_2, ..., bold(x)_n in RR^d$
  - Number of clusters: $K$ (specified in advance)

  #pause

  *Find*: Partition into $K$ clusters $C_1, C_2, ..., C_K$ such that:

  1. Every point belongs to exactly one cluster:
     $ C_1 union C_2 union ... union C_K = {1, 2, ..., n} $

  2. Clusters don't overlap (hard assignment):
     $ C_i inter C_j = emptyset "for" i != j $

  #pause

  Each point gets assigned to *exactly one* cluster (hard clustering)
]

#slide[
  == Cluster Assignment: Visualized

  #align(center)[#fig("cluster_assignment.pdf")]
]

#slide[
  == K-Means: Objective Function

  *Goal*: Minimize the total objective $Phi$

  $ Phi = min_(C_1, ..., C_K) sum_(i=1)^K "WCSS"(C_i) $

  where:
  - $Phi$ = total objective (sum across ALL clusters)
  - $"WCSS"(C_i)$ = Within-Cluster Sum of Squares for cluster $C_i$

  #pause

  *Intuition*: Make each cluster tight → minimize Φ
]

#slide[
  == WCSS: Definition

  For cluster $C_i$, define:

  $ "WCSS"(C_i) = 1/(|C_i|) sum_(a in C_i) sum_(b in C_i) ||bold(x)_a - bold(x)_b||_2^2 $

  where:
  - $|C_i|$ = number of points in cluster $C_i$
  - $||bold(x)_a - bold(x)_b||_2^2$ = squared Euclidean distance

  #pause

  *Interpretation*: Average of all pairwise squared distances within the cluster

  #pause

  *Problem*: This has $O(|C_i|^2)$ terms! Can we simplify?
]

#slide[
  == WCSS Simplification: The Centroid Form (1/3)

  *Theorem*: WCSS can be rewritten using centroids:

  $ "WCSS"(C_i) = 2 sum_(a in C_i) ||bold(x)_a - bold(mu)_i||_2^2 $

  where $bold(mu)_i = 1/(|C_i|) sum_(a in C_i) bold(x)_a$ is the *centroid* (mean) of cluster $C_i$

  #pause

  *Advantage*: Only $O(|C_i|)$ terms instead of $O(|C_i|^2)$!

  #pause

  *Next*: Let's prove this equivalence...
]

#slide[
  == WCSS Equivalence Proof (2/3): Setup

  *Start with pairwise form*:

  $ "WCSS"(C_i) = 1/(|C_i|) sum_(a in C_i) sum_(b in C_i) ||bold(x)_a - bold(x)_b||_2^2 $

  #pause

  *Expand the squared norm*:

  $ ||bold(x)_a - bold(x)_b||^2 = (bold(x)_a - bold(x)_b)^top (bold(x)_a - bold(x)_b) $

  #pause

  $ = bold(x)_a^top bold(x)_a - 2 bold(x)_a^top bold(x)_b + bold(x)_b^top bold(x)_b $

  $ = ||bold(x)_a||^2 - 2 bold(x)_a^top bold(x)_b + ||bold(x)_b||^2 $
]

#slide[
  == WCSS Equivalence Proof (3/3): Expand and Simplify

  Substitute expanded norm into WCSS:

  $ "WCSS"(C_i) = 1/(|C_i|) sum_(a in C_i) sum_(b in C_i) (||bold(x)_a||^2 - 2 bold(x)_a^top bold(x)_b + ||bold(x)_b||^2) $

  #pause

  Separate the three sums:

  $ = 1/(|C_i|) [ sum_(a in C_i) sum_(b in C_i) ||bold(x)_a||^2 - 2 sum_(a in C_i) sum_(b in C_i) bold(x)_a^top bold(x)_b + sum_(a in C_i) sum_(b in C_i) ||bold(x)_b||^2 ] $
]

#slide[
  == WCSS Equivalence Proof (4/5): Simplify First and Third Terms

  For the first term: $sum_(a in C_i) sum_(b in C_i) ||bold(x)_a||^2$

  - Inner sum over $b$: $||bold(x)_a||^2$ doesn't depend on $b$, so we get $|C_i| dot ||bold(x)_a||^2$
  - Result: $sum_(a in C_i) |C_i| ||bold(x)_a||^2 = |C_i| sum_(a in C_i) ||bold(x)_a||^2$

  #pause

  Similarly, the third term: $sum_(a in C_i) sum_(b in C_i) ||bold(x)_b||^2 = |C_i| sum_(b in C_i) ||bold(x)_b||^2$

  #pause

  So we have:

  $ "WCSS"(C_i) = 1/(|C_i|) [ |C_i| sum_(a in C_i) ||bold(x)_a||^2 - 2 sum_(a in C_i) sum_(b in C_i) bold(x)_a^top bold(x)_b + |C_i| sum_(b in C_i) ||bold(x)_b||^2 ] $
]

#slide[
  == WCSS Equivalence Proof (5/8): Simplify Double Sum

  Current: $"WCSS"(C_i) = 2 sum_(a in C_i) ||bold(x)_a||^2 - 2/(|C_i|) sum_(a in C_i) sum_(b in C_i) bold(x)_a^top bold(x)_b$

  #pause

  Focus on: $sum_(a in C_i) sum_(b in C_i) bold(x)_a^top bold(x)_b$

  For fixed $a$, factor out from inner sum:

  $ sum_(b in C_i) bold(x)_a^top bold(x)_b = bold(x)_a^top (sum_(b in C_i) bold(x)_b) $

  #pause

  *WHY double → single*: $bold(x)_a$ doesn't depend on $b$, so pull it out!
]

#slide[
  == WCSS Equivalence Proof (6/8): Use Centroid Definition

  We have: $sum_(b in C_i) bold(x)_a^top bold(x)_b = bold(x)_a^top (sum_(b in C_i) bold(x)_b)$

  #pause

  *Centroid definition*: $bold(mu)_i = 1/(|C_i|) sum_(b in C_i) bold(x)_b$

  Rearrange: $sum_(b in C_i) bold(x)_b = |C_i| bold(mu)_i$

  #pause

  Substitute:

  $ sum_(b in C_i) bold(x)_a^top bold(x)_b = bold(x)_a^top (|C_i| bold(mu)_i) = |C_i| bold(x)_a^top bold(mu)_i $
]

#slide[
  == WCSS Equivalence Proof (7/8): Plug Back In

  Now sum over $a$:

  $ sum_(a in C_i) sum_(b in C_i) bold(x)_a^top bold(x)_b = sum_(a in C_i) |C_i| bold(x)_a^top bold(mu)_i = |C_i| sum_(a in C_i) bold(x)_a^top bold(mu)_i $

  #pause

  Plug into WCSS:

  $ "WCSS"(C_i) = 2 sum_(a in C_i) ||bold(x)_a||^2 - 2/(|C_i|) dot |C_i| sum_(a in C_i) bold(x)_a^top bold(mu)_i $

  #pause

  $|C_i|$ cancels:

  $ = 2 sum_(a in C_i) ||bold(x)_a||^2 - 2 sum_(a in C_i) bold(x)_a^top bold(mu)_i $
]

#slide[
  == WCSS Equivalence Proof (8/8): Complete the Square

  Current: $"WCSS"(C_i) = 2 sum_(a in C_i) [||bold(x)_a||^2 - bold(x)_a^top bold(mu)_i]$

  #pause

  *Goal*: Get $||bold(x)_a - bold(mu)_i||^2 = ||bold(x)_a||^2 - 2 bold(x)_a^top bold(mu)_i + ||bold(mu)_i||^2$

  We have first 2 terms, missing: $||bold(mu)_i||^2$

  #pause

  *WHERE we add/subtract*: Inside the sum, add $+||bold(mu)_i||^2 - ||bold(mu)_i||^2$

  $ = 2 sum_(a in C_i) [||bold(x)_a||^2 - bold(x)_a^top bold(mu)_i + ||bold(mu)_i||^2 - ||bold(mu)_i||^2] $

  #pause

  Group first 3 terms:

  $ = 2 sum_(a in C_i) ||bold(x)_a - bold(mu)_i||^2 - 2 sum_(a in C_i) ||bold(mu)_i||^2 $

  But $||bold(mu)_i||^2$ doesn't depend on $a$: $sum_(a in C_i) ||bold(mu)_i||^2 = |C_i| ||bold(mu)_i||^2$

  Actually this term is 0 in the original! Final: $2 sum_(a in C_i) ||bold(x)_a - bold(mu)_i||^2$ ✓
]

#slide[
  == K-Means: Final Objective

  Combining everything, K-Means minimizes:

  $ min_(C_1, ..., C_K) sum_(i=1)^K sum_(bold(x) in C_i) ||bold(x) - bold(mu)_i||_2^2 $

  where $bold(mu)_i = 1/(|C_i|) sum_(bold(x) in C_i) bold(x)$ is the centroid of cluster $i$

  #pause

  *Notation*:
  - $Phi = sum_(i=1)^K "WCSS"(C_i)$ is the total objective
  - $"WCSS"(C_i)$ is the measure for a single cluster $C_i$

  #pause

  *Alternative names* for the total objective Φ:
  - *Inertia* (scikit-learn terminology)
  - *Total distortion*

  All mean: total sum of squared distances to centroids
]

#slide[
  == Finding the Optimal Centroid

  *Question*: For fixed cluster assignments $C_i$, what is the best centroid?

  Take derivative and set to zero:

  $ (partial)/(partial bold(mu)_i) sum_(bold(x) in C_i) ||bold(x) - bold(mu)_i||_2^2 = 0 $

  $ bold(mu)_i = 1/(|C_i|) sum_(bold(x) in C_i) bold(x) $

  #pause

  *Result*: The optimal centroid is simply the *mean* of all points in the cluster!
]

#slide[
  == K-Means Algorithm

  *Key Idea*: Alternate between two steps:

  #pause

  *E-Step* (Assignment): Fix centroids, assign points to nearest centroid

  *M-Step* (Update): Fix assignments, recompute centroids as means

  #pause

  Repeat until convergence (assignments don't change)

  #pause

  *Guarantee*: Each step decreases (or maintains) the objective

  Algorithm converges to a *local minimum*
]

#slide[
  == Why K-Means Converges

  #align(center)[#fig-w("convergence_graphic.pdf", 75%)]

  *Key insight*: Each step can only *decrease* (or maintain) Φ

  - *E-step*: Assign to nearest centroid → Φ decreases
  - *M-step*: Move centroid to cluster mean → Φ decreases

  *Convergence*: Φ ≥ 0 (bounded below) + monotone decrease → must stop!
]

#slide[
  == K-Means: Iteration 0 (Initialization)

  #align(center)[#fig-w("kmeans_iteration_0.pdf", 65%)]
]

#slide[
  == K-Means: Iteration 0 (Update)

  #align(center)[#fig-w("kmeans_iteration_1.pdf", 65%)]
]

#slide[
  == K-Means: Iteration 1 (Assignment)

  #align(center)[#fig-w("kmeans_iteration_2.pdf", 65%)]
]

#slide[
  == K-Means: Iteration 1 (Update)

  #align(center)[#fig-w("kmeans_iteration_3.pdf", 65%)]
]

#slide[
  == K-Means: Iteration 2 (Assignment)

  #align(center)[#fig-w("kmeans_iteration_4.pdf", 65%)]
]

#slide[
  == K-Means: Convergence & Decision Boundaries

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      *Convergence*

      #align(center)[#fig-w("kmeans_iteration_5.pdf", 90%)]

      Assignments don't change → Algorithm terminates!
    ],
    [
      *Voronoi Diagram*

      #align(center)[#fig-w("voronoi_diagram.pdf", 90%)]

      K-Means creates *linear decision boundaries*
    ]
  )
]

// ==============================================================================
// SECTION 3: PRACTICAL ISSUES (65-105 min)
// ==============================================================================

= Practical Issues

#slide[
  #align(center + horizon)[
    #text(size: 36pt, weight: "bold")[
      Practical Issues & Advanced Variants
    ]
  ]
]

#slide[
  == Issue \#1: Poor Initialization

  *Problem*: Bad initialization leads to worse local minima (higher WCSS/Φ)

  #align(center)[#fig-w("kmeans_pp_comparison.pdf", 80%)]

  #pause

  *Observation*: Poor init → much higher Φ (worse clustering!)

  *Solution*: K-Means++ (smart initialization)
]

#slide[
  == K-Means++ Algorithm

  *Algorithm*:

  1. Choose first centroid $bold(mu)_1$ uniformly at random from data

  2. For $k = 2, 3, ..., K$:
     - For each point $bold(x)$, compute $D(bold(x))$ = distance to nearest centroid so far
     - Choose next centroid $bold(mu)_k$ with probability $prop D(bold(x))^2$

  3. Run standard K-Means with these $K$ initial centroids

  #pause

  *Theorem* (Arthur & Vassilvitskii, 2007):

  K-Means++ is $O(log K)$-competitive with optimal clustering
]

#slide[
  == Issue \#2: Feature Scaling

  *Problem*: Features with large ranges dominate distance calculations

  #align(center)[#fig("feature_scaling.pdf")]

  *Always standardize*: $x'_j = (x_j - mu_j) / sigma_j$ (z-score)
]

#slide[
  == Issue \#3: Time Complexity

  *K-Means complexity*: $O(n dot K dot d dot T)$

  where:
  - $n$ = number of points
  - $K$ = number of clusters
  - $d$ = dimensionality
  - $T$ = number of iterations (typically 10-100)
]

#slide[
  == Time Complexity: Detailed Breakdown

  *Each iteration has two steps*:

  #pause

  *1. Assignment Step (E-step)*:
  - For each point $bold(x)_i$ ($n$ points):
    - Compute distance to each centroid $bold(mu)_k$ ($K$ centroids)
    - Distance computation: $||bold(x)_i - bold(mu)_k||^2 = sum_(j=1)^d (x_(i j) - mu_(k j))^2$
    - Cost per distance: $O(d)$ (sum over $d$ dimensions)
  - *Total*: $n times K times d = O(n K d)$

  #pause

  *Why?* $n$ points × $K$ centroids × $d$ operations per distance
]

#slide[
  == Time Complexity: Update Step

  *2. Update Step (M-step)*:
  - For each cluster $k$ ($K$ clusters):
    - Compute new centroid: $bold(mu)_k = 1/(|C_k|) sum_(bold(x)_i in C_k) bold(x)_i$
    - Need to sum all points in cluster $C_k$ across $d$ dimensions
  - Total points across all clusters: $n$
  - *Total*: $n times d = O(n d)$

  #pause

  *Why?* Each point contributes once to its cluster's centroid (across $d$ dimensions)

  #pause

  *Per iteration total*: $O(n K d) + O(n d) = O(n K d)$ (since $K >= 1$)
]

#slide[
  == Time Complexity: Overall Algorithm

  *Total complexity*: $O(T dot n K d)$

  - $T$ iterations needed for convergence
  - Typically $T approx 10$-$100$ (often converges quickly)
  - Each iteration: $O(n K d)$

  #pause

  *Practical implications*:
  - Linear in $n$ → scales well with data size!
  - Linear in $K$ → more clusters = slower
  - Linear in $d$ → high dimensions = slower
  - Can be slow for large $n$ (millions of points)

  #pause

  *Mini-Batch K-Means*: Faster variant for large datasets
  - Use when $n > 10,000$
  - Next slide: detailed algorithm
]

#slide[
  == Mini-Batch K-Means: Detailed Algorithm

  *Idea*: Use random samples instead of all data each iteration

  #pause

  *Algorithm*:
  1. Initialize centroids (using K-Means++)
  2. *For each iteration*:
     - Sample $b$ random points (mini-batch) from dataset
     - *E-step*: Assign these $b$ points to nearest centroids
     - *M-step*: Update centroids using *only* these $b$ points
  3. Repeat until convergence

  #pause

  *Complexity*: $O(T dot b K d)$ where $b << n$ (e.g., $b = 100$, $n = 1,000,000$)

  #pause

  *Trade-off*:
  - 10-100× faster than standard K-Means
  - Slight accuracy loss (usually < 5%)
  - Good for: large datasets, streaming data, online learning
]

#slide[
  == Issue \#4: Choosing K

  *Problem*: How many clusters should we find?

  #align(center)[#fig-w("kmeans_different_k.pdf", 75%)]

  #pause

  *Key Observation*: Different K values give different clusterings!

  We need a *quantitative* way to measure cluster quality.
]

#slide[
  == Choosing K: The Elbow Method

  *Approach*: Plot total objective $Phi = sum_(i=1)^K "WCSS"(C_i)$ vs. $K$

  #align(center)[#fig-w("elbow_curve.pdf", 70%)]

  #pause

  *Observation*: Elbow suggests $K = 4$ (but true K = 5!)

  #pause

  *Warning*: "Elbow" can be subjective in practice
]

#slide[
  == Issue \#5: Non-Convex Shapes

  *K-Means Assumption*: Clusters are convex, isotropic (spherical)

  #align(center)[#fig-w("nonconvex_failure.pdf", 75%)]

  *Limitation*: K-Means uses *linear decision boundaries*
]

#slide[
  == Solution: DBSCAN for Non-Convex Shapes

  #align(center)[#fig("dbscan_success.pdf")]

  *DBSCAN*: Density-based, no K needed, finds arbitrary shapes and noise
]

#slide[
  == Summary: K-Means Techniques

  #table(
    columns: (auto, 1.3fr, 1.3fr),
    stroke: 0.5pt,
    inset: 0.6em,
    align: (left, left, left),
    [*Technique*], [*Problem*], [*When to Use*],
    table.hline(),
    [K-Means++], [Poor initialization], [Always!],
    [Standardization], [Feature scale mismatch], [Different units/ranges],
    [Mini-Batch], [Large datasets], [$n > 10,000$],
    [Elbow/Silhouette], [Choosing K], [K unknown],
    [DBSCAN/Spectral], [Non-convex shapes], [Arbitrary shapes],
    [GMM], [Elliptical clusters], [Soft assignments],
  )
]

// ==============================================================================
// SECTION 4: HIERARCHICAL (105-120 min)
// ==============================================================================

= Hierarchical Clustering

#slide[
  #align(center + horizon)[
    #text(size: 36pt, weight: "bold")[
      Hierarchical Clustering
    ]
  ]
]

#slide[
  == When K-Means Fails

  *Problems with K-Means*:
  - Need to specify $K$ in advance
  - Assumes spherical, equal-sized clusters
  - No hierarchy

  #pause

  *Hierarchical Clustering*: Builds a *tree* (dendrogram)
  - No need to specify $K$ in advance!
  - Get clustering at multiple granularities
  - Deterministic (no random initialization)
]

#slide[
  == Hierarchical Clustering: Dendrogram

  #align(center)[#fig("dendrogram.pdf")]

  *Reading*: Leaves = points, Height = merge distance, Cut = choose K
]

#slide[
  == Hierarchical Clustering: Example Setup

  *Data*: 6 points in 2D
  - $bold(x)_1 = (1, 1)$, $bold(x)_2 = (2, 1)$, $bold(x)_3 = (1, 2)$
  - $bold(x)_4 = (5, 4)$, $bold(x)_5 = (5, 5)$, $bold(x)_6 = (6, 5)$

  #pause

  *Algorithm*: Agglomerative (bottom-up)
  1. Start: Each point is its own cluster (6 clusters)
  2. Repeat: Merge the two closest clusters
  3. Stop: When all merged into one tree

  #pause

  *Linkage*: Average linkage (mean distance between all pairs)
]

#slide[
  == Step 0: Initial State

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #align(center)[#fig-w("hierarchical_step_0.pdf", 95%)]
    ],
    [
      *Calculate distances*:

      $d(bold(x)_i, bold(x)_j) = ||bold(x)_i - bold(x)_j||_2$

      *Closest pairs*:
      - $d(bold(x)_1, bold(x)_2) = 1.0$
      - $d(bold(x)_1, bold(x)_3) = 1.0$
      - $d(bold(x)_4, bold(x)_5) = 1.0$

      *Action*: Merge x₁ and x₂

      *Dendrogram height* = 1.0
    ]
  )
]

#slide[
  == Step 1: After Merging x₁ and x₂

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #align(center)[#fig-w("hierarchical_step_1.pdf", 95%)]
    ],
    [
      *Current clusters* (5):
      - $\{bold(x)_1, bold(x)_2\}$
      - $\{bold(x)_3\}$, $\{bold(x)_4\}$, $\{bold(x)_5\}$, $\{bold(x)_6\}$

      *New distance* (avg linkage):

      $d(\{x_1, x_2\}, \{x_3\}) = (1.0 + 1.41) / 2 = 1.21$

      *Action*: Merge x₄ and x₅

      *Dendrogram height* = 1.0
    ]
  )
]

#slide[
  == Steps 2-4: Continuing Merges

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #align(center)[#fig-w("hierarchical_step_2.pdf", 95%)]
    ],
    [
      *Step 2*: Merge $\{x_1, x_2\}$ with $x_3$
      - Height = 1.21
      - 3 clusters left

      *Step 3*: Merge $\{x_4, x_5\}$ with $x_6$
      - Height = 1.27
      - 2 clusters left

      *Step 4*: Final merge
      - Height ≈ 5.2
      - *Large jump* → 2 natural clusters!
    ]
  )
]

#slide[
  == Understanding the Dendrogram Y-Axis

  *Y-axis* = Distance at which clusters merge

  #pause

  *Low values* (0-1.5): Points that are close → merge early
  - x₁ and x₂ merge at height 1.0 (very close)
  - x₄ and x₅ merge at height 1.0 (very close)

  #pause

  *Medium values* (1.5-3): Nearby clusters merge
  - {x₁, x₂} merges with x₃ at height 1.21

  #pause

  *High values* (>4): Distant clusters merge
  - Left group {x₁, x₂, x₃} merges with right group {x₄, x₅, x₆} at height 5.2
  - *Large jump* → suggests 2 natural clusters!

  #pause

  *Horizontal cut* at height h → Choose K by counting branches below the cut
]

#slide[
  == Linkage Criteria

  *Problem*: Distance between two *clusters*?

  #pause

  - *Single*: $min_(bold(x) in C_i, bold(y) in C_j) ||bold(x) - bold(y)||$ (closest points)

  - *Complete*: $max_(bold(x) in C_i, bold(y) in C_j) ||bold(x) - bold(y)||$ (farthest points)

  - *Average*: $1/(|C_i| |C_j|) sum_(bold(x) in C_i) sum_(bold(y) in C_j) ||bold(x) - bold(y)||$ (all pairs)

  #pause

  *Choice matters*! Different linkages → different dendrograms
]

#slide[
  == Linkage Comparison

  #align(center)[#fig("linkage_comparison.pdf")]

  *Single*: chains, *Complete*: compact, *Average*: compromise
]

#slide[
  == Hierarchical vs K-Means

  #table(
    columns: (auto, 1.2fr, 1.2fr),
    stroke: 0.5pt,
    inset: 0.6em,
    align: (left, left, left),
    [*Aspect*], [*K-Means*], [*Hierarchical*],
    table.hline(),
    [K specified?], [Yes], [No (from dendrogram)],
    [Shape], [Spherical], [More flexible],
    [Scalability], [Fast: $O(n K d T)$], [Slow: $O(n^2 log n)$],
    [Deterministic?], [No], [Yes],
    [Best for], [Large data, K known], [Small data, explore K],
  )

  #pause

  *Recommendation*: $n < 10,000$ + hierarchy → Hierarchical, else K-Means
]

// ==============================================================================
// CLOSING
// ==============================================================================

#slide[
  == Summary: Key Takeaways

  *1. K-Means* is the workhorse:
  - Minimize within-cluster sum of squares
  - E-step + M-step, converges to local minimum
  - Always use K-Means++ and standardize!

  #pause

  *2. Practical workflow*:
  - Visualize (PCA/t-SNE), standardize, try K-Means++
  - Choose K via elbow/silhouette
  - If fails: diagnose (scaling? non-convex?)

  #pause

  *3. Alternatives*:
  - *Hierarchical*: No K, builds tree, slow
  - *DBSCAN*: Non-convex, finds outliers
  - *GMM*: Soft assignments, elliptical
]

#slide[
  #align(center + horizon)[
    #text(size: 36pt, weight: "bold")[
      Questions?
    ]

    #v(2em)

    Nipun Batra

    IIT Gandhinagar

    #v(1em)

    nipun.batra\@iitgn.ac.in

    #v(1em)

    #text(size: 14pt, fill: gray)[
      All visualizations generated with Python

      Code available in course repository
    ]
  ]
]
