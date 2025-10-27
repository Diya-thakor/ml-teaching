# 🎉 New Features Added!

## 1. Expandable Cluster Views ✨

**What changed**: In Tab 1 (Clustering Visualization), cluster summaries now have expandable sections.

**Before**: Showed first 3 images, then "... and X more" text
**Now**: Shows first 3 images, then **"➕ Show X more images"** button that expands to reveal ALL images in that cluster!

**Try it**:
- Go to Tab 1: Clustering Visualization
- Scroll down to "Cluster Summary"
- Click on any "➕ Show X more images" button
- See all images in that cluster!

---

## 2. Within-Image Clustering 🖼️ (NEW TAB!)

**The most exciting feature!** Tab 4 lets you cluster regions **within a single image**.

### What It Does

Takes one image and:
1. Extracts overlapping patches (small regions)
2. Gets DINOv3 embeddings for each patch
3. Clusters the patches using K-Means
4. Shows you which parts of the image are similar to each other!

### Visual Output

**Side-by-side comparison**:
- **Left**: Original image
- **Right**: Same image with colored overlay showing clusters
  - Each color = one cluster
  - Same-colored regions = semantically similar patches

### Controls

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Patch Size | 32-128px | Size of extracted patches (smaller = finer details) |
| Stride | 16-64px | Distance between patches (smaller = more overlap) |
| Clusters | 2-10 | How many groups to create |

### What You Get

1. **Visual overlay**: See clustered regions with transparent colored rectangles
2. **Statistics**: How many patches per cluster (with percentages)
3. **Example patches**: Click to expand and see 10 example patches from each cluster
4. **Positions**: Each patch shows its location in the original image

### Amazing Use Cases

#### 🍕 Food Segmentation
- Select a pizza image
- See toppings, crust, and background cluster separately
- Different ingredients might form different clusters!

#### 🏔️ Landscape Parsing
- Select a mountain/beach image
- Watch sky, land, and water form distinct clusters
- Trees vs rocks vs water vs sky

#### 🐱 Animal Feature Detection
- Select a cat or dog
- Face, body, and background often cluster separately
- Eyes, nose, fur patterns might group together

#### 🏛️ Architecture Analysis
- Select a building
- Windows, walls, doors, and decorative elements cluster
- Repeating patterns (like windows) all get same cluster!

#### 🌸 Flower/Nature Details
- Petals vs leaves vs background
- Different colored petals might cluster differently

### Tips for Best Results

**For coarse segmentation** (sky vs land vs object):
- Large patch size (96-128px)
- Large stride (48-64px)
- Fewer clusters (2-4)

**For fine details** (texture analysis):
- Small patch size (32-56px)
- Small stride (16-28px)
- More clusters (5-10)

**For general exploration**:
- Medium patch size (56px) ← **Default**
- Medium stride (28px) ← **Default**
- Medium clusters (4) ← **Default**

### Example Experiments

1. **Pizza Analysis**
   - Image: pizza_1.jpg or pizza_2.jpg
   - Clusters: 4-5
   - Expected: Cheese, toppings, crust, background

2. **Beach Scene**
   - Image: beach_1.jpg or beach_2.jpg
   - Clusters: 3-4
   - Expected: Sky, sand, water, maybe people/objects

3. **Cat Portrait**
   - Image: cat_1.jpg, cat_2.jpg, cat_3.jpg
   - Clusters: 3-5
   - Expected: Face/eyes, body/fur, background

4. **Mountain Landscape**
   - Image: mountain_1.jpg or mountain_2.jpg
   - Clusters: 3-4
   - Expected: Sky, mountains, foreground, maybe trees

5. **Building Architecture**
   - Image: building_1.jpg or building_3.jpg
   - Clusters: 4-6
   - Expected: Windows, walls, sky, decorative elements

## Technical Details

### How It Works

1. **Sliding Window**: Moves across the image extracting patches
   - With stride < patch_size, patches overlap
   - Each patch is a square region of the image

2. **Feature Extraction**: Each patch gets processed by DINOv3
   - 224×224 resize (standard for the model)
   - 768-dimensional embedding vector

3. **Clustering**: K-Means groups similar patches
   - Patches with similar content get same label
   - Semantically similar regions cluster together

4. **Visualization**: Draw colored rectangles
   - Each patch position drawn on original image
   - Color = cluster assignment
   - Alpha=0.4 for transparency

### Performance Notes

- **Small images** (400×400): ~10-50 patches, very fast (<5 sec)
- **Medium images** (800×600): ~100-300 patches, fast (~15 sec)
- **Large images** (1200×900): ~500+ patches, slower (~30-60 sec)

### Why This Is Cool

**Self-supervised segmentation!**
- No training on segmentation data
- No manual labels needed
- Model learned these patterns from unlabeled images
- Works on ANY image, any domain

**Semantic understanding**:
- Not just color-based (would be trivial)
- DINOv3 clusters by **meaning/content**
- "Sky" patches cluster together even if different colors
- "Face" patches cluster together regardless of lighting

**Interactive exploration**:
- Adjust parameters in real-time
- See immediately how it affects clustering
- Educational tool for understanding vision models

## Quick Start Guide

### Step 1: Open Tab 4
Click on "🖼️ Within-Image Clustering" tab

### Step 2: Select an Image
Choose any image from the dropdown (try pizza or beach first!)

### Step 3: Use Defaults
The default parameters (patch=56, stride=28, clusters=4) work well!

### Step 4: Observe
Watch the colored overlay - what clustered together?

### Step 5: Explore
Click "Show example patches" for each cluster to see what's in it

### Step 6: Experiment
Try changing:
- Number of clusters (more = finer segmentation)
- Patch size (smaller = more detail)
- Different images (animals, food, landscapes)

## Summary

**Two major improvements**:

1. ✅ **Expandable cluster views**: Click to see all images in a cluster
2. ✅ **Within-image clustering**: Analyze regions within a single image

**Sample images**: 45 ready-to-use images with diverse content

**Educational value**: Perfect for teaching:
- Self-supervised learning
- Image segmentation without labels
- Semantic similarity
- Feature representations
- Clustering algorithms

**Try it now**: The app is running and auto-reloaded with these features!

🌐 **URL**: http://localhost:8507
