import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json

st.set_page_config(page_title="Grad-CAM: High-Res Images", layout="wide")
plt.rcParams['figure.dpi'] = 300

st.title("Grad-CAM Visualization on High-Resolution Images")

st.markdown("""
Explore how pretrained neural networks focus on different regions of **224×224 RGB images**.
This curated dataset contains 24 ImageNet images, including a special **⚠️ Spurious Correlations**
category that demonstrates when models focus on background/context instead of the actual object.
""")

@st.cache_data
def load_imagenet_classes():
    """Load ImageNet class names"""
    import urllib.request
    import json
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except:
        # Fallback to numeric labels
        return [f"class_{i}" for i in range(1000)]

# Model configuration
MODELS = {
    "ResNet18": {"constructor": models.resnet18, "target_layer": "layer4"},
    "ResNet50": {"constructor": models.resnet50, "target_layer": "layer4"},
}

@st.cache_data
def load_dataset():
    """Load the curated ImageNet samples"""
    dataset_path = Path("imagenet_samples")

    if not dataset_path.exists():
        st.error(f"Dataset not found at {dataset_path.absolute()}. Please run download_imagenet_samples.py first.")
        st.stop()

    # Load metadata
    metadata_file = dataset_path / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Load images
    images = []
    for name, info in metadata.items():
        # Try both JPG and PNG extensions
        img_path_jpg = dataset_path / f"{name}.jpg"
        img_path_png = dataset_path / f"{name}.png"

        img_path = None
        if img_path_jpg.exists():
            img_path = img_path_jpg
        elif img_path_png.exists():
            img_path = img_path_png

        if img_path:
            img = Image.open(img_path).convert('RGB')
            images.append({
                "name": name,
                "image": img,
                "class_idx": info["class_idx"],
                "class_name": info["class_name"]
            })

    return images

@st.cache_resource
def load_model(model_name):
    """Load pretrained ImageNet model"""
    model_info = MODELS[model_name]
    model = model_info["constructor"](weights='DEFAULT')
    model.eval()
    return model, model_info["target_layer"]

def get_target_layer(model, layer_name):
    """Get the target layer from model"""
    return getattr(model, layer_name)

def compute_gradcam(model, img_tensor, target_layer, target_class=None):
    """
    Compute Grad-CAM for a given image

    Args:
        model: The neural network model
        img_tensor: Normalized input image tensor (C, H, W)
        target_layer: Layer to compute Grad-CAM on
        target_class: Class to compute gradients for (None = predicted class)

    Returns:
        cam_normalized: Normalized CAM heatmap (H, W)
        predicted_class: The predicted class index
        confidence: Confidence score
    """
    features = []

    def hook_fn(module, input, output):
        features.append(output)

    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    output = model(img_tensor.unsqueeze(0))
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class = probabilities.max(1)

    # Use predicted class if target not specified
    if target_class is None:
        target_class = predicted_class.item()

    # Enable gradient retention
    features[0].retain_grad()

    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()

    # Get gradients and feature maps
    gradients = features[0].grad
    feature_maps = features[0].detach()

    # Global average pooling on gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted combination
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample to input size
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    cam_upsampled = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam_normalized = cam_upsampled.squeeze().cpu()
    cam_normalized = (cam_normalized - cam_normalized.min()) / (cam_normalized.max() - cam_normalized.min() + 1e-8)

    handle.remove()

    return cam_normalized.numpy(), predicted_class.item(), confidence.item()

# Load dataset and ImageNet class names
dataset = load_dataset()
imagenet_classes = load_imagenet_classes()

st.sidebar.markdown(f"""
**Dataset Info:**
- Images: {len(dataset)}
- Resolution: 224×224×3 (RGB)
- Categories: Cats, Dogs, Birds, Vehicles, Food, Animals, Spurious Examples
""")

# Model selection
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))

# Load model
model, target_layer_name = load_model(model_name)
target_layer = get_target_layer(model, target_layer_name)

st.sidebar.markdown(f"""
**Model:** {model_name}
**Target Layer:** `{target_layer_name}`
**Parameters:** {sum(p.numel() for p in model.parameters()):,}
""")

# Image selection
st.sidebar.header("Image Selection")

# Group images by category
categories = {}
spurious_category = []

for img_data in dataset:
    cat = img_data["class_name"]
    # Separate spurious correlation examples
    if cat.startswith("SPURIOUS:"):
        spurious_category.append(img_data)
    else:
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(img_data)

# Add spurious as a special category at the end
if spurious_category:
    categories["⚠️ SPURIOUS CORRELATIONS"] = spurious_category

# Display grid by category
st.sidebar.markdown("**Browse by Category:**")
selected_img = None

for cat_name in sorted(categories.keys()):
    with st.sidebar.expander(f"{cat_name.title()} ({len(categories[cat_name])} images)", expanded=(cat_name == sorted(categories.keys())[0])):
        cols = st.columns(4)
        for idx, img_data in enumerate(categories[cat_name]):
            with cols[idx % 4]:
                if st.button(f"#{dataset.index(img_data)}", key=img_data["name"]):
                    selected_img = img_data
                st.image(img_data["image"], width=60)

# Manual selection
manual_idx = st.sidebar.number_input("Or Enter Image Index", 0, len(dataset) - 1, 0)
if selected_img is None:
    selected_img = dataset[manual_idx]

# Target class selection
st.sidebar.header("Grad-CAM Target")
use_predicted = st.sidebar.checkbox("Use Predicted Class", value=True)

if not use_predicted:
    target_class = st.sidebar.number_input("Target Class Index (ImageNet)",
                                          0, 999, selected_img["class_idx"],
                                          help="Enter ImageNet class ID (0-999)")
else:
    target_class = None

# Prepare image
img_pil = selected_img["image"]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil)

# Compute Grad-CAM
with st.spinner("Computing Grad-CAM..."):
    cam, predicted_class, confidence = compute_gradcam(model, img_tensor, target_layer, target_class)

# Display results
img_idx = dataset.index(selected_img)
st.header(f"Image {img_idx}: {selected_img['name']} - {model_name}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(img_pil))
    ax.axis('off')

    # Get predicted class name
    pred_class_name = imagenet_classes[predicted_class]

    # Clean up true label (remove SPURIOUS prefix if present)
    true_label = selected_img['class_name'].replace("SPURIOUS: ", "")

    title_text = f"True: {true_label}\nPredicted: {pred_class_name} ({confidence:.1%})"
    ax.set_title(title_text, fontweight='bold', fontsize=12)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Grad-CAM Heatmap")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(img_pil))
    ax.imshow(cam, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax.axis('off')

    # Get target class name
    if target_class is None:
        target_text = pred_class_name
    else:
        target_text = imagenet_classes[target_class]
    ax.set_title(f"Attention for:\n{target_text}", fontweight='bold', fontsize=12)
    st.pyplot(fig)
    plt.close()

# Additional info
with st.expander("About This Visualization"):
    # Check if this is a spurious correlation example
    is_spurious = selected_img['class_name'].startswith("SPURIOUS:")
    spurious_note = "\n    - **⚠️ SPURIOUS EXAMPLE:** Pay attention to what the model focuses on!" if is_spurious else ""

    st.markdown(f"""
    ### Image Details
    - **File:** `{selected_img['name']}.jpg`
    - **True Class:** {true_label} (ImageNet ID: {selected_img['class_idx']})
    - **Predicted Class:** {pred_class_name} (ImageNet ID: {predicted_class})
    - **Confidence:** {confidence:.2%}{spurious_note}

    ### How Grad-CAM Works

    1. **Forward Pass:** Image goes through {model_name}, extracting feature maps from `{target_layer_name}`
    2. **Backward Pass:** Compute gradients of target class score w.r.t. feature maps
    3. **Weighting:** Average gradients spatially to get channel importance weights
    4. **Combination:** Weighted sum of feature maps, then ReLU
    5. **Visualization:** Upsample to 224×224 and overlay on original image

    **Red regions** = high importance for the prediction
    **Blue regions** = low importance

    ### Try This
    - Switch between ResNet18 and ResNet50 to compare attention patterns
    - Uncheck "Use Predicted Class" to see attention for other classes
    - Notice how the model focuses on distinctive features (cat faces, dog snouts, wheels, food textures)

    ### Spurious Correlations

    Some images demonstrate **spurious correlations** - when models make predictions based on
    background context rather than the actual object:

    - **Husky images**: Check if the model focuses on snow/background vs. the dog's features
    - **Animals in unusual settings**: Does the model rely on typical habitat cues?

    This is a known issue in deep learning where models learn shortcuts based on dataset biases.
    Grad-CAM helps identify when models are relying on spurious features!
    """)

with st.expander("Model Architecture"):
    if "ResNet" in model_name:
        st.markdown(f"""
        ### {model_name} Architecture

        ```
        Input (3, 224, 224)
            ↓
        Conv1 + BN + ReLU + MaxPool
            ↓
        Layer1 (64 channels, 56×56)
            ↓
        Layer2 (128 channels, 28×28)
            ↓
        Layer3 (256 channels, 14×14)
            ↓
        Layer4 (512 channels, 7×7)  ← Grad-CAM target
            ↓
        AdaptiveAvgPool + FC(1000)
            ↓
        Output (1000 classes)
        ```

        **Why Layer4?** It's the last convolutional layer with spatial information (7×7),
        providing the best balance between semantic understanding and spatial resolution.
        """)

# Performance comparison
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Image Resolution", "224×224")
with col2:
    st.metric("Model Size", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
with col3:
    st.metric("Prediction Confidence", f"{confidence:.1%}")
