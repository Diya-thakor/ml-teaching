"""
Generate example images for README showing spurious correlations
"""
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Load model
print("Loading ResNet18...")
model = models.resnet18(weights='DEFAULT')
model.eval()
target_layer = model.layer4

# Load ImageNet class names
import urllib.request
import json
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as response:
    imagenet_classes = json.loads(response.read().decode())

def compute_gradcam(model, img_tensor, target_layer, target_class=None):
    features = []
    def hook_fn(module, input, output):
        features.append(output)

    handle = target_layer.register_forward_hook(hook_fn)
    output = model(img_tensor.unsqueeze(0))
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class = probabilities.max(1)

    if target_class is None:
        target_class = predicted_class.item()

    features[0].retain_grad()
    model.zero_grad()
    output[0, target_class].backward()

    gradients = features[0].grad
    feature_maps = features[0].detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    H, W = img_tensor.shape[1], img_tensor.shape[2]
    cam_upsampled = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam_normalized = cam_upsampled.squeeze().cpu()
    cam_normalized = (cam_normalized - cam_normalized.min()) / (cam_normalized.max() - cam_normalized.min() + 1e-8)

    handle.remove()
    return cam_normalized.numpy(), predicted_class.item(), confidence.item()

# Process images
examples = [
    {"file": "spurious_cow_field.jpg", "true": "ox", "desc": "Cow in typical field setting"},
    {"file": "spurious_cow_ice.png", "true": "ox", "desc": "Cow on ice/snow"},
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for example in examples:
    print(f"\nProcessing {example['file']}...")

    # Load image
    img_path = Path("imagenet_samples") / example['file']
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil)

    # Compute Grad-CAM
    cam, predicted_class, confidence = compute_gradcam(model, img_tensor, target_layer)
    pred_name = imagenet_classes[predicted_class]

    print(f"  True: {example['true']}")
    print(f"  Predicted: {pred_name} (class {predicted_class}) - {confidence:.1%}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    # Original image
    axes[0].imshow(img_pil)
    axes[0].axis('off')
    axes[0].set_title(f"Original Image\nTrue: {example['true']}\nPredicted: {pred_name} ({confidence:.1%})",
                     fontweight='bold', fontsize=11)

    # Grad-CAM overlay
    axes[1].imshow(img_pil)
    axes[1].imshow(cam, cmap='jet', alpha=0.5, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title(f"Grad-CAM Heatmap\nAttention for: {pred_name}",
                     fontweight='bold', fontsize=11)

    plt.tight_layout()

    # Save
    output_name = example['file'].replace('.jpg', '').replace('.png', '') + '_gradcam.png'
    output_path = Path("readme_examples") / output_name
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"  Saved to {output_path}")
    plt.close()

print("\n✓ Done! Example images saved to readme_examples/")
