"""
Download sample images for simple clustering demonstration
"""

import requests
from PIL import Image
import io
from pathlib import Path

# Create samples directory
samples_dir = Path("sample_images")
samples_dir.mkdir(exist_ok=True)

# Curated images with clear regions/colors for clustering
image_urls = {
    # Images with distinct color regions
    "sunset_beach.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&h=400&fit=crop",
    "colorful_houses.jpg": "https://images.unsplash.com/photo-1555881400-74d7acaacd8b?w=600&h=400&fit=crop",
    "hot_air_balloons.jpg": "https://images.unsplash.com/photo-1498550744921-75f79806b163?w=600&h=400&fit=crop",
    "autumn_forest.jpg": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=400&fit=crop",
    "colorful_birds.jpg": "https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=600&h=400&fit=crop",
    "tulip_field.jpg": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=600&h=400&fit=crop",
    "mountain_lake.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&h=400&fit=crop",
    "city_skyline.jpg": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=600&h=400&fit=crop",
}

print("Downloading sample images for clustering demo...")
for filename, url in image_urls.items():
    filepath = samples_dir / filename
    if filepath.exists():
        print(f"✓ {filename} already exists")
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img.save(filepath, 'JPEG', quality=90)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print(f"\n✓ Done! Saved {len(list(samples_dir.glob('*.jpg')))} images to {samples_dir}")
