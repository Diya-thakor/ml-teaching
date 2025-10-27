"""
Create sample images for the DINOv3 clustering demo
Downloads diverse images from publicly available sources
"""

import requests
from PIL import Image
import io
from pathlib import Path

# Create samples directory
samples_dir = Path("sample_images")
samples_dir.mkdir(exist_ok=True)

# Unsplash random image URLs (public, free to use)
# Using specific search terms to get diverse categories
image_urls = {
    # Animals
    "cat_1.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop",
    "cat_2.jpg": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=400&fit=crop",
    "cat_3.jpg": "https://images.unsplash.com/photo-1573865526739-10c1dd7aa8b0?w=400&h=400&fit=crop",
    "dog_1.jpg": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&h=400&fit=crop",
    "dog_2.jpg": "https://images.unsplash.com/photo-1560807707-8cc77767d783?w=400&h=400&fit=crop",
    "dog_3.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400&h=400&fit=crop",
    "bird_1.jpg": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop",
    "bird_2.jpg": "https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=400&h=400&fit=crop",

    # Vehicles
    "car_1.jpg": "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=400&h=400&fit=crop",
    "car_2.jpg": "https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?w=400&h=400&fit=crop",
    "car_3.jpg": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=400&h=400&fit=crop",
    "bicycle_1.jpg": "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=400&h=400&fit=crop",
    "bicycle_2.jpg": "https://images.unsplash.com/photo-1532298229144-0ec0c57515c7?w=400&h=400&fit=crop",

    # Nature/Landscapes
    "mountain_1.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
    "mountain_2.jpg": "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=400&h=400&fit=crop",
    "beach_1.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400&h=400&fit=crop",
    "beach_2.jpg": "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400&h=400&fit=crop",
    "forest_1.jpg": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400&h=400&fit=crop",
    "forest_2.jpg": "https://images.unsplash.com/photo-1511497584788-876760111969?w=400&h=400&fit=crop",

    # Fruits - Many!
    "apple_1.jpg": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=400&h=400&fit=crop",
    "apple_2.jpg": "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=400&h=400&fit=crop",
    "apple_3.jpg": "https://images.unsplash.com/photo-1619546813926-a78fa6372cd2?w=400&h=400&fit=crop",
    "banana_1.jpg": "https://images.unsplash.com/photo-1603833665858-e61d17a86224?w=400&h=400&fit=crop",
    "banana_2.jpg": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400&h=400&fit=crop",
    "banana_3.jpg": "https://images.unsplash.com/photo-1587132137056-bfbf0166836e?w=400&h=400&fit=crop",
    "orange_1.jpg": "https://images.unsplash.com/photo-1580052614034-c55d20bfee3b?w=400&h=400&fit=crop",
    "orange_2.jpg": "https://images.unsplash.com/photo-1582979512210-99b6a53386f9?w=400&h=400&fit=crop",
    "orange_3.jpg": "https://images.unsplash.com/photo-1547514701-42782101795e?w=400&h=400&fit=crop",
    "strawberry_1.jpg": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=400&h=400&fit=crop",
    "strawberry_2.jpg": "https://images.unsplash.com/photo-1543528176-61b239494933?w=400&h=400&fit=crop",
    "strawberry_3.jpg": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=400&h=400&fit=crop",
    "grape_1.jpg": "https://images.unsplash.com/photo-1599819177153-1d7dfb5e4e8e?w=400&h=400&fit=crop",
    "grape_2.jpg": "https://images.unsplash.com/photo-1423483641154-5411ec9c0ddf?w=400&h=400&fit=crop",
    "watermelon_1.jpg": "https://images.unsplash.com/photo-1587049352846-4a222e784eaf?w=400&h=400&fit=crop",
    "watermelon_2.jpg": "https://images.unsplash.com/photo-1582281298055-e25b95d33d6b?w=400&h=400&fit=crop",
    "pineapple_1.jpg": "https://images.unsplash.com/photo-1550828520-4cb496926fc9?w=400&h=400&fit=crop",
    "pineapple_2.jpg": "https://images.unsplash.com/photo-1587334206515-81f8b5f50f44?w=400&h=400&fit=crop",
    "mango_1.jpg": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=400&h=400&fit=crop",
    "mango_2.jpg": "https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?w=400&h=400&fit=crop",

    # More Food
    "pizza_1.jpg": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400&h=400&fit=crop",
    "pizza_2.jpg": "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=400&h=400&fit=crop",
    "burger_1.jpg": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=400&fit=crop",
    "burger_2.jpg": "https://images.unsplash.com/photo-1550547660-d9450f859349?w=400&h=400&fit=crop",
    "sushi_1.jpg": "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400&h=400&fit=crop",
    "sushi_2.jpg": "https://images.unsplash.com/photo-1564489563601-c53cfc451e93?w=400&h=400&fit=crop",

    # Buildings/Architecture
    "building_1.jpg": "https://images.unsplash.com/photo-1486718448742-163732cd1544?w=400&h=400&fit=crop",
    "building_2.jpg": "https://images.unsplash.com/photo-1554909846-43072ae50ede?w=400&h=400&fit=crop",
    "building_3.jpg": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=400&h=400&fit=crop",

    # Flowers
    "flower_1.jpg": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400&h=400&fit=crop",
    "flower_2.jpg": "https://images.unsplash.com/photo-1508610048659-a06b669e3321?w=400&h=400&fit=crop",
    "flower_3.jpg": "https://images.unsplash.com/photo-1524386416438-98b9b2d4b433?w=400&h=400&fit=crop",
}

print("Downloading sample images...")
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
        img.save(filepath, 'JPEG', quality=85)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print(f"\n✓ Done! Saved {len(list(samples_dir.glob('*.jpg')))} images to {samples_dir}")
