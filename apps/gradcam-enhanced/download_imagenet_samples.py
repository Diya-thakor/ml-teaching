"""
Download a curated set of ImageNet images for Grad-CAM visualization
"""
import urllib.request
from pathlib import Path
from PIL import Image
import io
import json

# Curated ImageNet samples: 30 images across 6 categories
IMAGENET_SAMPLES = {
    # Category 1: Cats (5 images)
    "tabby_cat_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/600px-Cat03.jpg", "class_idx": 281, "class_name": "tabby cat"},
    "tabby_cat_2": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/600px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg", "class_idx": 281, "class_name": "tabby cat"},
    "persian_cat_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/White_Persian_Cat.jpg/600px-White_Persian_Cat.jpg", "class_idx": 283, "class_name": "persian cat"},
    "tiger_cat_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Feral_cat_Virginia_crop.jpg/600px-Feral_cat_Virginia_crop.jpg", "class_idx": 282, "class_name": "tiger cat"},
    "siamese_cat_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/600px-Siam_lilacpoint.jpg", "class_idx": 284, "class_name": "siamese cat"},

    # Category 2: Dogs (5 images)
    "golden_retriever_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg/600px-Golden_Retriever_Carlos_%2810581910556%29.jpg", "class_idx": 207, "class_name": "golden retriever"},
    "golden_retriever_2": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/600px-Golden_Retriever_Dukedestiny01_drvd.jpg", "class_idx": 207, "class_name": "golden retriever"},
    "german_shepherd_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/German_Shepherd_-_DSC_0346_%2810096362833%29.jpg/600px-German_Shepherd_-_DSC_0346_%2810096362833%29.jpg", "class_idx": 235, "class_name": "german shepherd"},
    "beagle_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Beagle_600.jpg/600px-Beagle_600.jpg", "class_idx": 162, "class_name": "beagle"},
    "corgi_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Welchcorgipembroke.JPG/600px-Welchcorgipembroke.JPG", "class_idx": 263, "class_name": "pembroke corgi"},

    # Category 3: Birds (5 images)
    "robin_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Erithacus_rubecula_with_cocked_head.jpg/600px-Erithacus_rubecula_with_cocked_head.jpg", "class_idx": 15, "class_name": "robin"},
    "goldfinch_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Carduelis-carduelis-001.jpg/600px-Carduelis-carduelis-001.jpg", "class_idx": 11, "class_name": "goldfinch"},
    "jay_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Garrulus_glandarius_1_Luc_Viatour.jpg/600px-Garrulus_glandarius_1_Luc_Viatour.jpg", "class_idx": 17, "class_name": "jay"},
    "magpie_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Pica_pica_-_Compans_Caffarelli_-_2012-03-16_-_2.jpg/600px-Pica_pica_-_Compans_Caffarelli_-_2012-03-16_-_2.jpg", "class_idx": 18, "class_name": "magpie"},
    "chickadee_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Parus_major_3_Luc_Viatour.jpg/600px-Parus_major_3_Luc_Viatour.jpg", "class_idx": 19, "class_name": "chickadee"},

    # Category 4: Vehicles (5 images)
    "sports_car_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/2018_Ferrari_488_GTB_F1_Edition_3.9.jpg/600px-2018_Ferrari_488_GTB_F1_Edition_3.9.jpg", "class_idx": 817, "class_name": "sports car"},
    "sports_car_2": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/McLaren_570S_at_Goodwood_2015_%2818515451093%29.jpg/600px-McLaren_570S_at_Goodwood_2015_%2818515451093%29.jpg", "class_idx": 817, "class_name": "sports car"},
    "convertible_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/2013_Mazda_MX-5_Roadster_Coupe_%282.0%29.jpg/600px-2013_Mazda_MX-5_Roadster_Coupe_%282.0%29.jpg", "class_idx": 511, "class_name": "convertible"},
    "school_bus_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/2007_IC_CE_200_Ft._Thomas.jpg/600px-2007_IC_CE_200_Ft._Thomas.jpg", "class_idx": 779, "class_name": "school bus"},
    "fire_engine_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Rosenbauer_Panther_airport_fire_truck.JPG/600px-Rosenbauer_Panther_airport_fire_truck.JPG", "class_idx": 555, "class_name": "fire engine"},

    # Category 5: Food (5 images)
    "pizza_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/600px-Eq_it-na_pizza-margherita_sep2005_sml.jpg", "class_idx": 963, "class_name": "pizza"},
    "pizza_2": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Supreme_pizza.jpg/600px-Supreme_pizza.jpg", "class_idx": 963, "class_name": "pizza"},
    "strawberry_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/PerfectStrawberry.jpg/600px-PerfectStrawberry.jpg", "class_idx": 949, "class_name": "strawberry"},
    "banana_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/600px-Banana-Single.jpg", "class_idx": 954, "class_name": "banana"},
    "ice_cream_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Ice_cream_with_whipped_cream%2C_chocolate_syrup%2C_and_a_wafer_%28cropped%29.jpg/600px-Ice_cream_with_whipped_cream%2C_chocolate_syrup%2C_and_a_wafer_%28cropped%29.jpg", "class_idx": 928, "class_name": "ice cream"},

    # Category 6: Animals (5 images)
    "elephant_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/600px-African_Bush_Elephant.jpg", "class_idx": 386, "class_name": "african elephant"},
    "zebra_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Plains_Zebra_Equus_quagga.jpg/600px-Plains_Zebra_Equus_quagga.jpg", "class_idx": 340, "class_name": "zebra"},
    "tiger_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Walking_tiger_female.jpg/600px-Walking_tiger_female.jpg", "class_idx": 292, "class_name": "tiger"},
    "panda_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/600px-Grosser_Panda.JPG", "class_idx": 388, "class_name": "giant panda"},
    "bear_1": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/2010-kodiak-bear-1.jpg/600px-2010-kodiak-bear-1.jpg", "class_idx": 294, "class_name": "brown bear"},

    # Category 7: SPURIOUS CORRELATIONS - Background Bias Examples
    # These demonstrate when models rely on background/context rather than object features
    # This is a critical concept in ML explainability!

    # Husky examples - models often confuse with wolves based on snow/background
    "spurious_husky_indoor": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Siberian-husky.jpg/600px-Siberian-husky.jpg", "class_idx": 250, "class_name": "SPURIOUS: siberian husky"},

    # Animals with unusual backgrounds - test if model relies on habitat cues
    "spurious_horse_beach": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Nokota_Horses_cropped.jpg/600px-Nokota_Horses_cropped.jpg", "class_idx": 340, "class_name": "SPURIOUS: sorrel horse"},
    "spurious_cow_field": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Cow_female_black_white.jpg/600px-Cow_female_black_white.jpg", "class_idx": 345, "class_name": "SPURIOUS: ox"},

    # Tennis ball - models may focus on person/court instead of ball
    "spurious_tennis_ball": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Tennis_Balls.jpg/600px-Tennis_Balls.jpg", "class_idx": 852, "class_name": "SPURIOUS: tennis ball"},

    # Seashore/boat - models may confuse based on water context
    "spurious_sailboat": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Kookaburra_at_sunset.jpg/600px-Kookaburra_at_sunset.jpg", "class_idx": 724, "class_name": "SPURIOUS: catamaran"},

    # Bird on unusual surface - models trained on birds in trees/sky
    "spurious_robin_ground": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Erithacus_rubecula_with_cocked_head.jpg/600px-Erithacus_rubecula_with_cocked_head.jpg", "class_idx": 15, "class_name": "SPURIOUS: robin"},
}

def download_dataset(output_dir="imagenet_samples"):
    """Download curated ImageNet samples"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Downloading {len(IMAGENET_SAMPLES)} ImageNet samples to {output_path}...")
    print("Categories: Cats, Dogs, Birds, Vehicles, Food, Animals, Spurious Correlations\n")

    metadata = {}
    success_count = 0

    for name, info in IMAGENET_SAMPLES.items():
        output_file = output_path / f"{name}.jpg"

        if output_file.exists():
            print(f"✓ {name:25s} already exists")
            metadata[name] = {"class_idx": info["class_idx"], "class_name": info["class_name"]}
            success_count += 1
            continue

        try:
            print(f"⬇ {name:25s} downloading...", end=" ")

            # Add user agent to avoid 403 errors
            req = urllib.request.Request(info["url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                img_data = response.read()
                img = Image.open(io.BytesIO(img_data)).convert('RGB')

                # Resize to 224x224 (standard ImageNet size)
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img.save(output_file, "JPEG", quality=95)

            metadata[name] = {"class_idx": info["class_idx"], "class_name": info["class_name"]}
            print("✓")
            success_count += 1

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Download complete! {success_count}/{len(IMAGENET_SAMPLES)} images successfully downloaded")
    print(f"Dataset saved to: {output_path.absolute()}")
    print(f"Metadata saved to: {metadata_file.absolute()}")

    return output_path

if __name__ == "__main__":
    download_dataset()
