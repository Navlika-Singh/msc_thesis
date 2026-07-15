import os

def save_images(sample, idx, images_dir="/projects/u6lm/ns1324/msc_thesis/data/beavertails-v"):
    os.makedirs(images_dir, exist_ok=True)
    img = sample["image"]
    image_path = os.path.join(
        images_dir,
        f"{idx:04d}.png",
    )
    img.convert("RGB").save(image_path)
    print(f"[DEBUG] Saved image {idx} to {image_path}.")