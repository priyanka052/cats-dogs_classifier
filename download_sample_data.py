import os
import torch
from torchvision.utils import save_image

def generate_dummy_images():
    # Directories
    dog_dir = "dataset/train/dog"
    cat_dir = "dataset/train/cat"
    
    # Ensure directories exist
    os.makedirs(dog_dir, exist_ok=True)
    os.makedirs(cat_dir, exist_ok=True)
    
    print("Generating dummy images...")
    
    # Generate 4 dummy images (2 per class)
    for category in ["dog", "cat"]:
        folder = dog_dir if category == "dog" else cat_dir
        for i in range(2):
            filename = os.path.join(folder, f"{category}_{i}.jpg")
            # Create a random RGB image (3, 128, 128)
            img = torch.rand(3, 128, 128)
            save_image(img, filename)
            print(f"Generated {filename}")

    print("\nGeneration complete! You can now run 'python train.py'.")

if __name__ == "__main__":
    generate_dummy_images()
