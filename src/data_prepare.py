from torchvision import transforms
import configuration as config
from tqdm import tqdm
import helpers as helper
from PIL import Image
import torch
import os


assert os.path.exists(config.TRAIN_DATASET_PATH)
helper.create_dir_if_non_existent(config.TENSOR_DATASET_PATH)

transform = transforms.Compose([
    transforms.Resize(config.IMAGE_RESIZE),                 # Resize the image to a specific size
    transforms.ToTensor(),                                  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensor
])

for image_file in tqdm(os.listdir(config.TRAIN_DATASET_PATH), desc="Preparing Images"):
    image_path = os.path.join(config.TRAIN_DATASET_PATH, image_file)
    original_image = Image.open(image_path)

    # Calculate the new dimensions while maintaining the aspect ratio
    width, height = original_image.size
    target_width, target_height = config.IMAGE_RESIZE
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if target_aspect_ratio > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    resized_image = original_image.resize((new_width, new_height), resample=Image.BICUBIC)

    # Create a new blank image with the target size
    padded_image = Image.new("RGB", config.IMAGE_RESIZE, (255, 255, 255))

    # Calculate the padding dimensions
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Paste the resized image onto the padded image
    padded_image.paste(resized_image, (left, top, right, bottom))

    # Transform to tensor and normalize
    if transform is not None:
        padded_image_tensor = transform(padded_image)

    # Save tensor as file
    tensor_path = os.path.join(
        config.TENSOR_DATASET_PATH,
        image_file.replace('.jpg', '.pt')
    )
    torch.save(padded_image_tensor, tensor_path)

    print(f"Saved {tensor_path}")
