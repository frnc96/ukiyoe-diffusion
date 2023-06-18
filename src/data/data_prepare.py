from torch.utils.data import Dataset
from torchvision import transforms
import configuration as config
from PIL import Image
import random
import os


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_RESIZE),                  # Resize the image to a specific size
            transforms.ToTensor(),                                  # Convert the image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
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

        if self.transform is not None:
            padded_image = self.transform(padded_image)

        print(padded_image.shape)

        return padded_image.to(config.DEVICE)

    def get_random(self):
        return self.__getitem__(random.randint(0, self.__len__()))
