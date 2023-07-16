from torch.utils.data import Dataset
import configuration as config
import random
import torch
import os


class ImageDataset(Dataset):
    def __init__(self, image_tensor_dir):
        self.image_tensor_dir = image_tensor_dir
        self.image_tensor_files = os.listdir(image_tensor_dir)

    def __len__(self):
        return len(self.image_tensor_files)

    def __getitem__(self, index):
        image_tensor_path = os.path.join(self.image_tensor_dir, self.image_tensor_files[index])
        
        return torch.load(image_tensor_path).to(config.DEVICE)

    def get_random(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))
