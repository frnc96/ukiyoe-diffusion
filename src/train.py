import torch
from tqdm import tqdm
import src.config as config
import src.helpers as helper
from torch.utils.data import DataLoader
from src.data.data_prepare import ImageDataset
from src.containers.diffusion_container import DiffusionContainer

helper.create_dir_if_non_existent(config.MODELS_PATH)

# Create an instance of the ImageDataset and pass the image directory and transform
dataset = ImageDataset(image_dir=config.TRAIN_DATASET_PATH)

# Create a DataLoader to handle batching and parallel data loading
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Create container instance where the model lives
container = DiffusionContainer()

for epoch in tqdm(range(config.EPOCHS, 1), desc="Epochs", position=0, leave=False, colour='green'):
    loss_hist = []

    for images_batch in tqdm(data_loader, desc="Batch", position=1, leave=False, colour='blue'):
        # Forward pass
        loss_val = container.batch_train(images_batch)

        # Record the batch loss
        loss_hist.append(loss_val)

    # Save model
    if epoch % 1 == 0:
        torch.save(container.model.state_dict(), f"{config.MODELS_PATH}/epoch-{epoch}.pth")

    # Print progress
    print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Loss: {loss_hist[-1]}")
