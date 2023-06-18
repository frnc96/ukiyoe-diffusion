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

for epoch in tqdm(range(config.EPOCHS), desc="Epochs", position=0, leave=False, colour='green'):
    loss_hist = []

    for images_batch in tqdm(data_loader, desc="Batch", position=1, leave=False, colour='blue'):
        # Forward pass
        loss_val = container.batch_train(images_batch)

        # Record the batch loss
        loss_hist.append(loss_val)

    # Save model
    if epoch % 1 == 0:
        torch.save(container.model.state_dict(), f"{config.MODELS_PATH}/epoch-{epoch}.pth")

    # Sampling
    output_tensor_list = container.sample(nr=20)
    min_image_mse = min(output_tensor_list, key=lambda x: x["mse"])['mse']

    mse_list = [d['mse'] for d in output_tensor_list]
    avg_image_mse = sum(mse_list) / len(mse_list)

    progress = {
        'epoch': epoch,
        'last_loss': loss_hist[-1],
        'avg_loss': sum(loss_hist) / len(loss_hist),
        'min-image-mse': min_image_mse,
        'avg-image-mse': avg_image_mse,
    }

    # Print progress
    tqdm.write(str(progress))
