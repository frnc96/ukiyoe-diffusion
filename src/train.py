import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import src.config as config
import src.helpers as helper
from torch.utils.data import DataLoader
from src.data.data_prepare import ImageDataset
from src.networks.diffusion import DiffusionNetwork


helper.create_dir_if_non_existent(config.MODELS_PATH)

# Create an instance of the ImageDataset and pass the image directory and transform
dataset = ImageDataset(image_dir=config.TRAIN_DATASET_PATH)

# Create a DataLoader to handle batching and parallel data loading
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

diffusion_net = DiffusionNetwork().to(config.DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(diffusion_net.parameters(), lr=config.LEARNING_RATE)

print(f"Starting training on {config.DEVICE}\n")
for epoch in tqdm(range(config.EPOCHS), desc="Epochs"):
    loss_hist = []

    for images_batch in data_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = diffusion_net(images_batch)

        # Compute loss
        loss = criterion(outputs, images_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record the batch loss
        loss_hist.append(loss.item())

    if epoch % 1:
        torch.save(diffusion_net.state_dict(), f"{config.MODELS_PATH}/epoch-{epoch}.pth")

    # Print progress
    print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Loss: {loss_hist[-1]}")
