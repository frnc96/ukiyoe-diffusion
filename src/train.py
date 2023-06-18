import torch
from tqdm import tqdm
import configuration as config
import helpers as helper
from torch.utils.data import DataLoader
from data.data_prepare import ImageDataset
from containers.diffusion_container import DiffusionContainer

helper.create_dir_if_non_existent(config.MODELS_PATH)
helper.create_dir_if_non_existent(config.WANDB_PATH)

# Create an instance of the ImageDataset and pass the image directory and transform
dataset = ImageDataset(image_dir=config.TRAIN_DATASET_PATH)

# Create a DataLoader to handle batching and parallel data loading
data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Create container instance where the model lives
container = DiffusionContainer()

# Init WANDB if needed
wandb = None
if config.USE_WEIGHTS_AND_BIASES:
    wandb = helper.init_wandb()

for epoch in tqdm(range(config.EPOCHS), desc="Epochs", position=0, leave=False, colour='green'):
    loss_hist = []

    for images_batch in tqdm(data_loader, desc="Batch", position=1, leave=False, colour='blue'):
        batch_size = images_batch.shape[0]  # Get the batch size
        print("Batch size:", batch_size)
        image_shape = images_batch.shape[1:]  # Get the shape of a single image in the batch
        print("Image shape:", image_shape)

        # Forward pass
        loss_val = container.batch_train(images_batch)

        # Record the batch loss
        loss_hist.append(loss_val)

    # Save model
    if epoch % 5 == 0:
        model_file_path = f"{config.MODELS_PATH}/model-epoch-{epoch}.pth"
        torch.save(container.model.state_dict(), model_file_path)

        if config.USE_WEIGHTS_AND_BIASES:
            wandb.log_artifact(model_file_path, name=f'model-epoch-{epoch}', type='Model')

    # Sampling
    output_tensor_list = container.sample(nr=9)
    min_image_mse = min(output_tensor_list, key=lambda x: x["mse"])['mse']

    mse_list = [d['mse'] for d in output_tensor_list]
    avg_image_mse = sum(mse_list) / len(mse_list)

    progress = {
        'last_loss': loss_hist[-1],
        'avg_loss': sum(loss_hist) / len(loss_hist),
        'min_image_mse': min_image_mse,
        'avg_image_mse': avg_image_mse,
    }

    # Log progress in wandb
    if config.USE_WEIGHTS_AND_BIASES:
        wandb.log(progress)
        wandb.log({
            "generated_images_plot": wandb.Image(
                helper.plot_sample_images(
                    output_tensor_list,
                    f"samples-epoch-{epoch}"
                )
            )
        })

    # Print progress
    tqdm.write(str(progress))
