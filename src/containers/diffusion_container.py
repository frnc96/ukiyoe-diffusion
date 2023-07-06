import os
import uuid
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import configuration as config
from diffusers import UNet2DModel
from accelerate import Accelerator
from diffusers import DDPMScheduler
from data.data_loader import ImageDataset
from skimage.metrics import mean_squared_error


def evaluate_sample(ground_truth_tensor, generated_tensor):
    # Convert tensors to numpy arrays
    output_array = generated_tensor.cpu().squeeze().numpy()
    ground_truth_array = ground_truth_tensor.cpu().squeeze().numpy()

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(ground_truth_array, output_array)

    # # Calculate Structural Similarity Index (SSIM)
    # ssim = structural_similarity(
    #     ground_truth_array,
    #     output_array,
    #     multichannel=True,
    #     channel_axis=0,
    # )

    return mse


class DiffusionContainer:

    def __init__(self, load_pretrained=False):
        self.model = UNet2DModel(
            sample_size=config.IMAGE_RESIZE[0],       # the target image resolution
            in_channels=3,                            # the number of input channels, 3 for RGB images
            out_channels=3,                           # the number of output channels
            layers_per_block=2,                       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 32, 64, 64),      # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(config.DEVICE)
        self.model_name = str(uuid.uuid4())

        if load_pretrained:
            models_path_list = os.listdir(config.MODELS_PATH)
            assert len(models_path_list) > 0

            self.model.load_state_dict(
                torch.load(
                    os.path.join(config.MODELS_PATH, models_path_list[-1])
                )
            )
            self.model_name = models_path_list[-1].replace('.pth', '')

        self.error_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.TRAIN_TIME_STEPS)
        self.accelerator = Accelerator()

    def batch_train(self, images_batch):
        # Sample noise to add to the images
        noise = torch.randn_like(images_batch).to(config.DEVICE)
        bs = images_batch.shape[0]

        # Sample a random timestep for each image
        time_steps = torch.randint(
            0, config.TRAIN_TIME_STEPS, (bs,), device=config.DEVICE
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images_batch, noise, time_steps)

        with self.accelerator.accumulate(self.model):
            # Predict the noise residual
            noise_pred = self.model(noisy_images, time_steps, return_dict=False)[0]
            loss = self.error_fn(noise_pred, noise)
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            # lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.item()

    def sample(self, nr=10):
        self.model.eval()

        image_tensors = []
        for _ in tqdm(range(nr), desc="Sampling", position=1, leave=False, colour='blue'):
            # Create an image composed of noise
            input_tensor = torch.randn(3, config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]).to(config.DEVICE)

            # Generate the output image using the pre-trained model
            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            # Get random ground truth tensor from dataset
            ground_truth_tensor = ImageDataset(image_tensor_dir=config.TENSOR_DATASET_PATH).get_random()

            mse = evaluate_sample(ground_truth_tensor, output_tensor)

            image_tensors.append({
                'tensor': output_tensor,
                'mse': mse,
            })

        return image_tensors
