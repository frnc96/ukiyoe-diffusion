import os
import uuid
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import configuration as config
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from data.data_loader import ImageDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers.optimization import get_cosine_schedule_with_warmup


def evaluate_sample(generated_image_tensors):
    fid = FrechetInceptionDistance(feature=64)

    dataset = ImageDataset(image_tensor_dir=config.TENSOR_DATASET_PATH)

    real_image_tensors = torch.cat([
        dataset.get_random().cpu().unsqueeze(0).to(dtype=torch.uint8) for _ in range(10)
    ])
    generated_image_tensors = torch.cat([
        t.cpu().unsqueeze(0).to(dtype=torch.uint8) for t in generated_image_tensors
    ])

    fid.update(real_image_tensors, real=True)
    fid.update(generated_image_tensors, real=False)

    fid_val = fid.compute()

    return fid_val.item()


class DiffusionContainer:

    def __init__(self, load_pretrained=False):
        self.model_name = str(uuid.uuid4())

        model = UNet2DModel(
            sample_size=config.IMAGE_RESIZE[0],       # the target image resolution
            in_channels=3,                            # the number of input channels, 3 for RGB images
            out_channels=3,                           # the number of output channels
            layers_per_block=2,                       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 32, 64, 64),      # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.LR_WARMUP_STEPS,
            num_training_steps=((len(os.listdir(config.TENSOR_DATASET_PATH)) // config.BATCH_SIZE) * config.EPOCHS),
        )
        
        model, optimizer, lr_scheduler = config.accelerator.prepare(model, optimizer, lr_scheduler)

        self.model = model
        self.optimizer = optimizer
        self.error_fn = nn.MSELoss()
        self.lr_scheduler = lr_scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.TRAIN_TIME_STEPS)


        if load_pretrained:
            models_path_list = os.listdir(config.MODELS_PATH)
            assert len(models_path_list) > 0

            self.model.load_state_dict(
                torch.load(
                    os.path.join(config.MODELS_PATH, models_path_list[-1])
                )
            )
            self.model_name = models_path_list[-1].replace('.pth', '')

    def batch_train(self, images_batch):
        # Sample noise to add to the images
        noise = torch.randn_like(images_batch).to(config.DEVICE)
        bs = images_batch.shape[0]

        # Sample a random timestep for each image
        time_steps = torch.randint(
            0, config.TRAIN_TIME_STEPS, (bs,), device=config.DEVICE
        ).long().to(config.DEVICE)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images_batch, noise, time_steps)

        # Predict the noise residual
        noise_pred = self.model(noisy_images, time_steps, return_dict=False)[0]
        loss = self.error_fn(noise_pred, noise)

        config.accelerator.backward(loss)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss.cpu().item()

    def sample(self):
        self.model.eval()

        # Get random ground truth tensor from dataset
        ground_truth_tensor = ImageDataset(image_tensor_dir=config.TENSOR_DATASET_PATH).get_random()

        # Create an image composed of noise
        input_tensor = torch.randn_like(ground_truth_tensor).unsqueeze(0).to(config.DEVICE)

        output_tensor_list = []
        last_steps_tensor_list = []
        for time_step in tqdm(self.noise_scheduler.timesteps, desc="Sampling", position=1, leave=False, colour='blue'):
            time_step.to(config.DEVICE)

            # Generate the output image using the pre-trained model
            with torch.no_grad():
                residual = self.model(input_tensor, time_step, return_dict=False)[0]

            # Update sample with step
            output_tensor = self.noise_scheduler.step(residual, time_step, input_tensor).prev_sample

            if time_step % (config.TRAIN_TIME_STEPS / 10) == 0:
                output_tensor_list.append(output_tensor.squeeze(0))

            # Save last 10 steps for evaluation
            if len(last_steps_tensor_list) == 10:
                last_steps_tensor_list.pop(0)

            last_steps_tensor_list.append(
                output_tensor.squeeze(0)
            )

        assert len(output_tensor_list) > 0, "No output samples were generated"

        fid_val = evaluate_sample(last_steps_tensor_list)

        return output_tensor_list, fid_val
