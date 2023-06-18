import os
import uuid
import torch
import torch.nn as nn
import src.config as config
import torch.optim as optim
from src.networks.diffusion import DiffusionNetwork


class DiffusionContainer:

    def __init__(self, load_pretrained=False):
        self.model = DiffusionNetwork().to(config.DEVICE)
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

    def batch_train(self, images_batch):
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(images_batch)

        # Compute loss
        loss = self.error_fn(outputs, images_batch)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample(self):
        self.model.eval()

        # Create an image composed of noise
        input_tensor = torch.randn(1, 3, config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]).to(config.DEVICE)

        # Generate the output image using the pre-trained model
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        return output_tensor
