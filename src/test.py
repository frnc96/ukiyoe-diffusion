import os
import torch
import src.config as config
import src.helpers as helpers
import torchvision.transforms as transforms
from src.networks.diffusion import DiffusionNetwork

helpers.create_dir_if_non_existent(config.SAMPLED_IMAGES_PATH)

models_path_list = os.listdir(config.MODELS_PATH)

if len(models_path_list) == 0:
    print("There are no pretrained models in the models folder")
    exit(1)

# Create an instance of the pre-trained model
model = DiffusionNetwork()

# Load the pre-trained model weights
latest_model_name = models_path_list[-1]
model.load_state_dict(
    torch.load(
        os.path.join(config.MODELS_PATH, latest_model_name)
    )
)
model.eval()

input_tensor = torch.randn(1, 3, config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1])

# Generate the output image using the pre-trained model
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert the output tensor to an image
output_image = transforms.ToPILImage()(output_tensor.squeeze())

# Save the output image
output_image.save(
    os.path.join(config.SAMPLED_IMAGES_PATH, f"{latest_model_name}.jpg")
)

print("Sample image generated successfully.")
