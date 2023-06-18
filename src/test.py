import os
import src.config as config
import src.helpers as helpers
import torchvision.transforms as transforms
from src.containers.diffusion_container import DiffusionContainer

helpers.create_dir_if_non_existent(config.SAMPLED_IMAGES_PATH)

container = DiffusionContainer(load_pretrained=True)

output_tensor = container.sample()

# Convert the output tensor to an image
output_image = transforms.ToPILImage()(output_tensor.squeeze())

# Save the output image
image_name = os.path.join(config.SAMPLED_IMAGES_PATH, f"{container.model_name}.jpg")
output_image.save(image_name)

print(f"Successfully generated {image_name}.")
