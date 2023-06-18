import os
import configuration as config
import helpers as helpers
import torchvision.transforms as transforms
from containers.diffusion_container import DiffusionContainer

helpers.create_dir_if_non_existent(config.SAMPLED_IMAGES_PATH)

container = DiffusionContainer(load_pretrained=True)

output_tensor_list = container.sample(nr=9)

# helpers.plot_sample_images(output_tensor_list, 'llazi')

# Select image with the lowest loss
output_tensor = min(output_tensor_list, key=lambda x: x["mse"])['tensor']

# Convert the output tensor to an image
output_image = transforms.ToPILImage()(output_tensor.squeeze())

# Save the output image
image_name = os.path.join(config.SAMPLED_IMAGES_PATH, f"{container.model_name}.jpg")
output_image.save(image_name)

print(f"Successfully generated {image_name}.")
