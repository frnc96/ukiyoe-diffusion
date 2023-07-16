import os
import wandb
from pathlib import Path
import configuration as config
import matplotlib.pyplot as plt


def create_dir_if_non_existent(path):
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)


def init_wandb() -> wandb:
    wandb.login(key=config.WANDB_KEY)
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config=config.HYPER_PARAMETERS,
        dir=config.WANDB_PATH
    )
    return wandb


def plot_sample_images(image_tensor_list, filename):
    images_len = len(image_tensor_list)

    fig, axs = plt.subplots(1, images_len, figsize=(images_len*5, 5))

    for i, ax in enumerate(axs):
        image = image_tensor_list[i].permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis('off')

    plot_image_path = f"{config.SAMPLED_IMAGES_PATH}/{filename}.jpg"
    plt.tight_layout()

    fig.savefig(Path(plot_image_path))
    plt.close(fig)

    return plot_image_path
