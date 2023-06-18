import os
import wandb
from pathlib import Path
import config as config
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


def plot_sample_images(image_dict_list, filename):
    grid_size = (3, 3)

    fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    ax = ax.flatten()

    assert len(image_dict_list) >= 9

    for i, dictionary in enumerate(image_dict_list, 0):
        image_tensor = dictionary['tensor']
        image = image_tensor.permute(1, 2, 0).cpu().numpy()

        ax[i].imshow(image)
        ax[i].axis('off')

        if i == 8:
            break

    plot_image_path = f"{config.SAMPLED_IMAGES_PATH}/{filename}.jpg"
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig.savefig(Path(plot_image_path))
    plt.close(fig)

    return plot_image_path
