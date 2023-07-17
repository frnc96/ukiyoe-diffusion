import os
import torch
from dotenv import load_dotenv
from accelerate import Accelerator

# Load environment variables from .env file
load_dotenv()

accelerator = Accelerator()
DEVICE = accelerator.device
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KAGGLE_JSON = os.getenv("KAGGLE_JSON")

# Paths
KAGGLE_PATH = os.path.abspath("./kaggle")
DATASET_PATH = os.path.abspath("./dataset")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train")
TENSOR_DATASET_PATH = os.path.join(DATASET_PATH, "tensor")
SAMPLED_IMAGES_PATH = os.path.join(DATASET_PATH, "sampled")
MODELS_PATH = os.path.abspath("./models")
WANDB_PATH = os.path.abspath("./wandb")

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES").lower() in ('true', '1')

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
IMAGE_RESIZE = (256, 256)
TRAIN_TIME_STEPS = 1000
LR_WARMUP_STEPS = 500

HYPER_PARAMETERS = {
    'learning_rate': LEARNING_RATE,
}
