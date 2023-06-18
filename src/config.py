import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KAGGLE_JSON = os.getenv("KAGGLE_JSON")

# Paths
KAGGLE_PATH = os.path.abspath("./kaggle")
DATASET_PATH = os.path.abspath("./dataset")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train")
SAMPLED_IMAGES_PATH = os.path.join(DATASET_PATH, "sampled")
MODELS_PATH = os.path.abspath("./models")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_RESIZE = (1920, 1080)
