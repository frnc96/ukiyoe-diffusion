import os
import json
import shutil
import zipfile
import subprocess
from tqdm import tqdm
import src.config as config
import src.helpers as helper


# Create dirs
helper.create_dir_if_non_existent(config.KAGGLE_PATH)
helper.create_dir_if_non_existent(config.DATASET_PATH)
helper.create_dir_if_non_existent(config.TRAIN_DATASET_PATH)

# Set the location of kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = config.KAGGLE_PATH

# Create kaggle.json file from .env variable
if not os.path.exists(os.path.join(config.KAGGLE_PATH, 'kaggle.json')):
    with open(os.path.join(config.KAGGLE_PATH, 'kaggle.json'), "w") as f:
        json.dump(json.loads(config.KAGGLE_JSON), f)
    print("created kaggle.json file")
else:
    print("kaggle.json file already exists")

# Change permissions on kaggle file
os.chmod(os.path.join(config.KAGGLE_PATH, 'kaggle.json'), 600)
print("changed permissions of kaggle.json to 600")

# Download dataset images from kaggle
if not os.path.exists(os.path.join(config.DATASET_PATH, 'the-metropolitan-museum-of-art-ukiyoe-dataset.zip')):
    subprocess.run([
        'kaggle',
        'datasets',
        'download',
        '-d',
        'kengoichiki/the-metropolitan-museum-of-art-ukiyoe-dataset',
        '-p',
        config.DATASET_PATH,
    ], shell=True)
    print("the-metropolitan-museum-of-art-ukiyoe-dataset.zip downloaded")
else:
    print("the-metropolitan-museum-of-art-ukiyoe-dataset.zip file already exists")

# Unzip the downloaded file
with zipfile.ZipFile(
        os.path.join(config.DATASET_PATH, 'the-metropolitan-museum-of-art-ukiyoe-dataset.zip'),
        'r'
) as zip_ref:
    # Extract all contents of the zip file to a directory
    zip_ref.extractall(config.DATASET_PATH)
    print("the-metropolitan-museum-of-art-ukiyoe-dataset.zip extracted successfully.")

# Remove the .zip file
os.remove(os.path.join(config.DATASET_PATH, 'the-metropolitan-museum-of-art-ukiyoe-dataset.zip'))
print("deleted the-metropolitan-museum-of-art-ukiyoe-dataset.zip")

directories = [
    os.path.join(config.DATASET_PATH, 'images'),
    os.path.join(config.DATASET_PATH, 'images_R'),
]

for directory in tqdm(directories, desc="Moving images"):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(directory, f'{folder_name}/{file_name}')
            shutil.move(file_path, config.TRAIN_DATASET_PATH)

    # Remove dir recursively after extraction
    shutil.rmtree(directory)
    print(f"deleted {directory}")
