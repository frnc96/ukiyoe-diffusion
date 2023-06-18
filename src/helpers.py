import os


def create_dir_if_non_existent(path):
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)