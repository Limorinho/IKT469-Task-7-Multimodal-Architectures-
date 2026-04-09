import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("gabrieltardochi/tiny-mm-imdb")

# move this to ./dataset
if not os.path.exists("./dataset"):
    os.makedirs("./dataset")
shutil.move(path, "./dataset/tiny-mm-imdb")

print("Dataset downloaded and moved to ./dataset/tiny-mm-imdb")

