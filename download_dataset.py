import os
import json
from zipfile import ZipFile


with open("kaggle.json", "r") as file:
    kaggle_credentials = json.load(file)

os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]

os.system('kaggle datasets download -d abdallahalidev/plantvillage-dataset')


with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

print("Download and extraction complete.")
