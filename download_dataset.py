import kagglehub
import shutil
import os

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Downloaded to:", path)

if not os.path.exists("dataset_raw"):
    shutil.copytree(path, "dataset_raw")

print("✅ Dataset ready!")