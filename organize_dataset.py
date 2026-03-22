import os
import shutil
import pandas as pd

base_dir = "dataset_raw"

image_dir1 = os.path.join(base_dir, "HAM10000_images_part_1")
image_dir2 = os.path.join(base_dir, "HAM10000_images_part_2")
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")

metadata = pd.read_csv(metadata_path)

output_dir = "dataset/train"

classes = metadata['dx'].unique()

# Create folders
for c in classes:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

# Move images
for _, row in metadata.iterrows():
    img_id = row['image_id'] + ".jpg"
    label = row['dx']

    src1 = os.path.join(image_dir1, img_id)
    src2 = os.path.join(image_dir2, img_id)
    dst = os.path.join(output_dir, label, img_id)

    if os.path.exists(src1):
        shutil.copy(src1, dst)
    elif os.path.exists(src2):
        shutil.copy(src2, dst)

print("✅ Dataset organized!")