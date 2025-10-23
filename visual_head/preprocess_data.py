import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw


dataset_path = "/home/ma-user/code/guixiyan/sparse/ocrbench.parquet"
data = load_dataset("parquet", data_files=dataset_path, split="train")

image_folder = "/obs/users/guixiyan/ocrbench/image"
json_folder = "/obs/users/guixiyan/ocrbench/json"


if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Directory '{image_folder}' created.")
else:
    print(f"Directory '{image_folder}' already exists.")


if not os.path.exists(json_folder):
    os.makedirs(json_folder)
    print(f"Directory '{json_folder}' created.")
else:
    print(f"Directory '{json_folder}' already exists.")


converted_data = []
for id, da in enumerate(tqdm(data)):
    json_data = {"id": id}
    
    if da.get("image") is not None:
        image_path = os.path.join(image_folder, f"{id}.jpg")
        json_data["image_name"] = image_path
        da["image"].save(image_path)
    
    
    if "2_coord" in da:
        json_data["2_coord"] = da["2_coord"]
    
    converted_data.append(json_data)


json_output_path = os.path.join(json_folder, "ocrbench.json")
with open(json_output_path, "w") as f:
    json.dump(converted_data, f, indent=4)

print("Finished processing and saving data.")
