import os
import xml.etree.ElementTree as ET
import shutil

def map_dataset(split):
    ann_dir = f"data/raw/dataset/{split}/annotations"

    crack_dir = f"data/processed/{split}/crack"
    no_crack_dir = f"data/processed/{split}/no_crack"

    os.makedirs(crack_dir, exist_ok=True)
    os.makedirs(no_crack_dir, exist_ok=True)

    for file in os.listdir(ann_dir):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(ann_dir, file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        label = root.find(".//name").text.lower()
        image_name = root.find(".//filename").text

        # image is inside class subfolder
        src_image = f"data/raw/dataset/{split}/images/{label}/{image_name}"

        if not os.path.exists(src_image):
            print(f"⚠️ Missing image: {src_image}")
            continue

        if label == "scratches":
            dst = os.path.join(crack_dir, image_name)
        else:
            dst = os.path.join(no_crack_dir, image_name)

        shutil.copy(src_image, dst)

    print(f"✅ {split} mapping completed")

if __name__ == "__main__":
    map_dataset("train")
    map_dataset("validation")