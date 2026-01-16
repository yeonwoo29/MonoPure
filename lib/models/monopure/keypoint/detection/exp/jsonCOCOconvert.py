import json
import os
from PIL import Image

with open("./data/carfusion/annotations/val.json") as f:
    data = json.load(f)

coco = {
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "car",
        "supercategory": "vehicle",
        "keypoints": ["kp{}".format(i+1) for i in range(12)],
        "skeleton": []
    }]
}

ann_id = 0
for img_id, item in enumerate(data):
    file_name = item["file_name"]
    img_path = os.path.join("./data/carfusion/images/val", file_name)
    width, height = Image.open(img_path).size

    coco["images"].append({
        "id": img_id,
        "file_name": file_name,
        "width": width,
        "height": height
    })

    coco["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "bbox": item["bbox"],
        "keypoints": item["keypoints"],
        "num_keypoints": len(item["keypoints"]) // 3
    })

    ann_id += 1

with open("coco_val.json", "w") as f:
    json.dump(coco, f)