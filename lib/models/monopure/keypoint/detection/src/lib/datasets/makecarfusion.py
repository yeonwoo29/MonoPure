import os
import json
import shutil

src_root = 'D:/detection/data/carfusion'  # car_butler1 등 폴더가 있는 경로
dst_root = './carfusion'

train_scenes = ['car_butler1', 'car_butler2', 'car_craig1', 'car_craig2', 'car_fifth1', 'car_fifth2', 'car_morewood1', 'car_morewood2', 'car_penn1', 'car_penn2']
val_scenes = ['car_morewood1', 'car_morewood2', 'car_penn1', 'car_penn2']

os.makedirs(os.path.join(dst_root, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'annotations'), exist_ok=True)

def convert_keypoints(kpts):
    # [[x, y, v], ...] → [x, y, v, ...]
    flat = []
    for pt in kpts:
        flat.extend(pt)
    return flat

def convert_split(scenes, split_name):
    results = []
    img_dst = os.path.join(dst_root, f'images/{split_name}')
    for scene in scenes:
        ann_path = os.path.join(src_root, scene, 'annotations.json')
        img_dir = os.path.join(src_root, scene, 'images_jpg')
        
        with open(ann_path, 'r') as f:
            anns = json.load(f)

        for i, ann in enumerate(anns):
            img_name = ann['file_name']
            new_img_name = f"{scene}_{img_name}"

            src_img_path = os.path.join(img_dir, img_name)
            dst_img_path = os.path.join(img_dst, new_img_name)

            # 이미지 복사
            if not os.path.isfile(src_img_path):
                print(f"이미지 누락: {src_img_path}")
                continue
            shutil.copyfile(src_img_path, dst_img_path)

            # annotation 가공
            ann['file_name'] = new_img_name
            ann['keypoints'] = convert_keypoints(ann['keypoints'])

            results.append(ann)

    # 결과 저장
    with open(os.path.join(dst_root, f'annotations/{split_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)

convert_split(train_scenes, 'train')
convert_split(val_scenes, 'val')
