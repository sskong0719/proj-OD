import os
import re
import json
import shutil
import yaml
from pathlib import Path

# === CONFIG ===
DATASETS = {
    'pets': {
        'yaml': 'yolo-pets/data.yaml',
        'images': {
            'train': 'yolo-pets/images/train',
            'val': 'yolo-pets/images/val',
        },
        'labels': {
            'train': 'yolo-pets/labels/train',
            'val': 'yolo-pets/labels/val',
        }
    },
    'dogs': {
        'yaml': 'yolo-dogs/data.yaml',
        'images': {
            'train': 'yolo-dogs/images/train',
            'val': 'yolo-dogs/images/val',
        },
        'labels': {
            'train': 'yolo-dogs/labels/train',
            'val': 'yolo-dogs/labels/val',
        }
    }
}
MERGED_PATH = Path('dataset')
MERGED_DATA_YAML = MERGED_PATH / 'merged-data.yaml'


# === Helper: Normalize class name ===
def normalize_class_name(raw_name):
    raw_name = re.sub(r'^n\d+-', '', raw_name)
    if raw_name.lower().endswith('.ds_store'):
        return None
    return raw_name.lower().replace('-', '_').strip()


# === Step 1: Merge and normalize class names ===
merged_classes = {}
index_maps = {}
counter = 0

for prefix, config in DATASETS.items():
    with open(config['yaml'], 'r') as f:
        raw_yaml = yaml.safe_load(f)

    index_maps[prefix] = {}
    for raw_idx, raw_name in raw_yaml['names'].items():
        norm_name = normalize_class_name(raw_name)
        if not norm_name:
            continue
        if norm_name not in merged_classes:
            merged_classes[norm_name] = counter
            counter += 1
        index_maps[prefix][str(raw_idx)] = merged_classes[norm_name]

# === Save merged data.yaml ===
MERGED_PATH.mkdir(parents=True, exist_ok=True)
merged_yaml = {
    'path': str(MERGED_PATH.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {v: k for k, v in merged_classes.items()}
}
with open(MERGED_DATA_YAML, 'w') as f:
    yaml.dump(merged_yaml, f)

# === Save index maps for reference ===
with open(MERGED_PATH / 'index_maps.json', 'w') as f:
    json.dump(index_maps, f, indent=2)

print("✅ Merged class list saved to:", MERGED_DATA_YAML)

# === Step 2: Copy and remap images + labels ===
def process_split(prefix, split):
    src_imgs = Path(DATASETS[prefix]['images'][split])
    src_lbls = Path(DATASETS[prefix]['labels'][split])
    dst_imgs = MERGED_PATH / 'images' / split
    dst_lbls = MERGED_PATH / 'labels' / split
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    for img_file in src_imgs.glob("*.*"):
        base_name = f"{prefix}_{img_file.stem}"
        new_img_path = dst_imgs / f"{base_name}{img_file.suffix}"
        new_lbl_path = dst_lbls / f"{base_name}.txt"

        shutil.copy(img_file, new_img_path)

        lbl_file = src_lbls / f"{img_file.stem}.txt"
        if lbl_file.exists():
            with open(lbl_file, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_idx = parts[0]
                new_idx = str(index_maps[prefix].get(old_idx, old_idx))  # fallback just in case
                new_line = ' '.join([new_idx] + parts[1:])
                new_lines.append(new_line)

            with open(new_lbl_path, 'w') as f:
                f.write('\n'.join(new_lines))

        else:
            print(f"⚠️ No label found for {img_file.name}, skipping label.")

# === Process both datasets and splits ===
for prefix in DATASETS:
    for split in ['train', 'val']:
        print(f"🔄 Processing {prefix.upper()} - {split}")
        process_split(prefix, split)

print("✅ All images and labels merged into 'dataset/'")
