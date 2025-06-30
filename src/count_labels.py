import os
from pathlib import Path
import json
from collections import Counter
from dataloader import FacialEmotionDataLoader
from tqdm import tqdm

# Dataset Path
dataset_path = "./data"

# Load Dataset (no train-test split needed)
loader = FacialEmotionDataLoader(dataset_path=dataset_path, model_type="cnn")

# Extract all labels from the dataset
# all_labels = [label for _, label in loader.full_dataset]
all_labels = list()
for _, label in tqdm(loader.full_dataset):
    all_labels.append(label)

# Count labels
class_counts = Counter(all_labels)
class_names = loader.classes

# Arrange counts in index order
# counts_list = [class_counts[i] for i in range(len(class_names))]
counts_list = list()
for i in tqdm(range(len(class_names))):
    counts_list.append(class_counts[i])
counts_dict = dict(zip(class_names, counts_list))

# Save to JSON
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "output"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "class_counts.json"

with open(output_file, "w") as file:
    json.dump(counts_dict, file, indent=4)

print(f"Class counts saved to `class_counts.json`: {counts_dict}")
