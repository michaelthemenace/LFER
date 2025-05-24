import os
import matplotlib.pyplot as plt
from PIL import Image
import random

DATA_DIR = "data"
SETS = ["Train", "Test"]

all_class_counts = {}
all_file_paths = []
all_labels = []
all_sets = []

for set_name in SETS:
    set_dir = os.path.join(DATA_DIR, set_name)
    if not os.path.isdir(set_dir):
        continue
    classes = [
        d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))
    ]
    for cls in classes:
        class_folder = os.path.join(set_dir, cls)
        images = [
            f
            for f in os.listdir(class_folder)
            if os.path.isfile(os.path.join(class_folder, f))
        ]
        all_class_counts[(set_name, cls)] = len(images)
        for img in images:
            all_file_paths.append(os.path.join(class_folder, img))
            all_labels.append(cls)
            all_sets.append(set_name)

# Plot class distribution for Train and Test
plt.figure(figsize=(12, 6))
labels = [f"{set_name}/{cls}" for (set_name, cls) in all_class_counts.keys()]
counts = list(all_class_counts.values())
bars = plt.bar(labels, counts)
plt.title("Class Distribution in Train and Test Sets")
plt.xlabel("Set/Class")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")

# Add count labels on top of each bar
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

# Show sample images per class for Train and Test
for set_name in SETS:
    set_dir = os.path.join(DATA_DIR, set_name)
    if not os.path.isdir(set_dir):
        continue
    classes = [
        d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))
    ]
    fig, axs = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
    for i, cls in enumerate(classes):
        class_folder = os.path.join(set_dir, cls)
        images = [
            f
            for f in os.listdir(class_folder)
            if os.path.isfile(os.path.join(class_folder, f))
        ]
        sample_imgs = random.sample(images, min(5, len(images)))
        for j, img_name in enumerate(sample_imgs):
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path)
            axs[i, j].imshow(img)
            axs[i, j].set_title(f"{set_name}/{cls}")
            axs[i, j].axis("off")
    plt.suptitle(f"Sample Images from {set_name} Set")
    plt.tight_layout()
    plt.show()
