import os
import matplotlib.pyplot as plt
from PIL import Image
import random

DATA_DIR = "data"
SETS = ["Train", "Test"]

all_class_counts = {"Train": {}, "Test": {}}
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
        all_class_counts[set_name][cls] = len(images)
        for img in images:
            all_file_paths.append(os.path.join(class_folder, img))
            all_labels.append(cls)
            all_sets.append(set_name)

for set_name in SETS:
    plt.figure(figsize=(8, 4))
    classes = list(all_class_counts[set_name].keys())
    counts = list(all_class_counts[set_name].values())
    bars = plt.bar(classes, counts)
    plt.title(f"Class Distribution in {set_name} Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

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

for set_name in SETS:
    set_dir = os.path.join(DATA_DIR, set_name)
    if not os.path.isdir(set_dir):
        continue
    classes = [
        d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))
    ]
    n_classes = len(classes)
    n_cols = 4
    n_rows = (n_classes * 2 + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axs = axs.flatten()
    img_idx = 0
    for cls in classes:
        class_folder = os.path.join(set_dir, cls)
        images = [
            f
            for f in os.listdir(class_folder)
            if os.path.isfile(os.path.join(class_folder, f))
        ]
        sample_imgs = random.sample(images, min(2, len(images)))
        for img_name in sample_imgs:
            if img_idx >= len(axs):
                break
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path)
            axs[img_idx].imshow(img)
            axs[img_idx].set_title(f"{set_name}/{cls}")
            axs[img_idx].axis("off")
            img_idx += 1

    for ax in axs[img_idx:]:
        ax.axis("off")
    plt.suptitle(f"Sample Images from {set_name} Set")
    plt.tight_layout()
    plt.show()
