import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mobileNetV3 import MobileNetV3_Small
import torch.nn as nn
import torch.optim as optim

DATA_DIR = "data/Train"
BATCH_SIZE = 64
IMG_SIZE = 224


# 1. Custom Dataset
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            class_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(class_folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(class_folder, fname))
                    self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


if __name__ == "__main__":
    # 2. Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # 3. DataLoader
    dataset = AffectNetDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    print("cuda available:", torch.cuda.is_available())
    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3_Small(num_classes=len(dataset.classes)).to(device)
    model.eval()  # Set to eval mode for inference

    # 5. Forward pass on one batch
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            print("Batch output shape:", outputs.shape)
            break  # Remove this break to process all batches

    EPOCHS = 30  # Set as needed

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()  # Set model to training mode

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        )

    print("Training finished.")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = "mobilenetv3_small_affectnet.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {onnx_path}")
