
# UTILS.py
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Dataset import CancerDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=True):

    train_dataset = CancerDataset(train_img_dir, train_mask_dir, "GroundTruth.csv", transform=train_transform)
    val_dataset = CancerDataset(val_img_dir, val_mask_dir, "GroundTruth.csv", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    print("\nChecking accuracy...")
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    all_preds_cls = []
    all_labels_cls = []

    with torch.no_grad():
        for x, y, labels in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            labels = labels.to(device)

            seg_preds, cls_preds = model(x)
            seg_preds_bin = (torch.sigmoid(seg_preds) > 0.5).float()
            num_correct += (seg_preds_bin == y).sum()
            num_pixels += torch.numel(seg_preds_bin)
            dice_score += (2 * (seg_preds_bin * y).sum()) / (seg_preds_bin.sum() + y.sum() + 1e-6)
            intersection = (seg_preds_bin * y).sum()
            union = (seg_preds_bin + y).sum() - intersection
            iou_score += intersection / (union + 1e-6)

            all_preds_cls.append((torch.sigmoid(cls_preds) > 0.5).cpu().numpy())
            all_labels_cls.append(labels.cpu().numpy())

    print(f"Segmentation Accuracy: {100 * num_correct / num_pixels:.2f}%")
    print(f"Dice Score: {dice_score / len(loader):.4f}")
    print(f"Mean IoU: {iou_score / len(loader):.4f}")

    y_true = np.concatenate(all_labels_cls, axis=0)
    y_pred = np.concatenate(all_preds_cls, axis=0)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"], zero_division=0))
    accuracy = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    print(f"Overall Classification Accuracy: {accuracy:.4f}")

    os.makedirs("confusion_matrices", exist_ok=True)
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    # Create a large confusion matrix heatmap
    plt.figure(figsize=(8, 6))  # Adjust size as necessary
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"], yticklabels=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"])
    plt.title('Confusion Matrix for All Classes')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Save the large confusion matrix as a single image
    plt.savefig("confusion_matrices/overall_confusion_matrix.png")
    plt.close()
    model.train()
    return dice_score / len(loader), iou_score / len(loader), accuracy

def save_predections_as_images(loader, model, folder="predictions", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x = x.to(device)
            preds, _ = model(x)
            preds = (torch.sigmoid(preds) > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/pred_{i}.png")
    model.train()

def plot_values_with_index(values, filename="plot.png"):
    # Convert torch tensor to numpy if needed
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()  # <-- the important fix

    indices = np.arange(len(values))

    plt.figure(figsize=(8, 6))
    plt.plot(indices, values, marker='o', color='b', linestyle='-', label='Values')
    plt.title("Plot of Values with Index")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")