import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from Model import UNet  # Make sure this matches your model file
import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]  # Adjust if needed

# Preprocessing
transform = Compose([
    Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
    ToTensorV2(),
])

# Load model
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

def inference(image_path):
    # Load image
    image = cv2.imread(image_path)
    original = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_pred, cls_pred = model(image_tensor)
        seg_pred = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
        seg_mask = (seg_pred > 0.5).astype(np.uint8)

        cls_pred = torch.sigmoid(cls_pred).squeeze().cpu().numpy()
        top_class = CLASS_NAMES[np.argmax(cls_pred)]

    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(seg_mask, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Classification: {top_class}")
    plt.bar(CLASS_NAMES, cls_pred)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("inference_outputs", exist_ok=True)
    save_path = os.path.join("inference_outputs", os.path.basename(image_path).replace(".jpg", "_result.png"))
    plt.savefig(save_path)
    print(f"Inference result saved to {save_path}")
    plt.show()

# Example usage
if __name__ == "__main__":
    inference("example.jpg")  # Replace with your image path
