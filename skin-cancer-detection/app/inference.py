import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from app.models.model import UNet

def load_config():
    with open("config/config.yaml", 'r') as config_file:
        return yaml.safe_load(config_file)

def inference(image_path):
    # Load configuration
    config = load_config()
    
    # Configuration
    DEVICE = torch.device(config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))
    IMAGE_HEIGHT = config.get('image_height', 256)
    IMAGE_WIDTH = config.get('image_width', 256)
    CHECKPOINT_PATH = config.get('checkpoint_path', "data/models/my_checkpoint.pth.tar")
    CLASS_NAMES = config.get('class_names', ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"])
    OUTPUT_DIR = config.get('output_directory', "output/inference_outputs")

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

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path).replace(".jpg", "_result.png"))
    plt.savefig(save_path)
    print(f"Inference result saved to {save_path}")
    plt.close()  # Close instead of show() for headless Docker execution
    
    return {
        'segmentation_mask': seg_mask,
        'classification': top_class,
        'classification_scores': dict(zip(CLASS_NAMES, cls_pred.tolist())),
        'output_path': save_path
    }