
# TRAIN.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Model import UNet
from Utils import get_loaders, check_accuracy, save_checkpoint, load_checkpoint, save_predections_as_images, plot_values_with_index


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "train_images"
TRAIN_MASK_DIR = "train_masks"
VAL_IMG_DIR = "val_images"
VAL_MASK_DIR = "val_masks"


def train_fn(loader, model, optimizer, seg_loss_fn, cls_loss_fn, scaler):
    loop = tqdm(loader)
    model.train()

    for data, masks, labels in loop:
        data = data.to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            seg_preds, cls_preds = model(data)
            seg_loss = seg_loss_fn(seg_preds, masks)
            cls_loss = cls_loss_fn(cls_preds, labels)
            loss = seg_loss + cls_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(seg_loss=seg_loss.item(), cls_loss=cls_loss.item(), total_loss=loss.item())


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY
    )
    dice = []
    iou = []
    accuracy = []

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train_fn(train_loader, model, optimizer, seg_loss_fn, cls_loss_fn, scaler)

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        di, i, acc = check_accuracy(val_loader, model, device=DEVICE)
        dice.append(di)
        iou.append(i)
        accuracy.append(acc)
        save_predections_as_images(val_loader, model, folder="saved_images/", device=DEVICE)
    
    plot_values_with_index(dice, "Dice Coefficient")
    plot_values_with_index(iou, "IoU")
    plot_values_with_index(accuracy, "Accuracy")

if __name__ == "__main__":
    main()

#set NO_ALBUMENTATIONS_UPDATE=1
#"C:/Program Files/Python38/python.exe" "d:/Pattern Recognition/Assignements/A2/Train.py"
