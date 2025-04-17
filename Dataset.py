# DATASET.py
import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels_df = pd.read_csv(csv_path)
        self.labels_df.set_index("image", inplace=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_id = os.path.splitext(image_name)[0]

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace('.jpg', '_segmentation.png'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)

        label = self.labels_df.loc[image_id].values.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, label
