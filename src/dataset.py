import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(self, root_dir, subset="train", img_size=256, transform=None):
        self.img_size = img_size
        self.transform = transform

        self.images_dir = os.path.join(root_dir, subset, "images")
        self.masks_dir = os.path.join(root_dir, subset, "masks")

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Klasör bulunamadı: {self.images_dir}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.image_list = sorted([f for f in os.listdir(self.images_dir)
                                  if f.lower().endswith(valid_extensions)])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # 1. Okuma
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. Augmentation (Veri Artırma) Uygulama
        if self.transform:
            # Albumentations hem resme hem maskeye aynı işlemi uygular
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Transform yoksa (Test aşaması) sadece resize yap
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        # 3. Kenar (Edge) Çıkarma (Augmentation SONRASI yapılmalı!)
        # Kenarların bozulmaması için maskeyi binary hale getirip kenar buluyoruz
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(binary_mask, 100, 200)

        # 4. Normalizasyon ve Tensor
        image = image.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        edge = edge.astype('float32') / 255.0

        mask = np.expand_dims(mask, axis=0)
        edge = np.expand_dims(edge, axis=0)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        edge = torch.from_numpy(edge).float()

        return image, mask, edge