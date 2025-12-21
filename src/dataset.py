import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(self, images_path, masks_path, img_size=256, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_size = img_size
        self.transform = transform
        self.image_list = sorted(os.listdir(images_path))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        # Dosya yolları
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, img_name)

        # 1. Görüntü ve Maskeyi Oku
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. Yeniden Boyutlandırma (Resize)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # 3. Kenar (Edge) Çıkarma
        edge = cv2.Canny(mask, 100, 200)

        # 4. Normalizasyon ve Tensor Dönüşümü
        image = image / 255.0
        mask = mask / 255.0
        edge = edge / 255.0

        # Maskeleri (1, H, W) formatına getirme
        mask = np.expand_dims(mask, axis=0)
        edge = np.expand_dims(edge, axis=0)

        # PyTorch Tensor'una çevir
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        edge = torch.from_numpy(edge).float()

        return image, mask, edge