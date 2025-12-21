import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from src.dataset import PolypDataset
from src.model import PTransUNet

# AYARLAR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
DATA_DIR = "data/ETIS-Larib"
MODEL_PATH = "saved_models/best_model.pth"


def calculate_metrics(pred, target, threshold=0.5):
    """
    Dice, IoU, Precision ve Recall Hesaplama
    TP: True Positive, FP: False Positive, FN: False Negative
    """
    # Tahminleri 0 ve 1'e çevir (Thresholding)
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    # Piksel bazlı hesaplama
    TP = (pred * target).sum()  # Doğru bilinen polip pikselleri
    FP = (pred * (1 - target)).sum()  # Polip sandığımız ama olmayan yerler
    FN = ((1 - pred) * target).sum()  # Kaçırdığımız polip pikselleri

    epsilon = 1e-8  # Sıfıra bölünme hatasını önlemek için

    dice = (2. * TP) / (pred.sum() + target.sum() + epsilon)
    iou = TP / (pred.sum() + target.sum() - TP + epsilon)

    # Precision: "Polip dediklerimin yüzde kaçı gerçekten polip?"
    precision = TP / (TP + FP + epsilon)

    # Recall (Sensitivity): "Gerçek poliplerin yüzde kaçını yakalayabildim?"
    recall = TP / (TP + FN + epsilon)

    return dice.item(), iou.item(), precision.item(), recall.item()


def visualize_results(model, loader, save_path="output/result_viz_metrics.png"):
    model.eval()
    try:
        data, mask, edge = next(iter(loader))
    except StopIteration:
        return

    data, mask = data.to(DEVICE), mask.to(DEVICE)

    with torch.no_grad():
        pred_seg, pred_edge = model(data)
        pred_seg = torch.sigmoid(pred_seg)
        pred_edge = torch.sigmoid(pred_edge)

    # CPU'ya al ve numpy yap
    data = data.cpu().numpy()
    mask = mask.cpu().numpy()
    edge = edge.cpu().numpy()
    pred_seg = pred_seg.cpu().numpy()
    pred_edge = pred_edge.cpu().numpy()

    # Görselleştirme (4 örnek)
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(min(4, data.shape[0])):
        # Orijinal Resim
        img = np.transpose(data[i], (1, 2, 0))
        ax[i, 0].imshow(img)
        ax[i, 0].set_title("Giriş Resmi")
        ax[i, 0].axis("off")

        # Gerçek Maske
        ax[i, 1].imshow(mask[i].squeeze(), cmap='gray')
        ax[i, 1].set_title("Gerçek Maske")
        ax[i, 1].axis("off")

        # Tahmin
        ax[i, 2].imshow(pred_seg[i].squeeze() > 0.5, cmap='gray')
        ax[i, 2].set_title("Tahmin")
        ax[i, 2].axis("off")

        # Kenar
        ax[i, 3].imshow(pred_edge[i].squeeze(), cmap='jet')
        ax[i, 3].set_title("Kenar Tahmini")
        ax[i, 3].axis("off")

    plt.tight_layout()
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(save_path)
    print(f"🖼️ Görsel sonuçlar '{save_path}' dosyasına kaydedildi!")


def main():
    print(f"🔎 Detaylı Test Başlıyor (Dice, IoU, Precision, Recall)...")

    # 1. Veri Seti
    full_dataset = PolypDataset(f"{DATA_DIR}/images", f"{DATA_DIR}/masks", img_size=IMG_SIZE)


    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    # 2. Modeli Yükle
    model = PTransUNet(num_classes=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("❌ Model bulunamadı!")
        return

    model.eval()

    # 3. Hesaplama
    metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    steps = 0

    print("📊 Metrikler Hesaplanıyor...")
    with torch.no_grad():
        for data, target, _ in tqdm(val_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            out_seg, _ = model(data)
            out_seg = torch.sigmoid(out_seg)

            d, i, p, r = calculate_metrics(out_seg, target)

            metrics["dice"] += d
            metrics["iou"] += i
            metrics["precision"] += p
            metrics["recall"] += r
            steps += 1

    # Ortalamaları al
    for k in metrics:
        metrics[k] /= steps

    print(f"\n✅ KARŞILAŞTIRMALI SONUÇLAR:")
    print(f"--------------------------------------------------")
    print(f"{'METRİK':<15} | {'DOĞRULAMA SONUCU':<15} | {'MAKALEDEKİ SONUÇ':<15}")
    print(f"--------------------------------------------------")
    print(f"{'mDice':<15} | {metrics['dice']:.4f}          | 0.9352")
    print(f"{'mIoU':<15} | {metrics['iou']:.4f}          | 0.8893")
    print(f"{'Recall':<15} | {metrics['recall']:.4f}          | 0.9389")
    print(f"{'Precision':<15} | {metrics['precision']:.4f}          | 0.9379")
    print(f"--------------------------------------------------")

    visualize_results(model, val_loader)


if __name__ == "__main__":
    main()