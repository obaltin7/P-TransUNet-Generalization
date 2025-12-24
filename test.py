import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import cv2

from src.dataset import PolypDataset
from src.model import PTransUNet

# --------------------------------------------------------
# AYARLAR
# --------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
DATA_DIR = "data/ETIS-Larib"
MODEL_PATH = "saved_models/best_model_etis.pth"

# KAZANAN EŞİK DEĞERİ (Sabitlendi)
BEST_THRESHOLD = 0.3


def calculate_metrics(pred, target, threshold=0.5):
    """Genel Metrik Hesaplama"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    epsilon = 1e-8

    dice = (2. * TP) / (pred_bin.sum() + target_bin.sum() + epsilon)
    iou = TP / (pred_bin.sum() + target_bin.sum() - TP + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    return dice.item(), iou.item(), precision.item(), recall.item()


def calculate_single_dice(pred, target, threshold=0.5):
    """Sıralama için tekil Dice skoru"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    TP = (pred_bin * target_bin).sum()
    epsilon = 1e-8
    dice = (2. * TP) / (pred_bin.sum() + target_bin.sum() + epsilon)
    return dice.item()


def visualize_best_results(model, dataset, threshold, save_path="output/best_results.png"):
    """
    Test setindeki en başarılı 4 görseli seçer ve kaydeder.
    """
    print("🖼️ Görseller taranıyor ve en iyiler seçiliyor...")
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    with torch.no_grad():
        for data, mask, edge in tqdm(loader, desc="Analiz"):
            data, mask = data.to(DEVICE), mask.to(DEVICE)

            # Tahmin
            pred_seg_logits, pred_edge_logits = model(data)
            pred_seg = torch.sigmoid(pred_seg_logits)
            pred_edge = torch.sigmoid(pred_edge_logits)

            # Dice skoru hesapla
            dice_score = calculate_single_dice(pred_seg, mask, threshold=threshold)

            results.append({
                "dice": dice_score,
                "img": data.cpu().numpy()[0],
                "mask": mask.cpu().numpy()[0],
                "edge": edge.cpu().numpy()[0],
                "pred_seg": pred_seg.cpu().numpy()[0],
                "pred_edge": pred_edge.cpu().numpy()[0]
            })

    # Skora göre sırala (En yüksek en üstte)
    results.sort(key=lambda x: x["dice"], reverse=True)
    top_results = results[:4]

    if not top_results:
        print("❌ Sonuç bulunamadı.")
        return

    # Çizim
    batch_n = len(top_results)
    fig, ax = plt.subplots(batch_n, 4, figsize=(12, 3 * batch_n))
    if batch_n == 1: ax = ax.reshape(1, -1)

    for i, res in enumerate(top_results):
        # 1. Giriş
        img = np.transpose(res["img"], (1, 2, 0))
        ax[i, 0].imshow(img)
        ax[i, 0].set_title("Giriş Resmi")
        ax[i, 0].axis("off")

        # 2. Gerçek Maske
        ax[i, 1].imshow(res["mask"].squeeze(), cmap='gray')
        ax[i, 1].set_title("Gerçek Maske")
        ax[i, 1].axis("off")

        # 3. Tahmin
        ax[i, 2].imshow(res["pred_seg"].squeeze() > threshold, cmap='gray')
        ax[i, 2].set_title(f"Tahmin (Dice: {res['dice']:.3f})")
        ax[i, 2].axis("off")

        # 4. Kenar
        ax[i, 3].imshow(res["pred_edge"].squeeze(), cmap='jet')
        ax[i, 3].set_title("Kenar")
        ax[i, 3].axis("off")

    plt.tight_layout()
    if not os.path.exists("output"): os.makedirs("output")
    plt.savefig(save_path)
    print(f"✨ En iyi {batch_n} sonuç '{save_path}' dosyasına kaydedildi!")


def main():
    print(f"🔎 Final Test Başlıyor (Threshold: {BEST_THRESHOLD})...")

    test_dataset = PolypDataset(root_dir=DATA_DIR, subset="test", img_size=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = PTransUNet(num_classes=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("❌ Model bulunamadı!");
        return

    model.eval()

    metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    steps = 0

    print("📊 Metrikler Hesaplanıyor...")
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            out_seg, _ = model(data)
            out_seg = torch.sigmoid(out_seg)

            d, i, p, r = calculate_metrics(out_seg, target, threshold=BEST_THRESHOLD)

            metrics["dice"] += d
            metrics["iou"] += i
            metrics["precision"] += p
            metrics["recall"] += r
            steps += 1

    # Ortalamalar
    for k in metrics: metrics[k] /= steps

    print(f"\n✅ ETIS-Larib NİHAİ TEST SONUÇLARI (Threshold: {BEST_THRESHOLD}):")
    print(f"--------------------------------------------------")
    print(f"{'mDice':<15} | {metrics['dice']:.4f}")
    print(f"{'mIoU':<15} | {metrics['iou']:.4f}")
    print(f"{'Recall':<15} | {metrics['recall']:.4f}")
    print(f"{'Precision':<15} | {metrics['precision']:.4f}")
    print(f"--------------------------------------------------")

    # En iyileri görselleştir
    visualize_best_results(model, test_dataset, threshold=BEST_THRESHOLD)


if __name__ == "__main__":
    main()