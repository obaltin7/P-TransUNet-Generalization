import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import PolypDataset
from src.model import PTransUNet

# --------------------------------------------------------
# AYARLAR (RTX 2060 6GB İçin Optimize Edildi)
# --------------------------------------------------------
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
IMG_SIZE = 256
DATA_DIR = "data/ETIS-Larib"


# --------------------------------------------------------
# KAYIP FONKSİYONLARI (LOSS FUNCTIONS)
# --------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


def ohem_loss(logits, target, rate=0.7):

    bce = nn.BCEWithLogitsLoss(reduction='none')(logits, target)

    loss_sorted, _ = bce.view(-1).sort(descending=True)
    num_kept = int(rate * loss_sorted.numel())
    return loss_sorted[:num_kept].mean()


# --------------------------------------------------------
# EĞİTİM FONKSİYONU
# --------------------------------------------------------
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets, edges) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        edges = edges.to(DEVICE)

        with autocast(device_type='cuda', enabled=True):
            # 1. Model Tahminleri (LOGITS - Ham Çıktılar)
            logits_seg, logits_edge = model(data)

            # 2. Segmentasyon Kayıpları
            # BCEWithLogitsLoss -> Ham çıktı (logits) ister
            loss_seg_bce = nn.BCEWithLogitsLoss()(logits_seg, targets)

            pred_seg = torch.sigmoid(logits_seg)
            loss_seg_dice = DiceLoss()(pred_seg, targets)

            # 3. Kenar (Edge) Kaybı
            loss_edge = ohem_loss(logits_edge, edges)

            # Toplam Kayıp (Makaledeki Katsayılar: 0.5, 0.3, 0.2)
            loss = 0.5 * loss_seg_bce + 0.3 * loss_seg_dice + 0.2 * loss_edge

        # Geri Yayılım (Backpropagation)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# --------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU
# --------------------------------------------------------
def main():
    print(f"🚀 Eğitim Başlıyor... Cihaz: {DEVICE}")

    # 1. Veri Setini Hazırla
    full_dataset = PolypDataset(
        images_path=f"{DATA_DIR}/images",
        masks_path=f"{DATA_DIR}/masks",
        img_size=IMG_SIZE
    )

    # %90 Eğitim, %10 Test olarak ayır
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Modeli Kur
    model = PTransUNet(num_classes=1).to(DEVICE)

    # 3. Optimizer ve Scaler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device='cuda')  # Yeni versiyon düzeltmesi

    # Klasör yoksa oluştur
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # 4. Eğitim Döngüsü
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        # Eğit
        model.train()
        avg_loss = train_fn(train_loader, model, optimizer, None, scaler)

        # Modeli Kaydet
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print(f"💾 Model kaydedildi! Loss: {avg_loss:.4f}")
        else:
            print(f"Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()