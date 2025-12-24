import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A

from src.dataset import PolypDataset
from src.model import PTransUNet

# --------------------------------------------------------
# AYARLAR
# --------------------------------------------------------
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
IMG_SIZE = 256
DATA_DIR = "data/ETIS-Larib"
MODEL_SAVE_PATH = "saved_models/best_model_etis.pth"

# Transfer Learning Dosyası (Eski projedeki model)
PRETRAINED_PATH = "saved_models/kvasir_pretrained.pth"
USE_TRANSFER_LEARNING = True

# --------------------------------------------------------
# GÜÇLÜ VERİ ARTIRMA (AUGMENTATION)
# --------------------------------------------------------
train_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Rotate(limit=35, p=0.5),           # Rastgele döndür
    A.HorizontalFlip(p=0.5),             # Yatay çevir
    A.VerticalFlip(p=0.5),               # Dikey çevir
    A.RandomBrightnessContrast(p=0.2),   # Işıkla oyna
    A.GaussNoise(p=0.1),                 # Hafif karıncalanma ekle (Dayanıklılık için)
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1) # Renkleri hafif kaydır
])

val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
])

# --------------------------------------------------------
# KAYIP FONKSİYONLARI
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
            logits_seg, logits_edge = model(data)

            loss_seg_bce = nn.BCEWithLogitsLoss()(logits_seg, targets)
            pred_seg = torch.sigmoid(logits_seg)
            loss_seg_dice = DiceLoss()(pred_seg, targets)
            loss_edge = ohem_loss(logits_edge, edges)

            # Dice Loss ağırlığını arttırıldı (Küçük nesneler için iyidir)
            # Eski: 0.5 BCE + 0.3 Dice
            # Yeni: 0.4 BCE + 0.4 Dice + 0.2 Edge
            loss = 0.4 * loss_seg_bce + 0.4 * loss_seg_dice + 0.2 * loss_edge

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print(f"🚀 Eğitim Başlıyor... Cihaz: {DEVICE}")
    print(f"📂 Transfer Learning Modu: {'AÇIK' if USE_TRANSFER_LEARNING else 'KAPALI'}")

    # Datasetleri yeni transformlarla yükle
    train_dataset = PolypDataset(DATA_DIR, subset="train", img_size=IMG_SIZE, transform=train_transform)
    val_dataset = PolypDataset(DATA_DIR, subset="test", img_size=IMG_SIZE, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = PTransUNet(num_classes=1).to(DEVICE)

    # TRANSFER LEARNING AŞAMASI
    if USE_TRANSFER_LEARNING and os.path.exists(PRETRAINED_PATH):
        print(f"🧠 Kvasir ağırlıkları yükleniyor: {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH), strict=False)
    elif USE_TRANSFER_LEARNING:
        print(f"⚠️ UYARI: {PRETRAINED_PATH} bulunamadı! Sıfırdan başlanıyor.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device='cuda')

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        model.train()
        avg_loss = train_fn(train_loader, model, optimizer, None, scaler)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Model kaydedildi! Loss: {avg_loss:.4f}")
        else:
            print(f"Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()