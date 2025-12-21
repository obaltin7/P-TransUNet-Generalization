# P-TransUNet Validation Project

Bu proje, "P-TransUNet: an improved parallel network for medical image segmentation" (Chong et al., 2023) makalesinde önerilen yöntemin PyTorch ile doğrulanması ve yeniden uygulanması amacıyla geliştirilmiştir.

## 🎯 Proje Amacı
* P-TransUNet mimarisini (P-Transformer ve GLF modülleri) kodlamak.
* Kvasir-SEG veri seti üzerinde polip segmentasyon başarısını test etmek.
* Makalede sunulan sonuçlarla karşılaştırmalı rapor hazırlamak.

## 📂 Klasör Yapısı
* `src/`: Model mimarisi, veri yükleyici ve yardımcı kodlar.
* `data/`: Kvasir-SEG veri seti (GitHub'a yüklenmemiştir, manuel eklenmelidir).
* `output/`: Test sonuçları ve görseller.

## 🚀 Kurulum
1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/obaltin7/P-TransUNetValidation.git](https://github.com/obaltin7/P-TransUNetValidation.git)

## 📂 Dataset
This project uses the **Kvasir-SEG** dataset. Due to licensing and size constraints, the dataset is not included in this repository.

Please download the dataset from the official website:
- **Official Link:** [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)

**Instructions:**
1. Download `Kvasir-SEG.zip` from the link above.
2. Extract the contents inside the `data` folder.
3. Organize the directory structure as follows:

```text
P-TransUNetValidation/
├── data/
│   ├── images/
│   │   ├── cju0qkwl35piu0993l0dewei2.jpg
│   │   └── ...
│   └── masks/
│       ├── cju0qkwl35piu0993l0dewei2.jpg
│       └── ...
```

## 🚀 Kullanım (Usage)

### 1. Eğitimi Başlatma (Training)
Modeli eğitmek için aşağıdaki komutu çalıştırın:
```bash
python train.py
```
Bu işlem model ağırlıklarını saved_models/best_model.pth olarak kaydedecektir.

Donanım: RTX 2060 (6GB) veya üzeri GPU önerilir.

Not: Eğitim parametrelerini (Batch size, Epoch vb.) train.py dosyasının içinden değiştirebilirsiniz.

### 2. Test ve Doğrulama (Testing)
Eğitilen modeli test etmek ve metrikleri (Dice, IoU, Precision, Recall) hesaplamak için:

```bash
python test.py
```

## 📊 Sonuçlar (Results)
Bu proje kapsamında yapılan deneylerde, orijinal makalede sunulan sonuçlar doğrulanmış ve optimize edilen eğitim stratejileri (Mixed Precision, OHEM Loss vb.) sayesinde daha yüksek başarı oranları elde edilmiştir:

| Metrik | Doğrulama Sonucu  | Makale Sonucu (Referans) |
| :--- |:------------------| :--- |
| **mDice** | **0.9798**        | 0.9352 |
| **mIoU** | **0.9609**        | 0.8893 |
| **Recall** | **0.9742**        | 0.9389 |
| **Precision**| **0.9860**        | 0.9379 |

> **Not:** Sonuçlar NVIDIA RTX 2060 donanımı üzerinde, rastgele ayrılmış %10 test seti (random split) kullanılarak elde edilmiştir.

## ⚖️ License & Citation
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This is an **unofficial** implementation of P-TransUNet. The original paper and architecture ideas belong to the respective authors. If you use this code for your research, please cite the original paper:

Chong, Yan-Wen & Xie, Ningdi & Liu, Xin & Pan, Shaoming. (2023). P-TransUNet: an improved parallel network for medical image segmentation. BMC Bioinformatics. 24. 10.1186/s12859-023-05409-7.
