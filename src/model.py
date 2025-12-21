import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AxialAttention(nn.Module):
    """
    Makalede belirtilen Eksenel Dikkat (Axial Attention) mekanizması.
    Görüntüyü H (Yükseklik) ve W (Genişlik) eksenlerinde ayrı ayrı işleyerek
    işlem yükünü azaltır (Şekil 2).
    """

    def __init__(self, in_channels, dim):
        super(AxialAttention, self).__init__()
        self.q_mlp = nn.Linear(in_channels, dim)
        self.k_mlp = nn.Linear(in_channels, dim)
        self.v_mlp = nn.Linear(in_channels, dim)
        self.proj = nn.Linear(dim, in_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        # Kanal boyutunu (C) sona al: (N, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)

        # Basitleştirilmiş Axial Attention
        # Global özellikleri yakalamak için boyutları düzleştiriyoruz
        flat_x = x_perm.reshape(N, H * W, C)

        q = self.q_mlp(flat_x)
        k = self.k_mlp(flat_x)
        v = self.v_mlp(flat_x)

        # Attention Skoru: Q x K
        attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = self.proj(out)

        # Eski boyutlara geri dön: (N, C, H, W)
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return out + x  # Residual Connection


class PTransformerBlock(nn.Module):
    """
    Paralel Transformer Bloğu (Şekil 1b).
    Bir kolda Transformer (Global), diğer kolda CNN (Yerel) çalışır.
    """

    def __init__(self, in_channels):
        super(PTransformerBlock, self).__init__()
        # 1. Kol: Improved Transformer (Global Bilgi)
        self.transformer = AxialAttention(in_channels, dim=in_channels)

        # 2. Kol: ResBlock (Yerel Bilgi - CNN)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # İki kolu paralel çalıştır
        out_trans = self.transformer(x)  # Global
        out_cnn = self.cnn_branch(x) + x  # Yerel (Residual ile)
        return out_trans, out_cnn


class GLF(nn.Module):
    """
    Global Local Fusing (GLF) Modülü (Şekil 1c).
    Transformer ve CNN çıktılarını akıllıca birleştirir.
    """

    def __init__(self, in_channels):
        super(GLF, self).__init__()
        self.conv_global = nn.Conv2d(in_channels, in_channels, 1)
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, f_global, f_local):
        # Spatial Attention: Global özelliklerin ortalaması
        sa_map = torch.mean(f_global, dim=1, keepdim=True)
        sa_map = torch.sigmoid(sa_map)

        # Channel Attention: Global özelliklerin uzamsal ortalaması
        ca_map = torch.mean(f_global, dim=(2, 3), keepdim=True)
        ca_map = torch.sigmoid(ca_map)

        # Yerel özellikleri filtrele (Local Features Refinement)
        refined_local = f_local * sa_map * ca_map

        # Birleştir (Concatenate)
        f_global_proc = self.conv_global(f_global)
        concat = torch.cat([f_global_proc, refined_local], dim=1)

        return self.final_conv(concat)


class PTransUNet(nn.Module):
    """
    Ana P-TransUNet Modeli (Şekil 1d).
    """

    def __init__(self, num_classes=1):
        super(PTransUNet, self).__init__()

        # Encoder: ResNet-50 (Pretrained)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)  # 256 ch
        self.encoder2 = resnet.layer2  # 512 ch
        self.encoder3 = resnet.layer3  # 1024 ch

        # P-Transformer Blokları
        self.reduce_dim = nn.Conv2d(1024, 512, 1)
        self.ptrans1 = PTransformerBlock(512)
        self.glf1 = GLF(512)

        # Decoder (Bilinear Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(512 + 512, 256, 3, padding=1)  # Skip connection ile birleşince

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = nn.Conv2d(256 + 256, 128, 3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = nn.Conv2d(128, 64, 3, padding=1)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, 1)

        # Kenar Çıkışı (Edge Output) - Makale gereksinimi
        self.edge_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.encoder1(x)  # (N, 256, H/4, W/4)
        x2 = self.encoder2(x1)  # (N, 512, H/8, W/8)
        x3 = self.encoder3(x2)  # (N, 1024, H/16, W/16)

        # Boyut düşürme ve P-Transformer
        x3_red = self.reduce_dim(x3)
        pt_g, pt_l = self.ptrans1(x3_red)
        glf_out = self.glf1(pt_g, pt_l)  # (N, 512, H/16, W/16)

        # --- Decoder ---
        # Adım 1
        d1 = self.up1(glf_out)  # H/8 boyutuna çık
        # Boyut uyuşmazlığı varsa düzelt (Interpolate)
        if d1.size()[2:] != x2.size()[2:]:
            d1 = F.interpolate(d1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x2], dim=1)  # Skip Connection
        d1 = self.conv_up1(d1)

        # Adım 2
        d2 = self.up2(d1)  # H/4 boyutuna çık
        if d2.size()[2:] != x1.size()[2:]:
            d2 = F.interpolate(d2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.conv_up2(d2)

        # Adım 3
        d3 = self.up3(d2)
        d3 = self.conv_up3(d3)

        # Adım 4 (Final)
        out = self.up4(d3)
        # Orijinal giriş boyutuna tam eşitle
        if out.size()[2:] != x.size()[2:]:
            out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)

        seg_out = self.final_conv(out)
        edge_out = self.edge_conv(out)

        return seg_out, edge_out


# --- Test Bloğu ---
if __name__ == "__main__":
    # Modeli test etmek için rastgele bir veri oluştur
    model = PTransUNet()
    # RTX 2060 için örnek girdi: (Batch=2, Channel=3, 256x256)
    dummy_input = torch.randn(2, 3, 256, 256)

    seg, edge = model(dummy_input)
    print(f"Giriş Boyutu: {dummy_input.shape}")
    print(f"Segmentasyon Çıktısı: {seg.shape}")
    print(f"Kenar Çıktısı: {edge.shape}")
    print("✅ Model mimarisi başarıyla kuruldu!")