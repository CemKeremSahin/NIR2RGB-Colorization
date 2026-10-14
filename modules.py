"""
modules.py - N2V ve PCF-Net Temel Yapı Taşları
=============================================
Kaynaklar:
  - IEEE Access 2026: DA-DCN, Deformable Distillation, N2V Modülü (Formül 1-5, Şekil 2)
  - ICIP 2024: PCF-Net Private Module, Cross-Fusion Module, Gate Mechanism (Formül 1-4, Şekil 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# 8 Pusula Yönü (Makale Formül 1: Merkez (0,0) hariç 3x3 komşuluk)
COMPASS_SHIFTS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]


def conv3(cin, cout):
    """3x3 Konvolüsyon + BatchNorm + PReLU aktivasyonu."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.PReLU()
    )


class DoubleConv(nn.Module):
    """PCF-Net Cross-Fusion içinde kullanılan çift evrişim bloğu (ICIP 2024 Şekil 3)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            conv3(cin, cout),
            conv3(cout, cout)
        )

    def forward(self, x):
        return self.block(x)


class DenseConv(nn.Module):
    """DenseConv Bloğu: Yoğun bağlantılı çok ölçekli öznitelik çıkarımı (Şekil 3)."""
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(channels, growth_rate, 3, padding=1), nn.PReLU())
        self.c2 = nn.Sequential(nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1), nn.PReLU())
        self.c3 = nn.Sequential(nn.Conv2d(channels + 2 * growth_rate, channels, 1), nn.PReLU())

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(torch.cat([x, c1], dim=1))
        out = self.c3(torch.cat([x, c1, c2], dim=1))
        return out + x


# ==============================================================================
# 1. DA-DCN: Distance-Assisted Deformable Convolution (IEEE Access 2026 Eq. 1-3)
# ==============================================================================
class DADCNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        assert channels % 8 == 0, f"DA-DCN için kanal sayısı 8'e tam bölünmelidir (Alınan: {channels})"
        self.num_heads = channels // 8
        self.head_dim = 8
        
        # 3x3 DCNv2: 9 örnekleme noktası -> 18 ofset koordinatı (2x9), 9 modülasyon maskesi
        self.conv_offset = nn.Conv2d(channels, 18, kernel_size=3, padding=1)
        self.conv_mask = nn.Conv2d(channels, 9, kernel_size=3, padding=1)
        
        # Sıfır ilklendirme (DCNv2 standardı)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)
        
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.bias = nn.Parameter(torch.zeros(channels))

    def compute_distance_map(self, x):
        """Formül (1): 8 pusula yönü boyunca Öklid mesafesi hesaplar."""
        B, C, H, W = x.shape
        x_heads = x.view(B, self.num_heads, self.head_dim, H, W)
        dist_list = []
        
        for dy, dx in COMPASS_SHIFTS:
            pad_left = max(0, -dx)
            pad_right = max(0, dx)
            pad_top = max(0, -dy)
            pad_bottom = max(0, dy)
            
            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            y_start = pad_top + dy
            x_start = pad_left + dx
            neighbor = x_padded[:, :, y_start:y_start+H, x_start:x_start+W]
            neighbor_heads = neighbor.view(B, self.num_heads, self.head_dim, H, W)
            
            # 8 kanal üzerinden L2 normu (+1e-8 ile sıfır bölme / NaN önleme)
            dist = torch.sqrt(torch.sum((x_heads - neighbor_heads) ** 2, dim=2) + 1e-8)
            dist_list.append(dist)
            
        # num_heads * 8 = C kanal (B x C x H x W)
        return torch.cat(dist_list, dim=1)

    def forward(self, x):
        D = self.compute_distance_map(x)
        offset = self.conv_offset(D)
        mask = torch.sigmoid(self.conv_mask(D))
        out = deform_conv2d(x, offset, self.weight, self.bias, stride=1, padding=1, mask=mask)
        return out, offset, mask


# ==============================================================================
# 2. N2V Modülü (IEEE Access 2026 Şekil 2, Eq. 4-5)
# ==============================================================================
class N2VModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1. Aşama: DA-DCN (Intra-NIR ayrıntı çıkarma)
        self.stage1_dadcn = DADCNBlock(channels)
        
        # 2. Aşama: Deformable Distillation için öğrenci ofset ve maske kestiricisi
        self.conv_offset_s = nn.Conv2d(channels, 18, kernel_size=3, padding=1)
        self.conv_mask_s = nn.Conv2d(channels, 9, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_offset_s.weight)
        nn.init.zeros_(self.conv_offset_s.bias)
        nn.init.zeros_(self.conv_mask_s.weight)
        nn.init.zeros_(self.conv_mask_s.bias)
        
        self.weight_s = nn.Parameter(torch.empty(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight_s, a=1)
        self.bias_s = nn.Parameter(torch.zeros(channels))
        
        # 1x1 Fusion Head: X (giriş) + Y (1. aşama) + Z (2. aşama) birleşimi
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x):
        # 1. Aşama (Intra-NIR)
        Y, _, _ = self.stage1_dadcn(x)
        
        # 2. Aşama (Deformable Distillation)
        offset_s = self.conv_offset_s(Y)
        mask_s = torch.sigmoid(self.conv_mask_s(Y))
        Z = deform_conv2d(Y, offset_s, self.weight_s, self.bias_s, stride=1, padding=1, mask=mask_s)
        
        # 1x1 Füzyon
        out = self.fuse(torch.cat([x, Y, Z], dim=1))
        return out, offset_s, mask_s


# ==============================================================================
# 3. PCF-Net Cross-Fusion ve Private Modülleri (ICIP 2024 Şekil 3, Eq. 1-4)
# ==============================================================================
class CrossFusionModule(nn.Module):
    """RFBNet Gate Mekanizmalı Çapraz Füzyon Bloğu (ICIP 2024 Eq. 3-4)."""
    def __init__(self, c_prev_cross, c_private):
        super().__init__()
        self.c_private = c_private
        c_cross = 3 * c_private
        
        # Her NIR bant özniteliği için regülasyon kapısı (Gate Regulation)
        self.gate1 = nn.Sequential(nn.Conv2d(3 * c_private, c_private, kernel_size=3, padding=1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(3 * c_private, c_private, kernel_size=3, padding=1), nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(3 * c_private, c_private, kernel_size=3, padding=1), nn.Sigmoid())
        
        in_channels = c_prev_cross + c_cross
        self.double_conv = DoubleConv(in_channels, c_cross)

    def forward(self, x_prev_cross, x_mid_1, x_mid_2, x_mid_3):
        cat_mids = torch.cat([x_mid_1, x_mid_2, x_mid_3], dim=1)
        reg1 = x_mid_1 * self.gate1(cat_mids)
        reg2 = x_mid_2 * self.gate2(cat_mids)
        reg3 = x_mid_3 * self.gate3(cat_mids)
        x_reg = torch.cat([reg1, reg2, reg3], dim=1)
        
        if x_prev_cross is not None:
            if x_prev_cross.shape[2:] != x_reg.shape[2:]:
                x_prev_cross = F.interpolate(x_prev_cross, size=x_reg.shape[2:], mode='bilinear', align_corners=False)
            fusion_in = torch.cat([x_prev_cross, x_reg], dim=1)
        else:
            fusion_in = x_reg
            
        x_cross = self.double_conv(fusion_in)
        return x_cross
