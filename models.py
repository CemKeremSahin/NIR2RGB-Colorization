"""
models.py - Öğretmen (Teacher) ve Öğrenci (Student) Ağ Mimarileri
==================================================================
Kaynaklar:
  - IEEE Access 2026: Şekil 3 (Genel Öğretmen-Öğrenci Çerçevesi, 3x3 Son Evrişim)
  - ICIP 2024: Şekil 2 ve 3 (PCF-Net Hem Encoder Hem Decoder'da Private + Cross Branch, Bilinear UNet Omurgası)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from modules import conv3, DenseConv, N2VModule, CrossFusionModule


# ==============================================================================
# 1. ÖĞRENCİ AĞI (Student Network): PCF-Net + 5 Kademeli N2V
# ==============================================================================
class PCFNetN2VStudent(nn.Module):
    """
    Girdi: 3-Bant NIR [B, 3, H, W] (785nm, 850nm, 940nm)
    Çıktı:
      - out_rgb: Tahmin edilen RGB görüntü [B, 3, H, W]
      - offsets_list: 5 encoder kademesindeki öğrenci ofsetleri (18 kanal)
      - masks_list: 5 encoder kademesindeki öğrenci modülasyon maskeleri (9 kanal)
    """
    def __init__(self, private_channels=[24, 48, 96, 160, 240]):
        super().__init__()
        self.num_stages = len(private_channels)
        self.private_channels = private_channels
        
        # ------------------ KODLAYICI (ENCODER) ------------------
        self.enc_p1_conv1 = nn.ModuleList()
        self.enc_p1_conv2 = nn.ModuleList()
        self.enc_p2_conv1 = nn.ModuleList()
        self.enc_p2_conv2 = nn.ModuleList()
        self.enc_p3_conv1 = nn.ModuleList()
        self.enc_p3_conv2 = nn.ModuleList()
        
        self.enc_cross = nn.ModuleList()
        self.enc_dense = nn.ModuleList()
        self.enc_n2v = nn.ModuleList()
        
        for i, c in enumerate(private_channels):
            c_in = 1 if i == 0 else private_channels[i-1]
            
            # Özel Dal 1. Evrişimi (ICIP 2024 Formül 1)
            self.enc_p1_conv1.append(conv3(c_in, c))
            self.enc_p2_conv1.append(conv3(c_in, c))
            self.enc_p3_conv1.append(conv3(c_in, c))
            
            # Çapraz Füzyon Bloğu (ICIP 2024 Formül 3-4)
            c_prev_cross = 0 if i == 0 else 3 * private_channels[i-1]
            self.enc_cross.append(CrossFusionModule(c_prev_cross, c))
            
            # DenseConv + N2V Modülü (IEEE Access 2026 Şekil 3)
            c_cross = 3 * c
            self.enc_dense.append(DenseConv(c_cross))
            self.enc_n2v.append(N2VModule(c_cross))
            
            # Özel Dal 2. Evrişimi: N2V ile rafine edilen çapraz öznitelikleri entegre eder (Formül 2)
            self.enc_p1_conv2.append(conv3(c * 2, c))
            self.enc_p2_conv2.append(conv3(c * 2, c))
            self.enc_p3_conv2.append(conv3(c * 2, c))
            
        # ------------------ KOD ÇÖZÜCÜ (DECODER) ------------------
        # ICIP 2024: Decoder'da da Private ve Cross Branch simetrik olarak bulunur.
        # "UNet with bilinear interpolation was used as a backbone" (Bölüm 4)
        self.dec_p1_conv1 = nn.ModuleList()
        self.dec_p1_conv2 = nn.ModuleList()
        self.dec_p2_conv1 = nn.ModuleList()
        self.dec_p2_conv2 = nn.ModuleList()
        self.dec_p3_conv1 = nn.ModuleList()
        self.dec_p3_conv2 = nn.ModuleList()
        self.dec_cross = nn.ModuleList()
        
        for i in reversed(range(self.num_stages - 1)):
            c_high = private_channels[i+1]
            c_low = private_channels[i]
            
            # Decoder Private Conv1: Bilinear upsampled (c_high) + encoder skip (c_low) -> c_low
            self.dec_p1_conv1.append(conv3(c_high + c_low, c_low))
            self.dec_p2_conv1.append(conv3(c_high + c_low, c_low))
            self.dec_p3_conv1.append(conv3(c_high + c_low, c_low))
            
            # Decoder Cross-Fusion (Encoder Cross'una bağlı DEĞİLDİR, bağımsız ardışık akıştır)
            is_first_dec = (i == self.num_stages - 2)
            c_prev_dec_cross = 0 if is_first_dec else 3 * private_channels[i+1]
            self.dec_cross.append(CrossFusionModule(c_prev_dec_cross, c_low))
            
            # Decoder Private Conv2: mid (c_low) + cross_subset (c_low) -> c_low
            self.dec_p1_conv2.append(conv3(c_low * 2, c_low))
            self.dec_p2_conv2.append(conv3(c_low * 2, c_low))
            self.dec_p3_conv2.append(conv3(c_low * 2, c_low))
            
        # Son Çıkış Evrişimi: IEEE Access 2026 Şekil 3'te yeşil "3x3 conv" olarak gösterilmiştir.
        self.final_conv = nn.Sequential(
            nn.Conv2d(private_channels[0] * 3, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        n1, n2, n3 = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        p1, p2, p3 = n1, n2, n3
        p1_skips, p2_skips, p3_skips = [], [], []
        
        x_cross = None
        offsets_list, masks_list = [], []
        
        # 5 Kademeli Kodlayıcı İleri Geçişi
        for i in range(self.num_stages):
            if i > 0:
                p1 = F.max_pool2d(p1, 2)
                p2 = F.max_pool2d(p2, 2)
                p3 = F.max_pool2d(p3, 2)
                
            m1 = self.enc_p1_conv1[i](p1)
            m2 = self.enc_p2_conv1[i](p2)
            m3 = self.enc_p3_conv1[i](p3)
            
            # Çapraz Füzyon
            x_cross_raw = self.enc_cross[i](x_cross, m1, m2, m3)
            # DenseConv + N2V (Aşama 1 DA-DCN + Aşama 2 Deformable Distillation)
            x_cross_dense = self.enc_dense[i](x_cross_raw)
            x_cross_n2v, off_s, msk_s = self.enc_n2v[i](x_cross_dense)
            
            offsets_list.append(off_s)
            masks_list.append(msk_s)
            x_cross = x_cross_n2v
            
            # N2V çıktısını 3 özel dala paylaştır
            c = self.private_channels[i]
            c1, c2, c3 = x_cross[:, 0:c], x_cross[:, c:2*c], x_cross[:, 2*c:3*c]
            
            p1 = self.enc_p1_conv2[i](torch.cat([m1, c1], dim=1))
            p2 = self.enc_p2_conv2[i](torch.cat([m2, c2], dim=1))
            p3 = self.enc_p3_conv2[i](torch.cat([m3, c3], dim=1))
            
            if i < self.num_stages - 1:
                p1_skips.append(p1)
                p2_skips.append(p2)
                p3_skips.append(p3)

        # Kod Çözücü İleri Geçişi (Bilinear Upsampling + Private Skips + Decoder Cross Branch)
        x_dec_cross = None
        for idx in range(self.num_stages - 1):
            skip_idx = -(idx + 1)
            target_size = p1_skips[skip_idx].shape[2:]
            
            # 1. Bilinear Upsampling
            u1 = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
            u2 = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
            u3 = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
            
            # 2. Decoder Private Conv1 (Upsampled + Skip)
            dm1 = self.dec_p1_conv1[idx](torch.cat([u1, p1_skips[skip_idx]], dim=1))
            dm2 = self.dec_p2_conv1[idx](torch.cat([u2, p2_skips[skip_idx]], dim=1))
            dm3 = self.dec_p3_conv1[idx](torch.cat([u3, p3_skips[skip_idx]], dim=1))
            
            # 3. Decoder Cross-Fusion
            x_dec_cross = self.dec_cross[idx](x_dec_cross, dm1, dm2, dm3)
            
            c_dec = dm1.shape[1]
            dc1, dc2, dc3 = x_dec_cross[:, 0:c_dec], x_dec_cross[:, c_dec:2*c_dec], x_dec_cross[:, 2*c_dec:3*c_dec]
            
            # 4. Decoder Private Conv2 (Mid + Cross)
            p1 = self.dec_p1_conv2[idx](torch.cat([dm1, dc1], dim=1))
            p2 = self.dec_p2_conv2[idx](torch.cat([dm2, dc2], dim=1))
            p3 = self.dec_p3_conv2[idx](torch.cat([dm3, dc3], dim=1))
            
        out_rgb = self.final_conv(torch.cat([p1, p2, p3], dim=1))
        return out_rgb, offsets_list, masks_list


# ==============================================================================
# 2. ÖĞRETMEN AĞI (Teacher Network): 4 Kanal Giriş + 5 Kademeli DCNv2
# ==============================================================================
class PCFNetTeacher(nn.Module):
    """
    Girdi: 3-Bant NIR + 1-Bant Grayscale [B, 4, H, W]
    Çıktı:
      - out_rgb: Tahmin edilen RGB görüntü [B, 3, H, W]
      - offsets_list: 5 encoder kademesindeki öğretmen ofsetleri (18 kanal)
      - masks_list: 5 encoder kademesindeki öğretmen modülasyon maskeleri (9 kanal)
    """
    def __init__(self, in_channels=4, channels=[72, 144, 288, 480, 720]):
        super().__init__()
        self.num_stages = len(channels)
        self.in_conv = conv3(in_channels, channels[0])
        
        self.enc_dense = nn.ModuleList()
        self.enc_dcn_offs = nn.ModuleList()
        self.enc_dcn_mask = nn.ModuleList()
        self.enc_dcn_weight = nn.ParameterList()
        self.enc_dcn_bias = nn.ParameterList()
        self.downsamples = nn.ModuleList()
        
        for i, c in enumerate(channels):
            self.enc_dense.append(DenseConv(c))
            
            # Öğretmenin DCNv2 ofset ve maske üreticisi
            off_conv = nn.Conv2d(c, 18, 3, padding=1)
            mask_conv = nn.Conv2d(c, 9, 3, padding=1)
            nn.init.zeros_(off_conv.weight); nn.init.zeros_(off_conv.bias)
            nn.init.zeros_(mask_conv.weight); nn.init.zeros_(mask_conv.bias)
            self.enc_dcn_offs.append(off_conv)
            self.enc_dcn_mask.append(mask_conv)
            
            w = nn.Parameter(torch.empty(c, c, 3, 3))
            nn.init.kaiming_uniform_(w, a=1)
            self.enc_dcn_weight.append(w)
            self.enc_dcn_bias.append(nn.Parameter(torch.zeros(c)))
            
            if i < self.num_stages - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(c, channels[i+1], 4, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.PReLU()
                ))
                
        # Kod Çözücü (Bilinear Interpolation + Skip Connection)
        self.dec_convs = nn.ModuleList()
        for i in reversed(range(self.num_stages - 1)):
            c_high = channels[i+1]
            c_low = channels[i]
            self.dec_convs.append(conv3(c_high + c_low, c_low))
            
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.in_conv(x)
        skips = []
        offsets_list, masks_list = [], []
        
        for i in range(self.num_stages):
            feat = self.enc_dense[i](feat)
            off = self.enc_dcn_offs[i](feat)
            msk = torch.sigmoid(self.enc_dcn_mask[i](feat))
            feat = deform_conv2d(feat, off, self.enc_dcn_weight[i], self.enc_dcn_bias[i], stride=1, padding=1, mask=msk)
            
            offsets_list.append(off)
            masks_list.append(msk)
            
            if i < self.num_stages - 1:
                skips.append(feat)
                feat = self.downsamples[i](feat)
                
        for idx in range(self.num_stages - 1):
            skip_idx = -(idx + 1)
            target_size = skips[skip_idx].shape[2:]
            u = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = self.dec_convs[idx](torch.cat([u, skips[skip_idx]], dim=1))
            
        out_rgb = self.out_conv(feat)
        return out_rgb, offsets_list, masks_list
