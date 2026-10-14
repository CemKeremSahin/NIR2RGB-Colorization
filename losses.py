"""
losses.py - Kayıp Fonksiyonları (IEEE Access 2026, Bölüm III-C, Eq. 6-10)
========================================================================
- L1 Kaybı (Manhattan Mesafesi, Eq. 9)
- VGG Algısal Kayıp (Perceptual Loss, Eq. 10)
- Deformable Distillation Kaybı (Ofset ve Maske L1 Mesafesi, Eq. 6)
- Öğretmen Toplam Kaybı: L^T = L1 + Lvgg (Eq. 7)
- Öğrenci Toplam Kaybı: L^S = L1 + Lvgg + alpha * L_deform (Eq. 8, alpha = 0.02)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Önceden eğitilmiş VGG-16 katmanları üzerinden L2 (MSE) mesafesi hesaplar (Eq. 10).
    Katmanlar: relu1_2, relu2_2, relu3_3, relu4_3
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).eval()    # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).eval() # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).eval()# relu3_3
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(16, 23)]).eval()# relu4_3
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, gt):
        # Tanh çıkışı [-1, 1] aralığını [0, 1] aralığına çevir
        p = (pred + 1.0) / 2.0
        g = (gt + 1.0) / 2.0
        
        p = (p - self.mean) / self.std
        g = (g - self.mean) / self.std
        
        h_p1 = self.slice1(p); h_g1 = self.slice1(g)
        h_p2 = self.slice2(h_p1); h_g2 = self.slice2(h_g1)
        h_p3 = self.slice3(h_p2); h_g3 = self.slice3(h_g2)
        h_p4 = self.slice4(h_p3); h_g4 = self.slice4(h_g3)
        
        loss = F.mse_loss(h_p1, h_g1) + \
               F.mse_loss(h_p2, h_g2) + \
               F.mse_loss(h_p3, h_g3) + \
               F.mse_loss(h_p4, h_g4)
        return loss


def deformable_distillation_loss(s_offsets, s_masks, t_offsets, t_masks):
    """
    Formül (6): L_deform = ||Δp_s - Δp_t||_1 + ||Δm_s - Δm_t||_1
    5 encoder kademesinin tamamındaki ofset ve maske kayıplarını toplar (Şekil 3).
    """
    total_loss = 0.0
    for s in range(len(s_offsets)):
        loss_offset = F.l1_loss(s_offsets[s], t_offsets[s])
        loss_mask = F.l1_loss(s_masks[s], t_masks[s])
        total_loss = total_loss + (loss_offset + loss_mask)
    return total_loss


class TeacherLoss(nn.Module):
    """L^T = L1 + Lvgg (Eq. 7)"""
    def __init__(self):
        super().__init__()
        self.vgg = VGGPerceptualLoss()

    def forward(self, pred_rgb, gt_rgb):
        loss_l1 = F.l1_loss(pred_rgb, gt_rgb)
        loss_vgg = self.vgg(pred_rgb, gt_rgb)
        return loss_l1 + loss_vgg, loss_l1, loss_vgg


class StudentLoss(nn.Module):
    """L^S = L1 + Lvgg + alpha * L_deform (Eq. 8, alpha = 0.02)"""
    def __init__(self, alpha=0.02):
        super().__init__()
        self.alpha = alpha
        self.vgg = VGGPerceptualLoss()

    def forward(self, pred_rgb, gt_rgb, s_offsets, s_masks, t_offsets, t_masks):
        loss_l1 = F.l1_loss(pred_rgb, gt_rgb)
        loss_vgg = self.vgg(pred_rgb, gt_rgb)
        loss_deform = deformable_distillation_loss(s_offsets, s_masks, t_offsets, t_masks)
        total_loss = loss_l1 + loss_vgg + (self.alpha * loss_deform)
        return total_loss, loss_l1, loss_vgg, loss_deform
