"""
metrics.py - Bilgisayarlı Görü ve Renklendirme Değerlendirme Metrikleri
======================================================================
Bu modül, IEEE Access 2026 ve ICIP 2024 (PCF-Net) makalelerindeki nicel
başarım metriklerini (PSNR, SSIM, CIELAB Delta E, LPIPS) ve hata haritalarını hesaplar.
Computer_Vision_Skill standartlarına tam uyumludur.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from skimage.color import rgb2lab, deltaE_cie76
from skimage.metrics import structural_similarity as ssim

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Opsiyonel LPIPS kütüphanesi kontrolü
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class LPIPSMetric(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) Metriği Sarmalayıcısı.
    Sistemde lpips paketi kuruluysa AlexNet tabanlı LPIPS modelini yükler.
    Kurulu değilse sessizce devre dışı kalır ve None döner.
    """
    def __init__(self, net: str = 'alex', device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.available = HAS_LPIPS
        self.model = None

        if self.available:
            try:
                self.model = lpips.LPIPS(net=net).to(self.device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[UYARI] LPIPS modeli yüklenirken hata oluştu: {e}. LPIPS devre dışı bırakıldı.")
                self.available = False

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
        """
        pred, gt: [-1, 1] aralığında (B, 3, H, W) PyTorch tensörleri.
        """
        if not self.available or self.model is None:
            return None
        
        pred_dev = pred.to(self.device)
        gt_dev = gt.to(self.device)
        score = self.model(pred_dev, gt_dev)
        return float(score.mean().item())


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) hesaplar.
    img1, img2: [0, 255] uint8 veya float numpy dizileri (H, W, C).
    """
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mse = np.mean((img1_f - img2_f) ** 2)
    
    if mse == 0.0:
        return 100.0  # Mükemmel eşleşme
    if np.isnan(mse) or np.isinf(mse):
        return 0.0
        
    return float(20.0 * math.log10(max_val / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    3 kanallı RGB görüntüler için Yapısal Benzerlik İndeksi (SSIM) hesaplar.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Görüntü boyutları uyuşmuyor: {img1.shape} vs {img2.shape}")
        
    # Minimum pencere boyutu kontrolü
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim) if min_dim >= 3 else 3
    if win_size % 2 == 0:
        win_size -= 1
        
    score = ssim(
        img1, 
        img2, 
        channel_axis=2 if img1.ndim == 3 else None, 
        data_range=max_val,
        win_size=win_size
    )
    return float(score)


def calculate_delta_e(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """
    CIELAB renk uzayında Renk Farkı (CIE76 Delta E) hesaplar.
    İnsan gözünün renk algısına göre sapmayı ölçer (düşük değer = daha doğru renkler).
    img1_rgb, img2_rgb: [0, 255] uint8 RGB dizileri (H, W, 3).
    """
    # skimage rgb2lab için [0, 1] aralığına normalize edilir
    norm1 = np.clip(img1_rgb.astype(np.float64) / 255.0, 0.0, 1.0)
    norm2 = np.clip(img2_rgb.astype(np.float64) / 255.0, 0.0, 1.0)
    
    lab1 = rgb2lab(norm1)
    lab2 = rgb2lab(norm2)
    delta_e = deltaE_cie76(lab1, lab2)
    
    return float(np.mean(delta_e))


def calculate_error_map(pred_rgb: np.ndarray, gt_rgb: np.ndarray, colormap: bool = True) -> np.ndarray:
    """
    Tahmin ile Ground Truth arasındaki piksel bazlı mutlak hata haritasını üretir.
    pred_rgb, gt_rgb: [0, 255] uint8 RGB dizileri (H, W, 3).
    Dönen: (H, W, 3) uint8 renkli ısı haritası (JET) veya tek kanallı gri tonlama.
    """
    diff = np.abs(pred_rgb.astype(np.float32) - gt_rgb.astype(np.float32))
    err = np.mean(diff, axis=2).clip(0, 255).astype(np.uint8)
    
    if colormap and HAS_CV2:
        return cv2.applyColorMap(err, cv2.COLORMAP_JET)
    elif colormap:
        # Fallback basit RGB ısı haritası (kırmızı = yüksek hata)
        heat = np.zeros((*err.shape, 3), dtype=np.uint8)
        heat[..., 0] = err  # R
        heat[..., 1] = 255 - err  # G
        return heat
    return err


def evaluate_batch(
    pred_tensor: torch.Tensor, 
    gt_tensor: torch.Tensor, 
    lpips_metric: Optional[LPIPSMetric] = None
) -> Dict[str, float]:
    """
    [-1, 1] aralığındaki (B, 3, H, W) PyTorch tensörlerini alarak
    PSNR, SSIM, Delta E (ve varsa LPIPS) metriklerinin batch ortalamalarını döner.
    """
    # [-1, 1] -> [0, 255] uint8 numpy
    pred_np = ((pred_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) * 127.5)
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
    
    gt_np = ((gt_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy() + 1.0) * 127.5)
    gt_np = np.clip(gt_np, 0, 255).astype(np.uint8)
    
    b_size = pred_np.shape[0]
    psnrs, ssims, delta_es = [], [], []
    
    for i in range(b_size):
        psnrs.append(calculate_psnr(pred_np[i], gt_np[i]))
        ssims.append(calculate_ssim(pred_np[i], gt_np[i]))
        delta_es.append(calculate_delta_e(pred_np[i], gt_np[i]))
        
    metrics = {
        'psnr': float(np.mean(psnrs)),
        'ssim': float(np.mean(ssims)),
        'delta_e': float(np.mean(delta_es))
    }
    
    if lpips_metric is not None and lpips_metric.available:
        lpips_val = lpips_metric(pred_tensor, gt_tensor)
        if lpips_val is not None:
            metrics['lpips'] = float(lpips_val)
            
    return metrics


class MetricTracker:
    """
    Tüm validasyon/test seti boyunca metrikleri kümülatif toplayan sınıf.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        
    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if v is not None and not np.isnan(v) and not np.isinf(v):
                self.sums[k] = self.sums.get(k, 0.0) + (v * n)
                self.counts[k] = self.counts.get(k, 0) + n
                
    def summary(self) -> Dict[str, float]:
        return {
            k: float(self.sums[k] / max(self.counts[k], 1))
            for k in self.sums
        }
