"""
dataset.py - NIR ve RGB Veri Seti Yükleyicisi
============================================
Data Pipeline Specialist Standartları:
  1. I/O ve GPU Açlığı (Starvation) Önleme: build_dataloader ile pin_memory, prefetch_factor, optimize worker'lar.
  2. Savunmacı Mühendislik: NaN/Inf kontrolleri, bozuk dosya koruması, sıkı tip denetimi.
  3. Bellek Karmaşıklığı: O(1) disk akışı (RAM'i şişirmez, diskten dinamik yükler).
  4. Ters Normalizasyon: Model öncesi tensörlerin doğruluğunu teyit eden denormalize aracı.
"""

import os
import glob
from typing import Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class NIRRGBDataset(Dataset):
    """
    3-Bant NIR (785nm, 850nm, 940nm) ve GT RGB görüntü çiftlerini yükler.
    RGB'den Grayscale (Y) türeterek Öğretmen ağı için 4 kanallı girdi üretir.
    Tensörleri [-1, 1] aralığına normalize eder.
    """
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # 1. Dosya Çiftlerini Güvenli Tarama
        self.pairs: List[Tuple[str, str]] = self._find_image_pairs(data_dir)
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"[Data Pipeline Error] '{data_dir}' dizininde eşleşen NIR ve GT görsel çifti bulunamadı!")
            
        print(f"[Dataset] Yüklendi: {len(self.pairs)} çift | Eğitim Modu: {self.is_train} | Çözünürlük: {self.image_size}")

    def _find_image_pairs(self, data_dir: str) -> List[Tuple[str, str]]:
        gt_files = sorted(glob.glob(os.path.join(data_dir, "**", "*GROUND_TRUTH*.tif"), recursive=True))
        if len(gt_files) == 0:
            gt_files = sorted(glob.glob(os.path.join(data_dir, "**", "*gt*.png"), recursive=True))
            
        pairs = []
        for gt_path in gt_files:
            nir_candidate = gt_path.replace("GROUND_TRUTH", "NIR_INPUT")
            if not os.path.exists(nir_candidate):
                nir_candidate = gt_path.replace("gt", "nir")
            if os.path.exists(nir_candidate):
                pairs.append((nir_candidate, gt_path))
                
        # Eğer özel anahtar kelimeler yoksa tüm görselleri eşle
        if len(pairs) == 0:
            all_files = sorted(glob.glob(os.path.join(data_dir, "*.*")))
            valid = [f for f in all_files if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
            pairs = [(f, f) for f in valid]
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nir_path, gt_path = self.pairs[idx]
        
        try:
            # 1. RGB GT Yükleme
            gt_img = Image.open(gt_path).convert('RGB')
            
            # 2. NIR Yükleme (Tek kanallıysa 3 kanala genişlet)
            nir_img = Image.open(nir_path)
            if nir_img.mode != 'RGB':
                nir_img = nir_img.convert('RGB')
        except Exception as e:
            raise IOError(f"[Data Pipeline] Görsel okuma hatası ({nir_path} veya {gt_path}): {str(e)}")

        # 3. Çözünürlük Uyarlaması
        gt_img = gt_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        nir_img = nir_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # 4. Veri Artırma (Yalnızca eğitimde rastgele yatay çevirme)
        if self.is_train and torch.rand(1).item() > 0.5:
            gt_img = TF.hflip(gt_img)
            nir_img = TF.hflip(nir_img)

        # 5. Normalizasyon: [0, 255] -> [-1.0, 1.0] (Tanh aktivasyonuna uygun)
        gt_tensor = TF.to_tensor(gt_img) * 2.0 - 1.0   # [3, H, W]
        nir_tensor = TF.to_tensor(nir_img) * 2.0 - 1.0 # [3, H, W]

        # Savunmacı Mühendislik: NaN veya Sonsuz değer kontrolü
        if torch.isnan(gt_tensor).any() or torch.isnan(nir_tensor).any():
            raise ValueError(f"[Data Pipeline Error] NaN tensör tespit edildi! Dosya: {gt_path}")
        if torch.isinf(gt_tensor).any() or torch.isinf(nir_tensor).any():
            raise ValueError(f"[Data Pipeline Error] Inf tensör tespit edildi! Dosya: {gt_path}")

        # 6. RGB'den Grayscale türet: Y = 0.299 R + 0.587 G + 0.114 B (Teacher için görünür alan geometrisi)
        gray_tensor = 0.299 * gt_tensor[0:1, :, :] + 0.587 * gt_tensor[1:2, :, :] + 0.114 * gt_tensor[2:3, :, :]

        # 7. Çıktı Paketleri
        # Öğretmen Girdisi: [NIR1, NIR2, NIR3, Gray] -> 4 Kanal
        teacher_input = torch.cat([nir_tensor, gray_tensor], dim=0) # [4, H, W]
        # Öğrenci Girdisi: [NIR1, NIR2, NIR3] -> 3 Kanal
        student_input = nir_tensor                                    # [3, H, W]

        return {
            'teacher_input': teacher_input,
            'student_input': student_input,
            'gt_rgb': gt_tensor
        }

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """
        [-1, 1] aralığındaki PyTorch tensörünü [0, 255] uint8 RGB numpy dizisine dönüştürür.
        Görsel teyit ve doğrulama için kullanılır.
        """
        t = tensor.detach().cpu().squeeze(0) if tensor.dim() == 4 else tensor.detach().cpu()
        np_img = ((t.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return np_img


def build_dataloader(
    data_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    is_train: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    GPU açlığını engelleyen, pin_memory ve prefetch_factor destekli DataLoader oluşturucu.
    """
    dataset = NIRRGBDataset(data_dir=data_dir, image_size=image_size, is_train=is_train)
    use_cuda = torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        prefetch_factor=2 if (use_cuda and num_workers > 0) else None,
        drop_last=is_train
    )
