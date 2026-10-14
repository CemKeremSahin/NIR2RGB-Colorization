"""
dataset.py - Multi-Band NIR & RGB Veri Yükleyici (Data Pipeline)
==============================================================
Bu modül, hem 3 ayrı dalga boyu klasörünü (785nm, 850nm, 940nm + gt_rgb)
hem de standart (nir + rgb) klasör yapılarını otomatik olarak algılar ve yükler.
IEEE Access 2026 ve data-pipeline-specialist standartlarına tam uyumludur.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


class NIRRGBDataset(Dataset):
    """
    Multi-Band NIR ve RGB Veri Kümesi Sınıfı.
    
    Desteklenen Klasör Yapıları:
    1. 4-Sensör Yapısı (Önerilen / Gerçek Veri):
       data_dir/
       ├── 785nm/   (Kanal 1)
       ├── 850nm/   (Kanal 2)
       ├── 940nm/   (Kanal 3)
       └── gt_rgb/  (veya rgb/) (Hedef Renkli Görüntü)
       
    2. 2-Klasörlü Klasik Yapı:
       data_dir/
       ├── nir/     (3-Bant Birleşik)
       └── rgb/     (Hedef Renkli Görüntü)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True
    ):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.image_size = image_size  # (H, W)
        self.is_train = is_train

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"[Data Pipeline] Belirtilen veri klasörü bulunamadı: {self.data_dir}")

        # 1. Klasör Modunu Otomatik Tespit Et
        # 4-Sensör modunu kontrol et (785nm, 850nm, 940nm)
        path_785 = os.path.join(self.data_dir, "785nm")
        path_850 = os.path.join(self.data_dir, "850nm")
        path_940 = os.path.join(self.data_dir, "940nm")

        if os.path.isdir(path_785) and os.path.isdir(path_850) and os.path.isdir(path_940):
            self.mode = "multiband_folders"
            self.dir_785 = path_785
            self.dir_850 = path_850
            self.dir_940 = path_940
            
            # RGB klasörü tespiti (gt_rgb veya rgb)
            if os.path.isdir(os.path.join(self.data_dir, "gt_rgb")):
                self.dir_rgb = os.path.join(self.data_dir, "gt_rgb")
            elif os.path.isdir(os.path.join(self.data_dir, "rgb")):
                self.dir_rgb = os.path.join(self.data_dir, "rgb")
            else:
                raise FileNotFoundError(f"[Data Pipeline] {self.data_dir} içinde 'gt_rgb' veya 'rgb' klasörü bulunamadı!")

            ref_dir = self.dir_785
        else:
            # Standart 2-klasörlü mod (nir + rgb)
            self.mode = "standard_folders"
            self.dir_nir = os.path.join(self.data_dir, "nir")
            self.dir_rgb = os.path.join(self.data_dir, "rgb")
            
            if not os.path.isdir(self.dir_nir) or not os.path.isdir(self.dir_rgb):
                raise FileNotFoundError(
                    f"[Data Pipeline] Desteklenen klasör yapısı bulunamadı! "
                    f"Ya '785nm', '850nm', '940nm', 'gt_rgb' klasörleri ya da 'nir', 'rgb' klasörleri olmalıdır. "
                    f"Taranan dizin: {self.data_dir}"
                )
            ref_dir = self.dir_nir

        # 2. Dosya Listelerini Eşleştir
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        all_files = sorted([f for f in os.listdir(ref_dir) if os.path.splitext(f)[1].lower() in valid_exts])
        
        # Sadece RGB klasöründe de karşılığı olan dosyaları tut
        self.filenames = [f for f in all_files if os.path.exists(os.path.join(self.dir_rgb, f))]

        if len(self.filenames) == 0:
            raise RuntimeError(f"[Data Pipeline Hatası] {self.data_dir} dizininde eşleşen hiçbir NIR ve RGB görseli bulunamadı!")

        print(f"[Data Pipeline] Mod: '{self.mode}' | Toplam {len(self.filenames)} eşleşen görsel yüklendi.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]
        rgb_path = os.path.join(self.dir_rgb, fname)

        try:
            # 1. RGB Hedef Görüntüyü Yükle
            gt_img = Image.open(rgb_path).convert('RGB')
            
            # 2. NIR Görüntülerini Yükle
            if self.mode == "multiband_folders":
                # 3 ayrı dalga boyunu tek kanallı (L - Grayscale) olarak oku
                img_785 = Image.open(os.path.join(self.dir_785, fname)).convert('L')
                img_850 = Image.open(os.path.join(self.dir_850, fname)).convert('L')
                img_940 = Image.open(os.path.join(self.dir_940, fname)).convert('L')
                
                # Boyut uyarlaması
                img_785 = img_785.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                img_850 = img_850.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                img_940 = img_940.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            else:
                nir_img = Image.open(os.path.join(self.dir_nir, fname))
                if nir_img.mode != 'RGB':
                    nir_img = nir_img.convert('RGB')
                nir_img = nir_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            gt_img = gt_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        except Exception as e:
            raise IOError(f"[Data Pipeline] Görsel okuma hatası ({fname}): {str(e)}")

        # 3. Senkronize Veri Artırma (Eğitimde yatay çevirme)
        if self.is_train and torch.rand(1).item() > 0.5:
            gt_img = TF.hflip(gt_img)
            if self.mode == "multiband_folders":
                img_785 = TF.hflip(img_785)
                img_850 = TF.hflip(img_850)
                img_940 = TF.hflip(img_940)
            else:
                nir_img = TF.hflip(nir_img)

        # 4. Normalizasyon: [0, 255] -> [-1.0, 1.0] (Tanh aktivasyonuna uygun)
        gt_tensor = TF.to_tensor(gt_img) * 2.0 - 1.0  # [3, H, W]

        if self.mode == "multiband_folders":
            t_785 = TF.to_tensor(img_785) * 2.0 - 1.0  # [1, H, W]
            t_850 = TF.to_tensor(img_850) * 2.0 - 1.0  # [1, H, W]
            t_940 = TF.to_tensor(img_940) * 2.0 - 1.0  # [1, H, W]
            nir_tensor = torch.cat([t_785, t_850, t_940], dim=0)  # [3, H, W]
        else:
            nir_tensor = TF.to_tensor(nir_img) * 2.0 - 1.0  # [3, H, W]

        # 5. Savunmacı Mühendislik: NaN veya Sonsuz değer kontrolü
        if torch.isnan(gt_tensor).any() or torch.isnan(nir_tensor).any():
            raise ValueError(f"[Data Pipeline Error] NaN tensör tespit edildi! Dosya: {fname}")
        if torch.isinf(gt_tensor).any() or torch.isinf(nir_tensor).any():
            raise ValueError(f"[Data Pipeline Error] Inf tensör tespit edildi! Dosya: {fname}")

        # 6. RGB'den Grayscale Kılavuzu: Y = 0.299 R + 0.587 G + 0.114 B (Teacher geometrisi)
        gray_tensor = 0.299 * gt_tensor[0:1, :, :] + 0.587 * gt_tensor[1:2, :, :] + 0.114 * gt_tensor[2:3, :, :]

        # 7. Model Çıktı Paketleri
        # Öğretmen Girdisi: [785nm, 850nm, 940nm, Grayscale] -> 4 Kanal
        teacher_input = torch.cat([nir_tensor, gray_tensor], dim=0)  # [4, H, W]
        # Öğrenci Girdisi: [785nm, 850nm, 940nm] -> 3 Kanal
        student_input = nir_tensor                                    # [3, H, W]

        return {
            'teacher_input': teacher_input,
            'student_input': student_input,
            'gt_rgb': gt_tensor
        }

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """
        [-1, 1] tensörünü [0, 255] uint8 RGB numpy dizisine dönüştürür.
        """
        arr = (tensor.detach().cpu().permute(1, 2, 0).float().numpy() + 1.0) * 127.5
        return np.clip(arr, 0, 255).astype(np.uint8)


def build_dataloader(
    data_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    is_train: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    GPU Starvation (GPU açlığı) önleyici yapılandırılmış DataLoader.
    """
    dataset = NIRRGBDataset(
        data_dir=data_dir,
        image_size=image_size,
        is_train=is_train
    )

    is_cuda = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=is_cuda,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_train
    )

    return loader
