# Google Colab Çalıştırma Rehberi: N2V Colorization

Bu rehber, **IEEE Access 2026** ve **ICIP 2024 (PCF-Net)** mimarilerini Google Colab (T4 / V100 / A100 GPU) üzerinde en yüksek performansla eğitmeniz ve kıyaslamanız için hazırlanmıştır.

---

## 1. Hazırlık ve Google Drive Yüklemesi

1. Masaüstünüzdeki `C:\Users\Kerem\Desktop\NIR2RGB\N2V_Colorization` klasörünü Google Drive'ınıza yükleyin.
   Önerilen konum:
   `MyDrive/NIR2RGB/N2V_Colorization/`
2. Veri setinizin (NIR ve RGB çiftleri) Drive'da hazır olduğundan emin olun:
   ```text
   NIR_Dataset/
   ├── nir/
   │   ├── 0001.png
   │   └── ...
   └── rgb/
       ├── 0001.png
       └── ...
   ```

---

## 2. Google Colab Hücreleri (Sırayla Çalıştırın)

### Hücre 1: GPU Kontrolü ve Google Drive Bağlama
```python
# GPU türünü kontrol edin (T4, V100 veya A100 olmalı)
!nvidia-smi

# Google Drive'ı bağlayın
from google.colab import drive
drive.mount('/content/drive')
```

### Hücre 2: Proje Klasörüne Geçiş ve Gerekli Paketlerin Kurulumu
```python
import os
# Proje dizinine geçiş yapın
%cd /content/drive/MyDrive/NIR2RGB/N2V_Colorization

# CV Metrikleri ve Profilleme için gerekli kütüphaneleri kurun
!pip install -q scikit-image thop lpips torchvision
```

### Hücre 3: Sanity Check (Hızlı Doğrulama - 10 Saniye)
Eğitime başlamadan önce boru hattının, DCNv2 modüllerinin ve veri akışının hatasız çalıştığını doğrulamak için tek batch'lik overfit testini çalıştırın:
```python
# Teacher ve Student için hızlı test
!python train_teacher.py --data_dir /content/drive/MyDrive/NIR_Dataset --sanity_check --epochs 1 --amp
!python train_student.py --data_dir /content/drive/MyDrive/NIR_Dataset --sanity_check --epochs 1 --amp
```

### Hücre 4: 1. Aşama - Öğretmen Ağı Eğitimi (Teacher Training)
Öğretmen modeli, 3-Bant NIR + 1-Bant Grayscale kılavuzluğuyla eğitilir:
```python
!python train_teacher.py \
    --data_dir /content/drive/MyDrive/NIR_Dataset \
    --epochs 1000 \
    --batch_size 8 \
    --lr 0.0001 \
    --amp \
    --save_path teacher_best.pth
```
*(Not: `teacher_best.pth` ağırlıkları aynı klasöre kaydedilecektir.)*

### Hücre 5: 2. Aşama - Öğrenci Modeli ve Deformable Distillation Eğitimi
Öğrenci modeli, yalnızca 3-Bant NIR girdi alarak öğretmenin 5 kademedeki DCNv2 geometrik ofsetlerini öğrenir:
```python
!python train_student.py \
    --data_dir /content/drive/MyDrive/NIR_Dataset \
    --teacher_weights teacher_best.pth \
    --epochs 4000 \
    --batch_size 8 \
    --lr 0.0001 \
    --alpha 0.02 \
    --amp \
    --save_path student_best.pth
```

### Hücre 6: Kıyaslama, Donanım Profilleme ve Görsel Hata Analizi (Benchmark)
Eğitim tamamlandıktan sonra PSNR, SSIM, Delta E, LPIPS metriklerini hesaplamak, FPS / VRAM / FLOPs ölçmek ve hata ısı haritalarını üretmek için:
```python
!python benchmark.py \
    --data_dir /content/drive/MyDrive/NIR_Dataset \
    --teacher_weights teacher_best.pth \
    --student_weights student_best.pth \
    --model both \
    --warmup 30 \
    --iters 100 \
    --batch_size 1 \
    --save_dir benchmark_results
```

### Hücre 7: Sonuç Tablosu ve Hata Analiz Panellerini Görüntüleme
```python
from IPython.display import Markdown, display, Image
import glob

# 1. Markdown Raporunu Göster
with open("benchmark_results/benchmark_report.md", "r", encoding="utf-8") as f:
    display(Markdown(f.read()))

# 2. Üretilen Görsel Hata Haritalarını Göster
visuals = glob.glob("benchmark_results/visual_analysis/*/*.png")
for v in visuals:
    print(f"Görsel: {v}")
    display(Image(v))
```
