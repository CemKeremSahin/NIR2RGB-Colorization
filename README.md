# Multi-Band NIR to RGB Colorization via Deformable Distillation & Cross-Fusion

Official PyTorch implementation of the two-stage Multi-Band Near-Infrared (NIR) to RGB colorization framework based on:
1. **IEEE Access 2026:** *"Overcoming the Domain Gap Between Multi-Band NIR and RGB for Colorization"* (Youm & Kim)
2. **ICIP 2024:** *"Cross-Fusion of Band-Specific Spectral Features for Multi-Band NIR Colorization"* (PCF-Net)

---

## 📌 Mimari Genel Bakış (Architecture Overview)

```
[Stage 1: Teacher Network (4-Ch: 3 NIR + 1 Grayscale)]
                  ↓ 
       DCNv2 Offsets & Modulations (5 Stages)
                  ↓  (α * L_deform distillation)
[Stage 2: Student Network (3-Ch: 3 NIR Bands)]
                  ↓
       Decoder Cross-Fusion with RFBNet Gate Regulation
                  ↓
       High-Fidelity RGB Color Reconstruction
```

- **Öğretmen Ağı (Teacher Network):** 3 NIR bandı + 1 Grayscale (RGB GT parlaklık kılavuzu) olmak üzere 4 kanal girdi alır. Görünür alan geometrisi ile zenginleştirilmiş 5 kademeli DCNv2 ofset ve modülasyon maskeleri üretir.
- **Öğrenci Ağı (Student Network):** Yalnızca 3 NIR bandı girdi alarak test zamanında bağımsız çalışır. 5 encoder kademesinde öğretmenin DCNv2 deformasyon geometrisini distilasyon kaybı ($\alpha \cdot \mathcal{L}_{deform}$) ile taklit eder.
- **Çapraz Füzyon Kod Çözücü (Cross-Fusion Decoder):** Bantlar arası spektral bağımlılıkları korumak için RFBNet geçit düzenlemesi (Gate Regulation) ile zenginleştirilmiştir.

---

## 🚀 Hızlı Başlangıç (Quickstart)

### 1. Kurulum (Installation)
```bash
git clone https://github.com/<kullanici_adi>/NIR2RGB-N2V-Colorization.git
cd NIR2RGB-N2V-Colorization
pip install -r requirements.txt
```

### 2. Veri Seti Yapısı (Dataset Structure)
Veri setinizin aşağıdaki formatta düzenlenmesi önerilir:
```text
data/
├── nir/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── rgb/
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

---

## 🏋️ Eğitim (Training)

### Aşama 1: Öğretmen Modeli Eğitimi (Teacher)
```bash
python train_teacher.py \
    --data_dir ./data \
    --epochs 1000 \
    --batch_size 8 \
    --lr 0.0001 \
    --amp \
    --save_path teacher_best.pth
```

### Aşama 2: Öğrenci Modeli ve Deformable Distilasyon (Student)
```bash
python train_student.py \
    --data_dir ./data \
    --teacher_weights teacher_best.pth \
    --epochs 4000 \
    --batch_size 8 \
    --lr 0.0001 \
    --alpha 0.02 \
    --amp \
    --save_path student_best.pth
```

---

## 📊 Kıyaslama ve Donanım Profilleme (Benchmarking)

`Computer_Vision_Skill` standartlarına tam uyumlu kıyaslama betiği:
- Senkronize CUDA Event zamanlaması (min 30 iterasyon warm-up sonrası hassas gecikme ve FPS).
- Tepe VRAM (Peak VRAM), Parametre ve FLOPs hesabı.
- PSNR, SSIM, CIELAB $\Delta E$ ve LPIPS metrikleri.
- Görsel hata analizi (Piksel bazlı JET ısı haritası panelleri).
- Otomatik Markdown kıyaslama tablosu üretimi.

```bash
python benchmark.py \
    --data_dir ./data \
    --teacher_weights teacher_best.pth \
    --student_weights student_best.pth \
    --model both \
    --warmup 30 \
    --iters 100 \
    --batch_size 1 \
    --save_dir benchmark_results
```

---

## ⚡ Google Colab Tek Komutla Çalıştırma

Google Colab GPU oturumunda (T4/V100/A100):
```python
# 1. Repoyu klonlayın
!git clone https://github.com/<kullanici_adi>/NIR2RGB-N2V-Colorization.git
%cd NIR2RGB-N2V-Colorization

# 2. Paketleri kurun
!pip install -q -r requirements.txt

# 3. Eğitimi başlatın
!python train_teacher.py --data_dir /path/to/dataset --amp
```
