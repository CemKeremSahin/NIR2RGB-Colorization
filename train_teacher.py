"""
train_teacher.py - 1. Aşama: Öğretmen Modelinin Eğitimi
======================================================
Kaynak: IEEE Access 2026, Bölüm III-C (Formül 7) & Bölüm IV-A
DL Engineer Standartları:
  - Mixed Precision Training (torch.cuda.amp.autocast & GradScaler)
  - Gradient Clipping (torch.nn.utils.clip_grad_norm_)
  - Cosine Annealing Learning Rate Scheduler
  - Deterministik Tohum (Fixed Seed)
  - Hızlı Doğrulama Modu (--sanity_check)
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import PCFNetTeacher
from losses import TeacherLoss
from dataset import build_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sanity_check(model, criterion, device, batch):
    """
    DL Engineer Standardı: Binlerce epoch çalıştırmadan önce tek bir batch
    üzerinde modelin hatasız çalıştığını ve overfit olabildiğini doğrular.
    """
    print("\n[Sanity Check] Tek batch üzerinde overfit testi başlatılıyor (30 adım)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_in = batch['teacher_input'].to(device)
    gt_rgb = batch['gt_rgb'].to(device)
    
    initial_loss = None
    final_loss = None
    for step in range(30):
        optimizer.zero_grad()
        pred_rgb, _, _ = model(t_in)
        loss, _, _ = criterion(pred_rgb, gt_rgb)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"[Sanity Check] İlk Loss: {initial_loss:.4f} -> 30. Adım Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Model tek batch'e overfit olamadı, mimari veya kayıp fonksiyonunu kontrol edin!"
    print("[Sanity Check BAŞARILI] Gradyan akışı ve optimizasyon kusursuz çalışıyor.\n")


def main():
    parser = argparse.ArgumentParser(description="1. Aşama: Teacher Eğitimi")
    parser.add_argument("--data_dir", type=str, required=True, help="Veri setinin bulunduğu klasör yolu")
    parser.add_argument("--epochs", type=int, default=1000, help="Eğitim epoch sayısı (Makale: 1000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch boyutu (Makale: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Öğrenme oranı (Makale: 0.0001)")
    parser.add_argument("--img_h", type=int, default=256, help="Görüntü yüksekliği")
    parser.add_argument("--img_w", type=int, default=256, help="Görüntü genişliği")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker sayısı")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping eşiği")
    parser.add_argument("--seed", type=int, default=42, help="Tekrarlanabilirlik tohumu")
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed Precision (AMP) kullan")
    parser.add_argument("--sanity_check", action="store_true", help="Eğitim öncesi hızlı overfit testi yap")
    parser.add_argument("--save_path", type=str, default="teacher_best.pth", help="Model ağırlıklarının kaydedileceği yol")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    print(f"[Aşama 1] Cihaz: {device} | Mixed Precision (AMP): {use_amp}")

    # 1. Veri Yükleyici (data-pipeline standardı)
    dataloader = build_dataloader(
        data_dir=args.data_dir,
        image_size=(args.img_h, args.img_w),
        batch_size=args.batch_size,
        is_train=True,
        num_workers=args.num_workers
    )

    # 2. Model, Kayıp, Optimizer, Scheduler ve GradScaler
    teacher = PCFNetTeacher(in_channels=4, channels=[72, 144, 288, 480, 720]).to(device)
    criterion = TeacherLoss().to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 3. İsteğe Bağlı Sanity Check
    if args.sanity_check:
        sample_batch = next(iter(dataloader))
        run_sanity_check(teacher, criterion, device, sample_batch)
        # Modeli sıfırdan yeniden ilklendir
        teacher = PCFNetTeacher(in_channels=4, channels=[72, 144, 288, 480, 720]).to(device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[Aşama 1] Öğretmen eğitimi başlıyor... Toplam {args.epochs} epoch.")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        teacher.train()
        total_loss, total_l1, total_vgg = 0.0, 0.0, 0.0
        
        for batch in dataloader:
            t_in = batch['teacher_input'].to(device)
            gt_rgb = batch['gt_rgb'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_rgb, _, _ = teacher(t_in)
                loss, loss_l1, loss_vgg = criterion(pred_rgb, gt_rgb)
            
            # AMP GradScaler ile geri yayılım ve gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_vgg += loss_vgg.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)
        avg_vgg = total_vgg / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Teacher Epoch [{epoch+1:04d}/{args.epochs}] -> Total: {avg_loss:.4f} | L1: {avg_l1:.4f} | VGG: {avg_vgg:.4f} | LR: {current_lr:.6f}")

        # En iyi modeli kaydet
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(teacher.state_dict(), args.save_path)

    print(f"[Aşama 1 Tamamlandı] En iyi model '{args.save_path}' konumuna kaydedildi (En düşük kayıp: {best_loss:.4f}).")


if __name__ == "__main__":
    main()
