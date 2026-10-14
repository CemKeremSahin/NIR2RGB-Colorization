"""
train_student.py - 2. Aşama: Öğrenci Deformable Distillation Eğitimi
===================================================================
Kaynak: IEEE Access 2026, Bölüm III-C (Formül 8) & Bölüm IV-A
DL Engineer Standartları:
  - Dondurulmuş Öğretmen (Teacher Weights Frozen, no_grad)
  - 5 Encoder Seviyesinde Ofset & Maske Distilasyonu (alpha = 0.02)
  - Mixed Precision Training (torch.cuda.amp.autocast & GradScaler)
  - Gradient Clipping (torch.nn.utils.clip_grad_norm_)
  - Cosine Annealing Learning Rate Scheduler
  - Hızlı Doğrulama Modu (--sanity_check)
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import PCFNetTeacher, PCFNetN2VStudent
from losses import StudentLoss
from dataset import build_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sanity_check(student, teacher, criterion, device, batch, use_amp):
    """
    DL Engineer Standardı: Öğrenci modelinin dondurulmuş öğretmen eşliğinde
    tek bir batch üzerinde overfit olduğunu ve loss'un azaldığını doğrular.
    """
    print("\n[Sanity Check] Öğrenci distilasyonu için tek batch overfit testi (30 adım)...")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    t_in = batch['teacher_input'].to(device)
    s_in = batch['student_input'].to(device)
    gt_rgb = batch['gt_rgb'].to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, t_offsets, t_masks = teacher(t_in)
            
    initial_loss = None
    final_loss = None
    for step in range(30):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            s_out_rgb, s_offsets, s_masks = student(s_in)
            loss, _, _, _ = criterion(s_out_rgb, gt_rgb, s_offsets, s_masks, t_offsets, t_masks)
            
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"[Sanity Check] İlk Loss: {initial_loss:.4f} -> 30. Adım Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Öğrenci modeli tek batch'e overfit olamadı, distilasyon veya öğrenci mimarisini kontrol edin!"
    print("[Sanity Check BAŞARILI] Deformable distillation ve geri yayılım kusursuz çalışıyor.\n")


def main():
    parser = argparse.ArgumentParser(description="2. Aşama: Student Deformable Distillation Eğitimi")
    parser.add_argument("--data_dir", type=str, required=True, help="Veri setinin bulunduğu klasör yolu")
    parser.add_argument("--teacher_weights", type=str, default="teacher_best.pth", help="Öğretmen model ağırlık dosyası")
    parser.add_argument("--epochs", type=int, default=4000, help="Eğitim epoch sayısı (Makale: 4000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch boyutu (Makale: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Öğrenme oranı (Makale: 0.0001)")
    parser.add_argument("--alpha", type=float, default=0.02, help="Deformable distillation katsayısı (Makale: 0.02)")
    parser.add_argument("--img_h", type=int, default=256, help="Görüntü yüksekliği")
    parser.add_argument("--img_w", type=int, default=256, help="Görüntü genişliği")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker sayısı")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping eşiği")
    parser.add_argument("--seed", type=int, default=42, help="Tekrarlanabilirlik tohumu")
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed Precision (AMP) kullan")
    parser.add_argument("--sanity_check", action="store_true", help="Eğitim öncesi hızlı overfit testi yap")
    parser.add_argument("--save_path", type=str, default="student_best.pth", help="Öğrenci model ağırlıklarının kaydedileceği yol")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    print(f"[Aşama 2] Cihaz: {device} | Mixed Precision (AMP): {use_amp}")

    # 1. Dondurulmuş Öğretmen Modelini Yükle
    teacher = PCFNetTeacher(in_channels=4, channels=[72, 144, 288, 480, 720]).to(device)
    if os.path.exists(args.teacher_weights):
        teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
        print(f"[Aşama 2] Öğretmen ağırlıkları başarıyla yüklendi: '{args.teacher_weights}'")
    else:
        raise FileNotFoundError(f"[Hata] Öğretmen ağırlık dosyası bulunamadı: '{args.teacher_weights}'! Lütfen önce train_teacher.py çalıştırın.")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. Öğrenci Modeli, Kayıp, Optimizer, Scheduler ve GradScaler
    student = PCFNetN2VStudent(private_channels=[24, 48, 96, 160, 240]).to(device)
    criterion = StudentLoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 3. Veri Yükleyici
    dataloader = build_dataloader(
        data_dir=args.data_dir,
        image_size=(args.img_h, args.img_w),
        batch_size=args.batch_size,
        is_train=True,
        num_workers=args.num_workers
    )

    # 4. İsteğe Bağlı Sanity Check
    if args.sanity_check:
        sample_batch = next(iter(dataloader))
        run_sanity_check(student, teacher, criterion, device, sample_batch, use_amp)
        # Modeli sıfırdan yeniden ilklendir
        student = PCFNetN2VStudent(private_channels=[24, 48, 96, 160, 240]).to(device)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[Aşama 2] Öğrenci Deformable Distillation eğitimi başlıyor... Toplam {args.epochs} epoch (alpha={args.alpha}).")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        student.train()
        total_loss, total_l1, total_vgg, total_deform = 0.0, 0.0, 0.0, 0.0
        
        for batch in dataloader:
            t_in = batch['teacher_input'].to(device)
            s_in = batch['student_input'].to(device)
            gt_rgb = batch['gt_rgb'].to(device)
            
            # Öğretmenden referans 5 seviyeli ofset ve maskeleri çıkar (gradyansız)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, t_offsets, t_masks = teacher(t_in)
                
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                s_out_rgb, s_offsets, s_masks = student(s_in)
                loss, loss_l1, loss_vgg, loss_deform = criterion(
                    s_out_rgb, gt_rgb, s_offsets, s_masks, t_offsets, t_masks
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_vgg += loss_vgg.item()
            total_deform += loss_deform.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)
        avg_vgg = total_vgg / len(dataloader)
        avg_deform = total_deform / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Student Epoch [{epoch+1:04d}/{args.epochs}] -> Total: {avg_loss:.4f} | L1: {avg_l1:.4f} | VGG: {avg_vgg:.4f} | Deform: {avg_deform:.4f} | LR: {current_lr:.6f}")

        # En iyi modeli kaydet
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), args.save_path)

    print(f"[Aşama 2 Tamamlandı] En iyi öğrenci modeli '{args.save_path}' konumuna kaydedildi (En düşük kayıp: {best_loss:.4f}).")


if __name__ == "__main__":
    main()
