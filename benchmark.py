"""
benchmark.py - Donanım Profilleme, Başarım Kıyaslama ve Hata Analizi
===================================================================
IEEE Access 2026 ve ICIP 2024 modelleri için:
- Senkronize CUDA Timer (torch.cuda.Event) ile gecikme (ms) ve FPS ölçümü (min 30 warm-up)
- Parametre (M) ve FLOPs (G) hesaplama
- Tepe VRAM (Peak VRAM in MB) ölçümü
- Test veri seti üzerinde PSNR, SSIM, CIELAB Delta E ve LPIPS değerlendirmesi
- Görsel Hata Analizi (Girdi, GT, Tahmin, Isı Haritası) çıktısı
- Otomatik Markdown Kıyaslama Tablosu üretimi
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import PCFNetN2VStudent, PCFNetTeacher
from dataset import build_dataloader, NIRRGBDataset
from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_delta_e,
    calculate_error_map,
    evaluate_batch,
    MetricTracker,
    LPIPSMetric,
    HAS_LPIPS
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def count_parameters(model: nn.Module) -> Tuple[float, float]:
    """Toplam ve eğitilebilir parametre sayısını Milyon cinsinden döner."""
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total_params, trainable_params


def compute_flops(model: nn.Module, dummy_input: torch.Tensor) -> str:
    """Modelin FLOPs / MACs değerini hesaplar."""
    if not HAS_THOP:
        return "N/A (pip install thop)"
    try:
        # thop profilini sessizce çalıştır
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2.0) / 1e9  # 1 MAC ~= 2 FLOPs
        return f"{gflops:.2f} G"
    except Exception as e:
        return f"Hata ({e})"


def profile_hardware(
    model: nn.Module,
    dummy_input: torch.Tensor,
    warmup: int = 30,
    iters: int = 100,
    device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """
    Computer_Vision_Skill gereksinimlerine göre donanım gecikmesi ve VRAM profillemesi.
    """
    model.eval()
    batch_size = dummy_input.shape[0]
    is_cuda = device.type == 'cuda' and torch.cuda.is_available()

    # Bellek ve Isınma (Warm-up)
    with torch.no_grad():
        if is_cuda:
            for _ in range(warmup):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            
            # Senkronize CUDA Event zamanlaması
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            for i in range(iters):
                start_events[i].record()
                _ = model(dummy_input)
                end_events[i].record()

            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            # CPU Fallback (Hafif test)
            cpu_warmup = min(2, warmup)
            cpu_iters = min(5, iters)
            for _ in range(cpu_warmup):
                _ = model(dummy_input)
            times_ms = []
            for _ in range(cpu_iters):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            peak_vram_mb = 0.0

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    fps = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    return {
        'latency_mean_ms': mean_ms,
        'latency_std_ms': std_ms,
        'fps': fps,
        'peak_vram_mb': peak_vram_mb
    }


def save_visual_error_analysis(
    nir_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    save_path: str,
    title_suffix: str = ""
):
    """
    Görsel Hata Analizi:
    [Girdi NIR (Yalancı Renk) | Tahmin RGB | Ground Truth RGB | Hata Isı Haritası]
    """
    if not HAS_MATPLOTLIB:
        return

    # Tensörleri [0, 255] uint8 formatına dönüştür
    def tensor_to_np(t):
        arr = ((t.detach().cpu().permute(1, 2, 0).float().numpy() + 1.0) * 127.5)
        return np.clip(arr, 0, 255).astype(np.uint8)

    nir_np = tensor_to_np(nir_tensor[:3])
    gt_np = tensor_to_np(gt_tensor)
    pred_np = tensor_to_np(pred_tensor)
    
    # Mutlak fark hata haritası
    err_map = calculate_error_map(pred_np, gt_np, colormap=False)

    psnr_val = calculate_psnr(pred_np, gt_np)
    ssim_val = calculate_ssim(pred_np, gt_np)
    de_val = calculate_delta_e(pred_np, gt_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(nir_np)
    axes[0].set_title("Girdi NIR (3-Bant)", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(pred_np)
    axes[1].set_title(f"Model Tahmini {title_suffix}\nPSNR: {psnr_val:.2f} dB", fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(gt_np)
    axes[2].set_title("Gerçek Renkli (GT RGB)", fontsize=11)
    axes[2].axis('off')

    im_err = axes[3].imshow(err_map, cmap='jet', vmin=0, vmax=100)
    axes[3].set_title(f"Hata Haritası |Pred - GT|\nSSIM: {ssim_val:.3f} | ΔE: {de_val:.2f}", fontsize=11)
    axes[3].axis('off')
    
    cbar = fig.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Piksel Hatası', rotation=270, labelpad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    input_key: str,
    device: torch.device,
    save_visuals_dir: Optional[str] = None,
    max_visuals: int = 5,
    lpips_metric: Optional[LPIPSMetric] = None
) -> Dict[str, float]:
    """Test veri seti üzerinde metrikleri hesaplar ve hata analiz panelleri kaydeder."""
    model.eval()
    tracker = MetricTracker()
    saved_count = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            inp = batch[input_key].to(device)
            rgb = batch['gt_rgb'].to(device)

            out = model(inp)
            pred_rgb = out[0] if isinstance(out, tuple) else out

            batch_metrics = evaluate_batch(pred_rgb, rgb, lpips_metric=lpips_metric)
            tracker.update(batch_metrics, n=inp.size(0))

            # Görsel hata analizi kaydet
            if save_visuals_dir and saved_count < max_visuals:
                for s in range(inp.size(0)):
                    if saved_count >= max_visuals:
                        break
                    save_path = os.path.join(save_visuals_dir, f"sample_{saved_count + 1}.png")
                    save_visual_error_analysis(
                        batch['student_input'][s], rgb[s], pred_rgb[s], save_path,
                        title_suffix=f"#{saved_count + 1}"
                    )
                    saved_count += 1

    return tracker.summary()


def format_markdown_table(results: List[Dict[str, str]]) -> str:
    """Computer_Vision_Skill standardında Markdown kıyaslama tablosu oluşturur."""
    headers = [
        "Model", "Cozunurluk", "Param (M)", "FLOPs", 
        "Gecikme (ms)", "FPS", "Tepe VRAM (MB)", "PSNR (dB)", "SSIM", "Delta E (CIE76)", "LPIPS"
    ]
    
    col_keys = [
        "model", "resolution", "params", "flops", 
        "latency", "fps", "vram", "psnr", "ssim", "delta_e", "lpips"
    ]

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows = []
    for r in results:
        row_str = "| " + " | ".join(str(r.get(k, "-")) for k in col_keys) + " |"
        rows.append(row_str)

    return "\n".join([header_row, separator_row] + rows)


def main():
    parser = argparse.ArgumentParser(description="Model Kıyaslama, Donanım Profilleme ve Hata Analizi")
    parser.add_argument('--data_dir', type=str, default="", help="Değerlendirme veri seti klasörü (opsiyonel)")
    parser.add_argument('--student_weights', type=str, default="", help="Student checkpoint .pth yolu")
    parser.add_argument('--teacher_weights', type=str, default="", help="Teacher checkpoint .pth yolu")
    parser.add_argument('--model', type=str, default="both", choices=['both', 'student', 'teacher'])
    parser.add_argument('--img_h', type=int, default=256, help="Görüntü yüksekliği")
    parser.add_argument('--img_w', type=int, default=256, help="Görüntü genişliği")
    parser.add_argument('--batch_size', type=int, default=1, help="İnference batch boyutu")
    parser.add_argument('--warmup', type=int, default=30, help="Donanım profil warm-up iterasyonu")
    parser.add_argument('--iters', type=int, default=100, help="Donanım profil iterasyonu")
    parser.add_argument('--save_dir', type=str, default="benchmark_results", help="Sonuç ve görsel kayıt klasörü")
    parser.add_argument('--max_visuals', type=int, default=5, help="Kaydedilecek görsel analiz sayısı")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"=== Kıyaslama Başlatılıyor (Cihaz: {device}, Çözünürlük: {args.img_w}x{args.img_h}) ===")

    lpips_eval = LPIPSMetric(device=device) if HAS_LPIPS else None

    # Değerlendirilecek modeller: (Ad, Model Nesnesi, Girdi Kanalı, Veri Kümesi Girdi Anahtarı)
    models_to_test = []
    if args.model in ['both', 'teacher']:
        t_model = PCFNetTeacher(in_channels=4).to(device)
        if args.teacher_weights and os.path.exists(args.teacher_weights):
            t_model.load_state_dict(torch.load(args.teacher_weights, map_location=device))
            print(f"[OK] Teacher ağırlıkları yüklendi: {args.teacher_weights}")
        models_to_test.append(("PCFNet-Teacher (1. Aşama)", t_model, 4, 'teacher_input'))

    if args.model in ['both', 'student']:
        s_model = PCFNetN2VStudent().to(device)
        if args.student_weights and os.path.exists(args.student_weights):
            s_model.load_state_dict(torch.load(args.student_weights, map_location=device))
            print(f"[OK] Student ağırlıkları yüklendi: {args.student_weights}")
        models_to_test.append(("N2V-Student (2. Aşama)", s_model, 3, 'student_input'))

    # Test veri seti yükleyici
    test_loader = None
    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"[OK] Veri seti yükleniyor: {args.data_dir}")
        test_loader = build_dataloader(
            data_dir=args.data_dir,
            image_size=(args.img_h, args.img_w),
            batch_size=args.batch_size,
            is_train=False,
            num_workers=0
        )

    results_table_data = []

    for name, model, in_ch, input_key in models_to_test:
        print(f"\n--- Model Profili: {name} (Giriş Kanalı: {in_ch}) ---")
        dummy_input = torch.randn(args.batch_size, in_ch, args.img_h, args.img_w, device=device)
        
        total_p, train_p = count_parameters(model)
        flops_str = compute_flops(model, dummy_input)
        
        # Donanım Profilleme
        hw_stats = profile_hardware(
            model=model,
            dummy_input=dummy_input,
            warmup=args.warmup,
            iters=args.iters,
            device=device
        )

        res_row = {
            "model": name,
            "resolution": f"{args.img_w}x{args.img_h}",
            "params": f"{total_p:.2f}M",
            "flops": flops_str,
            "latency": f"{hw_stats['latency_mean_ms']:.2f} ± {hw_stats['latency_std_ms']:.2f}",
            "fps": f"{hw_stats['fps']:.1f}",
            "vram": f"{hw_stats['peak_vram_mb']:.1f}" if hw_stats['peak_vram_mb'] > 0 else "N/A"
        }

        # Veri seti değerlendirmesi (varsa)
        if test_loader is not None and len(test_loader) > 0:
            print(f"[{name}] Test veri seti üzerinde metrikler hesaplanıyor...")
            vis_dir = os.path.join(args.save_dir, "visual_analysis", name.replace(" ", "_"))
            eval_metrics = evaluate_dataset(
                model=model,
                dataloader=test_loader,
                input_key=input_key,
                device=device,
                save_visuals_dir=vis_dir,
                max_visuals=args.max_visuals,
                lpips_metric=lpips_eval
            )
            res_row["psnr"] = f"{eval_metrics.get('psnr', 0.0):.2f}"
            res_row["ssim"] = f"{eval_metrics.get('ssim', 0.0):.4f}"
            res_row["delta_e"] = f"{eval_metrics.get('delta_e', 0.0):.2f}"
            res_row["lpips"] = f"{eval_metrics.get('lpips', 0.0):.4f}" if "lpips" in eval_metrics else "-"
        else:
            res_row["psnr"] = "Belirtilmedi"
            res_row["ssim"] = "Belirtilmedi"
            res_row["delta_e"] = "Belirtilmedi"
            res_row["lpips"] = "-"

        results_table_data.append(res_row)

    # Markdown Tablosu Üretimi
    table_md = format_markdown_table(results_table_data)
    print("\n" + "=" * 60)
    print("### KIYASLAMA VE DEĞERLENDİRME SONUÇ TABLOSU")
    print("=" * 60)
    print(table_md)
    print("=" * 60 + "\n")

    report_path = os.path.join(args.save_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Kıyaslama Raporu\n\n")
        f.write(f"- **Tarih / Saat:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Çalışma Cihazı:** `{device}`\n")
        f.write(f"- **Giriş Çözünürlüğü:** {args.img_w}x{args.img_h}\n\n")
        f.write(table_md)
        f.write("\n\n*Not: Bu tablo Computer_Vision_Skill yönergelerine uygun olarak üretilmiştir.*\n")

    print(f"[OK] Kıyaslama raporu kaydedildi: {report_path}")
    if args.data_dir:
        print(f"[OK] Görsel hata analiz panelleri '{args.save_dir}/visual_analysis/' klasörüne kaydedildi.")


if __name__ == '__main__':
    main()
