"""Script utama untuk menjalankan pipeline deteksi, penghitungan, dan pelacakan objek."""

import os
import sys
import json
import argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import TrackingConfig
from src.tracker import InstanceTracker
from src.visualizer import Visualizer


def run_pipeline(config: TrackingConfig):
    print("=" * 65)
    print(" PIPELINE DETEKSI DAN PELACAKAN OBJEK VIDEO ")
    print("=" * 65)
    print(f" Video Input     : {config.video_path}")
    print(f" Model           : {config.model_name} (conf={config.conf_threshold}, iou={config.iou_threshold})")
    print(f" Kelas Target    : {config.target_classes}")
    print(f" Folder Output   : {config.output_dir}")
    print("-" * 65)

    if not os.path.exists(config.video_path):
        raise FileNotFoundError(f"File video tidak ditemukan: {config.video_path}")

    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Buka file video
    cap = cv2.VideoCapture(config.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = int(width * config.render_scale)
    out_height = int(height * config.render_scale)

    print(f" Dimensi Video   : {width}x{height} -> Output: {out_width}x{out_height} @ {fps:.1f} FPS")
    print(f" Total Frame     : {total_frames}")

    # Tentukan rentang frame yang akan diproses
    if config.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)
        print(f" Mulai Frame     : {config.start_frame}")

    frames_to_process = total_frames - config.start_frame
    if config.max_frames:
        frames_to_process = min(frames_to_process, config.max_frames)

    # 2. Siapkan VideoWriter untuk menyimpan video hasil anotasi
    video_writer = None
    output_video_path = os.path.join(config.output_dir, config.output_video_name)
    if config.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
        if not video_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    # 3. Inisialisasi modul tracker dan visualizer
    tracker = InstanceTracker(config)
    visualizer = Visualizer(render_scale=config.render_scale)

    # 4. Loop pemrosesan setiap frame video
    print("\nMemproses frame...")
    processed_count = 0
    pbar = tqdm(total=frames_to_process, desc="Progress Pelacakan", unit="frame")

    while cap.isOpened() and processed_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = config.start_frame + processed_count

        # Jalankan deteksi, tracking, dan perhitungan objek aktif/hilang
        frame_summary = tracker.update(frame, current_frame_idx)

        # Gambar anotasi bounding box dan HUD
        annotated_frame = visualizer.draw(frame, frame_summary, tracker.instance_trajectories)

        if video_writer:
            video_writer.write(annotated_frame)

        processed_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\n[SELESAI] Video hasil anotasi disimpan di: {output_video_path}")

    # 5. Simpan metrik per frame ke file CSV
    print("\nMenyimpan metrik dan ringkasan...")
    df_history = pd.DataFrame(tracker.history_records)
    csv_path = os.path.join(config.output_dir, config.metrics_csv_name)
    df_history.to_csv(csv_path, index=False)
    print(f"[SELESAI] CSV metrik per frame disimpan di: {csv_path}")

    # Simpan ringkasan statistik ke file JSON
    summary_stats = tracker.get_summary_statistics()
    json_path = os.path.join(config.output_dir, config.metrics_json_name)
    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"[SELESAI] Ringkasan JSON disimpan di: {json_path}")

    # 6. Buat grafik visualisasi temporal
    if config.save_plots and len(df_history) > 0:
        plot_path = os.path.join(config.output_dir, config.summary_plot_name)
        generate_plots(df_history, summary_stats, plot_path)
        print(f"[SELESAI] Grafik analitik disimpan di: {plot_path}")

    # 7. Tampilkan rekap ringkasan akhir di terminal
    print("\n" + "=" * 65)
    print("                    RINGKASAN HASIL EKSEKUSI                     ")
    print("=" * 65)
    print(f" Total Frame Diproses         : {summary_stats.get('total_frames_processed', 0)}")
    print(f" Total Objek Unik (ID)        : {summary_stats.get('cumulative_unique_instances', 0)}")
    print(f" Objek Aktif per Frame        : Min={summary_stats.get('min_active_per_frame')}, "
          f"Max={summary_stats.get('max_active_per_frame')}, "
          f"Rata-rata={summary_stats.get('avg_active_per_frame', 0):.2f}")
    print(f" Total Kejadian Objek Hilang  : {summary_stats.get('total_lost_events', 0)}")
    print("=" * 65 + "\n")


def generate_plots(df: pd.DataFrame, summary: dict, output_path: str):
    """Membuat grafik temporal multi-panel hasil pelacakan objek."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Panel 1: Jumlah objek aktif per frame
    ax1 = axes[0]
    if "raw_det_count" in df.columns:
        ax1.plot(df["frame_idx"], df["raw_det_count"], color="#aec7e8", lw=1.2, ls=":", label="Deteksi Mentah YOLO")
    ax1.plot(df["frame_idx"], df["active_count"], color="#1f77b4", lw=2, label="Objek Aktif Terlacak")
    ax1.axhline(summary.get("avg_active_per_frame", 0), color="orange", ls="--", label=f"Rata-rata Terlacak ({summary.get('avg_active_per_frame', 0):.1f})")
    ax1.set_ylabel("Jumlah Objek", fontsize=11, fontweight="bold")
    ax1.set_title("Deteksi Mentah YOLO vs Objek Aktif Terlacak per Frame", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: Objek baru dan objek yang hilang per frame
    ax2 = axes[1]
    ax2.bar(df["frame_idx"], df["new_count"], color="#2ca02c", alpha=0.7, label="Objek Baru", width=1.0)
    ax2.bar(df["frame_idx"], -df["lost_count"], color="#d62728", alpha=0.7, label="Objek Hilang", width=1.0)
    ax2.set_ylabel("Kejadian Instance", fontsize=11, fontweight="bold")
    ax2.set_title("Dinamika Objek: Baru vs Hilang per Frame", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Panel 3: Akumulasi total objek unik seiring waktu
    ax3 = axes[2]
    ax3.plot(df["frame_idx"], df["cumulative_unique"], color="#9467bd", lw=2, label="Akumulasi Objek Unik")
    ax3.set_xlabel("Indeks Frame", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Jumlah ID Unik", fontsize=11, fontweight="bold")
    ax3.set_title("Akumulasi Objek Unik Terlacak Seiring Waktu", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Deteksi dan Pelacakan Objek Video")
    parser.add_argument("--video", type=str, default="data/Normal_Videos_780_x264.mp4", help="Jalur file video input")
    parser.add_argument("--start-frame", type=int, default=100, help="Indeks frame mulai")
    parser.add_argument("--max-frames", type=int, default=400, help="Jumlah frame maksimal yang diproses")
    parser.add_argument("--conf", type=float, default=0.15, help="Ambang batas confidence deteksi")
    parser.add_argument("--iou", type=float, default=0.45, help="Ambang batas IoU NMS")
    parser.add_argument("--scale", type=float, default=2.0, help="Faktor perbesaran render")
    parser.add_argument("--no-video", action="store_true", help="Jangan simpan file video output")
    args = parser.parse_args()

    cfg = TrackingConfig(
        video_path=args.video,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        render_scale=args.scale,
        save_video=not args.no_video,
    )

    run_pipeline(cfg)
