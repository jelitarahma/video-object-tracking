# Video Object Detection and Tracking

## Cara Clone Repository

```bash
git clone https://github.com/jelitarahma/video-object-tracking.git
cd video-object-tracking
```

## Instalasi

1. Pastikan Python sudah terinstall (versi 3.9 ke atas).
2. Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Pastikan file video input berada di dalam folder `data/` dan file model `yolov8m.pt` berada di folder utama.

## Cara Menjalankan

Jalankan perintah berikut untuk memproses video:

```bash
python main.py
```

Perintah di atas akan memproses video mulai dari frame 100 sebanyak 400 frame, lalu menyimpan hasilnya ke folder `outputs/`:
- `annotated_tracked_video.mp4`: video hasil pelacakan objek
- `per_frame_counts.csv`: log jumlah objek per frame
- `tracking_metrics.json`: ringkasan metrik pelacakan
- `tracking_analysis_plot.png`: grafik analisis temporal

## Pilihan Argumen

Argumen berikut dapat ditambahkan saat menjalankan `main.py`:

- `--video`: lokasi file video input (default: `data/Normal_Videos_780_x264.mp4`)
- `--start-frame`: indeks frame awal pemrosesan (default: `100`)
- `--max-frames`: jumlah frame yang diproses (default: `400`)
- `--conf`: batas confidence deteksi YOLO (default: `0.15`)
- `--iou`: batas IoU NMS (default: `0.45`)
- `--scale`: skala perbesaran video hasil anotasi (default: `2.0`)
- `--no-video`: hanya menyimpan data CSV, JSON, dan grafik tanpa menyimpan video output

Contoh perintah:

```bash
# Menjalankan dari frame 0 sebanyak 200 frame
python main.py --start-frame 0 --max-frames 200

# Menjalankan tanpa menyimpan video (hanya ambil data dan grafik)
python main.py --no-video

# Mengubah threshold confidence deteksi
python main.py --conf 0.25
```
