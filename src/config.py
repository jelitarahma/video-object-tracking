from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrackingConfig:
    """Konfigurasi parameter pipeline deteksi dan pelacakan objek."""

    # Jalur file video dan folder output
    video_path: str = "data/Normal_Videos_780_x264.mp4"
    output_dir: str = "outputs"
    output_video_name: str = "annotated_tracked_video.mp4"
    metrics_json_name: str = "tracking_metrics.json"
    metrics_csv_name: str = "per_frame_counts.csv"
    summary_plot_name: str = "tracking_analysis_plot.png"

    # Parameter detektor YOLO
    model_name: str = "yolov8m.pt"
    conf_threshold: float = 0.15
    iou_threshold: float = 0.45
    device: str = "cpu"

    # Algoritma tracker dan kelas target yang dilacak
    tracker_type: str = "bytetrack.yaml"
    target_classes: Optional[List[str]] = field(
        default_factory=lambda: ["person", "car", "motorcycle", "bus", "truck", "bicycle"]
    )

    # Parameter pemulihan objek hilang
    enable_reid: bool = True
    max_lost_buffer: int = 75  # Batas frame maksimal memori objek hilang disimpan
    reid_spatial_thresh: float = 110.0
    reid_hist_thresh: float = 0.60

    # Pengaturan frame dan skala rendering video
    start_frame: int = 100
    max_frames: Optional[int] = 400
    render_scale: float = 2.0  # Skala pembesaran render output (640x480)
    save_video: bool = True
    save_plots: bool = True
