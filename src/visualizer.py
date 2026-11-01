"""Modul visualisasi untuk menampilkan anotasi objek, lintasan, dan status HUD pada frame."""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class Visualizer:
    """Menggambar anotasi bounding box, lintasan gerak, dan HUD banner pada frame."""

    def __init__(self, render_scale: float = 2.0):
        self.render_scale = render_scale
        # Palet warna BGR untuk membedakan ID objek
        self.colors = [
            (0, 230, 115),   # Hijau Zamrud
            (255, 128, 0),   # Biru Terang
            (0, 165, 255),   # Oranye
            (204, 0, 255),   # Ungu Neon
            (0, 255, 255),   # Cyan
            (255, 51, 153),  # Magenta
            (51, 204, 255),  # Biru Muda
            (102, 255, 51),  # Hijau Terang
            (255, 204, 0),   # Emas
            (0, 102, 255),   # Oranye Tua
            (204, 255, 102), # Hijau Muda
            (255, 102, 102), # Merah Koral
        ]

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]

    def draw(
        self,
        frame: np.ndarray,
        frame_summary: Dict[str, Any],
        trajectories: Dict[int, List[Tuple[int, int]]]
    ) -> np.ndarray:
        """Menambahkan visualisasi bounding box, badge ID, dan dashboard status ke frame."""
        orig_h, orig_w = frame.shape[:2]
        scale = self.render_scale
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Perbesar ukuran frame untuk meningkatkan kejelasan visual
        annotated = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        active_instances = frame_summary.get("active_instances", {})
        active_count = frame_summary.get("active_count", 0)
        lost_ids = frame_summary.get("lost_ids", [])
        new_ids = frame_summary.get("new_ids", [])
        cumulative_count = frame_summary.get("cumulative_unique_count", 0)
        frame_idx = frame_summary.get("frame_idx", 0)

        # 1. Gambar lintasan gerak objek
        for tid, points in trajectories.items():
            if tid in active_instances and len(points) > 1:
                color = self._get_color(tid)
                scaled_pts = [(int(pt[0] * scale), int(pt[1] * scale)) for pt in points]
                for i in range(1, len(scaled_pts)):
                    p1 = scaled_pts[i - 1]
                    p2 = scaled_pts[i]
                    if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 25.0 * scale:
                        cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)

        # 2. Gambar bounding box dan label badge
        for tid, inst in active_instances.items():
            bbox = inst["bbox"]
            x1 = int(bbox[0] * scale)
            y1 = int(bbox[1] * scale)
            x2 = int(bbox[2] * scale)
            y2 = int(bbox[3] * scale)
            color = self._get_color(tid)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            corner_len = min(10, int((x2 - x1) * 0.2))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

            # Label badge: "<class> #ID"
            cls_name = inst.get("class_name", "")
            label = f"{cls_name} #{tid}" if cls_name else f"#{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            if y1 - th - 6 < 68:
                badge_y1 = y1 + 2
                badge_y2 = badge_y1 + th + 4
            else:
                badge_y1 = y1 - th - 6
                badge_y2 = badge_y1 + th + 4
            badge_x2 = min(new_w, x1 + tw + 6)

            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, badge_y2 - 3),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # 3. Tampilkan dashboard HUD di bagian atas
        banner_h = 65
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (new_w, banner_h), (16, 16, 20), -1)
        cv2.addWeighted(hud_overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.line(annotated, (0, banner_h), (new_w, banner_h), (60, 60, 70), 1)

        # Baris atas HUD: Nomor frame, jumlah objek aktif, dan total ID unik
        cv2.putText(
            annotated,
            f"FRAME: {frame_idx:04d}",
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

        active_color = (0, 255, 128) if active_count >= 5 else (0, 180, 255)
        cv2.putText(
            annotated,
            f"ACTIVE OBJECTS: {active_count}",
            (170, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            active_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"TOTAL UNIQUE: {cumulative_count}",
            (440, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 215, 0),
            1,
            cv2.LINE_AA
        )

        # Baris bawah HUD: ID baru dan ID yang hilang
        new_text = f"NEW: {new_ids if new_ids else '-'}"
        cv2.putText(
            annotated,
            new_text,
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (120, 255, 120),
            1,
            cv2.LINE_AA
        )

        lost_text = f"LOST IN FRAME: {lost_ids if lost_ids else 'None'}"
        lost_color = (60, 60, 255) if lost_ids else (160, 160, 160)
        cv2.putText(
            annotated,
            lost_text,
            (220, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            lost_color,
            2 if lost_ids else 1,
            cv2.LINE_AA
        )

        # Indikator badge merah jika terdapat objek yang hilang pada frame ini
        if lost_ids:
            badge_text = f"! LOST #{lost_ids} !"
            (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            bx1 = new_w - bw - 20
            by1 = 36
            cv2.rectangle(annotated, (bx1 - 4, by1), (bx1 + bw + 4, by1 + bh + 8), (0, 0, 200), -1)
            cv2.putText(
                annotated,
                badge_text,
                (bx1, by1 + bh + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated
