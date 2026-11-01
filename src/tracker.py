"""Modul pelacakan objek (Multi-Object Tracking) dan identifikasi objek hilang."""

from typing import List, Dict, Any, Set, Tuple, Optional
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from .config import TrackingConfig


def compute_iou_matrix(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Menghitung matriks Intersection-over-Union (IoU) antar dua kumpulan bounding box."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((0, 0), dtype=float)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection + 1e-6

    return intersection / union


class SingleTrack:
    """Representasi state temporal untuk satu objek yang sedang dilacak."""

    def __init__(self, track_id: int, bbox: np.ndarray, conf: float, class_id: int, class_name: str):
        self.id = track_id
        self.bbox = np.array(bbox, dtype=float)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

        # Voting kelas untuk menjaga konsistensi label
        self.class_votes: Dict[str, float] = {str(class_name): float(conf)}

        # Status temporal
        self.time_since_update: int = 0
        self.hits: int = 1
        self.first_frame: int = 0
        self.last_frame: int = 0

        # Kecepatan linear (vx, vy) untuk estimasi gerak
        self.vx: float = 0.0
        self.vy: float = 0.0

        # Riwayat lintasan objek
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        foot_y = self.bbox[3]
        self.history: List[Tuple[int, int]] = [(int(cx), int(foot_y))]

    def predict(self) -> np.ndarray:
        """Prediksi posisi bounding box berikutnya menggunakan model gerak linear."""
        pred = self.bbox.copy()
        if self.time_since_update > 0:
            pred[0] += self.vx
            pred[2] += self.vx
            pred[1] += self.vy
            pred[3] += self.vy
        return pred

    def update(self, new_bbox: np.ndarray, conf: float, frame_idx: int, class_name: Optional[str] = None):
        """Perbarui posisi bounding box, kecepatan, dan label kelas objek."""
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        # Estimasi kecepatan dengan exponential smoothing
        dt = max(1, self.time_since_update)
        cur_vx = (new_cx - old_cx) / dt
        cur_vy = (new_cy - old_cy) / dt
        self.vx = 0.65 * self.vx + 0.35 * cur_vx
        self.vy = 0.65 * self.vy + 0.35 * cur_vy

        had_gap = (self.time_since_update > 2)
        self.bbox = np.array(new_bbox, dtype=float)
        self.conf = float(conf)
        self.time_since_update = 0
        self.hits += 1
        self.last_frame = frame_idx

        # Konsensus voting untuk mencegah perubahan kelas tiba-tiba
        if class_name:
            if self.class_name in ["motorcycle", "bicycle"] and class_name == "person":
                self.class_votes[self.class_name] = self.class_votes.get(self.class_name, 0.0) + float(conf)
            else:
                self.class_votes[class_name] = self.class_votes.get(class_name, 0.0) + float(conf)
                self.class_name = max(self.class_votes.items(), key=lambda x: x[1])[0]

        # Simpan titik lintasan gerak
        foot_pt = (int(new_cx), int(new_bbox[3]))
        if had_gap:
            self.history = [foot_pt]
        else:
            self.history.append(foot_pt)
            if len(self.history) > 30:
                self.history.pop(0)


def are_classes_compatible(
    cls1: str,
    cls2: str,
    dist: float = 0.0,
    iou: float = 0.0,
    is_spatial_recovery: bool = False
) -> bool:
    """Memeriksa kompatibilitas kelas objek untuk asosiasi pelacakan."""
    if cls1 == cls2:
        return True
    set_classes = {cls1, cls2}
    if set_classes == {"truck", "bus"}:
        return True
    if set_classes == {"motorcycle", "bicycle"}:
        return True
    # Kendaraan roda dua dapat menyerap deteksi pengendara (person)
    if cls1 in ["motorcycle", "bicycle"] and cls2 == "person":
        if is_spatial_recovery or iou >= 0.05 or dist <= 28.0:
            return True
    return False


class InstanceTracker:
    """Pengelola pelacakan objek multi-kelas dan status objek hilang."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.model = YOLO(config.model_name)
        self.class_names = self.model.names

        self.target_ids = (
            [i for i, n in self.class_names.items() if n in config.target_classes]
            if config.target_classes else None
        )

        self.next_id: int = 1
        self.tracks: List[SingleTrack] = []
        self.all_time_unique_ids: Set[int] = set()

        self.instance_metadata: Dict[int, Dict[str, Any]] = {}
        self.instance_trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.history_records: List[Dict[str, Any]] = []
        self.prev_active_ids: Set[int] = set()

    def update(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Memproses satu frame untuk deteksi, asosiasi data, dan perhitungan status objek."""
        # 1. Jalankan deteksi objek YOLO
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            classes=self.target_ids,
            device=self.config.device,
            verbose=False
        )[0]

        raw_det_count = len(results.boxes) if results.boxes is not None else 0

        high_dets: List[Tuple[np.ndarray, float, int, str]] = []
        low_dets: List[Tuple[np.ndarray, float, int, str]] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for b, c, cid in zip(boxes, confs, clss):
                cname = self.class_names.get(cid, str(cid))
                if cname == "bus":
                    cname = "truck"
                    cid = 7

                # Pembagian threshold confidence
                high_thresh = 0.20 if cname in ["motorcycle", "person", "bicycle"] else 0.28
                if c >= high_thresh:
                    high_dets.append((b, float(c), cid, cname))
                else:
                    low_dets.append((b, float(c), cid, cname))

        # Pisahkan track aktif
        active_track_indices = [i for i, t in enumerate(self.tracks) if t.time_since_update == 0]

        matched_tracks: Set[int] = set()
        matched_high: Set[int] = set()
        reconnected_ids: List[int] = []

        # Tahap 1: Pencocokan Hungarian track aktif dengan deteksi confidence tinggi
        if len(active_track_indices) > 0 and len(high_dets) > 0:
            pred_boxes = np.array([self.tracks[i].predict() for i in active_track_indices])
            det_boxes = np.array([d[0] for d in high_dets])
            ious = compute_iou_matrix(pred_boxes, det_boxes)

            cost_matrix = np.zeros_like(ious)
            for r in range(len(active_track_indices)):
                t_idx = active_track_indices[r]
                t = self.tracks[t_idx]
                tc = ((pred_boxes[r][0] + pred_boxes[r][2]) / 2.0, (pred_boxes[r][1] + pred_boxes[r][3]) / 2.0)
                th = t.bbox[3] - t.bbox[1]

                has_nearby_bike_det = False
                if t.class_name in ["motorcycle", "bicycle"]:
                    for c_check in range(len(high_dets)):
                        if high_dets[c_check][3] in ["motorcycle", "bicycle"]:
                            dc_check = ((high_dets[c_check][0][0] + high_dets[c_check][0][2]) / 2.0, (high_dets[c_check][0][1] + high_dets[c_check][0][3]) / 2.0)
                            if np.hypot(tc[0] - dc_check[0], tc[1] - dc_check[1]) <= 28.0:
                                has_nearby_bike_det = True
                                break

                for c in range(len(high_dets)):
                    d_box, d_conf, cid, cname = high_dets[c]
                    dc = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
                    dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])

                    if not are_classes_compatible(t.class_name, cname, dist, ious[r, c]):
                        cost_matrix[r, c] = 1e5
                        continue

                    if has_nearby_bike_det and t.class_name in ["motorcycle", "bicycle"] and cname == "person":
                        cost_matrix[r, c] = 1e5
                        continue

                    dh = d_box[3] - d_box[1]
                    scale_ratio = max(dh / (th + 1e-4), th / (dh + 1e-4))
                    is_rider = (t.class_name in ["motorcycle", "bicycle"] and cname == "person")
                    scale_limit = 1.60 if is_rider else 1.25
                    scale_penalty = max(0.0, scale_ratio - scale_limit) * 3.0
                    rider_penalty = 0.50 if is_rider else 0.0

                    cost_matrix[r, c] = (1.0 - ious[r, c]) + 0.3 * min(1.0, dist / 50.0) + scale_penalty + rider_penalty

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                t_idx = active_track_indices[r]
                track = self.tracks[t_idx]
                d_box, d_conf, cid, cname = high_dets[c]

                tc = ((pred_boxes[r][0] + pred_boxes[r][2]) / 2.0, (pred_boxes[r][1] + pred_boxes[r][3]) / 2.0)
                dc = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
                dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])

                if not are_classes_compatible(track.class_name, cname, dist, ious[r, c]):
                    continue

                th = track.bbox[3] - track.bbox[1]
                dh = d_box[3] - d_box[1]
                scale_ratio = max(dh / (th + 1e-4), th / (dh + 1e-4))
                is_boundary = min(tc[0], tc[1], frame.shape[1] - tc[0], frame.shape[0] - tc[1]) < 30
                is_rider = (track.class_name in ["motorcycle", "bicycle"] and cname == "person")
                max_scale = 1.80 if is_boundary else (1.60 if is_rider else 1.35)

                if (ious[r, c] >= 0.10 or dist <= 28.0) and scale_ratio <= max_scale:
                    effective_cname = track.class_name if is_rider else cname
                    track.update(d_box, d_conf, frame_idx, effective_cname)
                    matched_tracks.add(t_idx)
                    matched_high.add(c)

        # Tahap 2: Pencocokan ByteTrack untuk track yang tersisa dengan deteksi confidence rendah
        unmatched_active = [i for i in active_track_indices if i not in matched_tracks]
        remaining_dets = (
            [(d, False) for j, d in enumerate(high_dets) if j not in matched_high]
            + [(d, True) for d in low_dets]
        )

        if len(unmatched_active) > 0 and len(remaining_dets) > 0:
            pred_boxes = np.array([self.tracks[i].predict() for i in unmatched_active])
            det_boxes = np.array([d[0][0] for d in remaining_dets])
            ious = compute_iou_matrix(pred_boxes, det_boxes)

            cost_matrix = np.zeros_like(ious)
            for r in range(len(unmatched_active)):
                t_idx = unmatched_active[r]
                t = self.tracks[t_idx]
                tc = ((pred_boxes[r][0] + pred_boxes[r][2]) / 2.0, (pred_boxes[r][1] + pred_boxes[r][3]) / 2.0)
                th = t.bbox[3] - t.bbox[1]
                for c in range(len(remaining_dets)):
                    (d_box, d_conf, cid, cname), is_low = remaining_dets[c]
                    dc = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
                    dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])
                    if not are_classes_compatible(t.class_name, cname, dist, ious[r, c]):
                        cost_matrix[r, c] = 1e5
                        continue

                    dh = d_box[3] - d_box[1]
                    scale_ratio = max(dh / (th + 1e-4), th / (dh + 1e-4))
                    is_rider = (t.class_name in ["motorcycle", "bicycle"] and cname == "person")
                    scale_limit = 1.60 if is_rider else 1.25
                    scale_penalty = max(0.0, scale_ratio - scale_limit) * 3.0
                    rider_penalty = 0.35 if is_rider else 0.0

                    cost_matrix[r, c] = (1.0 - ious[r, c]) + 0.3 * min(1.0, dist / 50.0) + scale_penalty + rider_penalty

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                t_idx = unmatched_active[r]
                track = self.tracks[t_idx]
                (d_box, d_conf, cid, cname), is_low = remaining_dets[c]

                tc = ((pred_boxes[r][0] + pred_boxes[r][2]) / 2.0, (pred_boxes[r][1] + pred_boxes[r][3]) / 2.0)
                dc = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
                dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])

                if not are_classes_compatible(track.class_name, cname, dist, ious[r, c]):
                    continue

                th = track.bbox[3] - track.bbox[1]
                dh = d_box[3] - d_box[1]
                scale_ratio = max(dh / (th + 1e-4), th / (dh + 1e-4))
                is_boundary = min(tc[0], tc[1], frame.shape[1] - tc[0], frame.shape[0] - tc[1]) < 30
                is_rider = (track.class_name in ["motorcycle", "bicycle"] and cname == "person")
                max_scale = 1.80 if is_boundary else (1.60 if is_rider else 1.35)

                if (ious[r, c] >= 0.08 or dist <= 25.0) and scale_ratio <= max_scale:
                    effective_cname = track.class_name if is_rider else cname
                    track.update(d_box, d_conf, frame_idx, effective_cname)
                    matched_tracks.add(t_idx)

        # Update umur track yang tidak mendapat pasangan
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.time_since_update += 1

        # Tahap 3: Pemulihan spasial untuk track yang baru hilang
        claimed_lost_track_ids: Set[int] = set()

        for j, (d_box, d_conf, cid, cname) in enumerate(high_dets):
            if j in matched_high:
                continue

            dc = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
            dh = d_box[3] - d_box[1]
            best_track = None
            best_dist = 999.0

            for i, track in enumerate(self.tracks):
                if i in matched_tracks or track.id in claimed_lost_track_ids:
                    continue

                if track.time_since_update <= self.config.max_lost_buffer:
                    dt = float(track.time_since_update)
                    tc_pred = (
                        (track.bbox[0] + track.bbox[2]) / 2.0 + track.vx * dt,
                        (track.bbox[1] + track.bbox[3]) / 2.0 + track.vy * dt,
                    )
                    dist = np.hypot(tc_pred[0] - dc[0], tc_pred[1] - dc[1])

                    if not are_classes_compatible(track.class_name, cname, dist, 0.0, is_spatial_recovery=True):
                        continue

                    th = track.bbox[3] - track.bbox[1]
                    scale_ratio = max(dh / (th + 1e-4), th / (dh + 1e-4))
                    speed = np.hypot(track.vx, track.vy)
                    is_vehicle = track.class_name in ["car", "truck", "motorcycle", "bus", "bicycle"]
                    max_allowed_dist = 16.0 if (speed < 0.6 and not is_vehicle) else (45.0 + 3.0 * dt)

                    if dist <= max_allowed_dist and scale_ratio <= 1.60 and dist < best_dist:
                        best_dist = dist
                        best_track = track

            if best_track is not None:
                is_rider = (best_track.class_name in ["motorcycle", "bicycle"] and cname == "person")
                effective_cname = best_track.class_name if is_rider else cname
                best_track.update(d_box, d_conf, frame_idx, effective_cname)
                reconnected_ids.append(best_track.id)
                claimed_lost_track_ids.add(best_track.id)
            else:
                # Tahap 4: Registrasi objek baru
                # Supresi pengendara di atas sepeda motor
                if cname == "person":
                    is_rider_of_bike = False
                    for track in self.tracks:
                        if track.class_name in ["motorcycle", "bicycle"] and track.time_since_update <= 8:
                            dt = float(track.time_since_update)
                            tc_pred = (
                                (track.bbox[0] + track.bbox[2]) / 2.0 + track.vx * dt,
                                (track.bbox[1] + track.bbox[3]) / 2.0 + track.vy * dt,
                            )
                            dist = np.hypot(tc_pred[0] - dc[0], tc_pred[1] - dc[1])
                            if dist <= 28.0:
                                is_rider_of_bike = True
                                break
                    if is_rider_of_bike:
                        continue

                # Supresi deteksi duplikat pada kendaraan roda dua
                if cname in ["motorcycle", "bicycle"]:
                    is_dup_bike = False
                    for track in self.tracks:
                        if track.class_name in ["motorcycle", "bicycle"] and track.time_since_update <= 2:
                            tc = ((track.bbox[0] + track.bbox[2]) / 2.0, (track.bbox[1] + track.bbox[3]) / 2.0)
                            dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])
                            if dist <= 22.0:
                                is_dup_bike = True
                                break
                    if is_dup_bike:
                        continue

                reg_thresh = 0.18 if cname in ["motorcycle", "bicycle"] else (0.24 if cname == "person" else 0.28)
                if d_conf >= reg_thresh:
                    new_track = SingleTrack(self.next_id, d_box, d_conf, cid, cname)
                    new_track.first_frame = frame_idx
                    new_track.last_frame = frame_idx
                    self.tracks.append(new_track)
                    self.next_id += 1

        # Hapus track yang melewati batas toleransi memori
        self.tracks = [t for t in self.tracks if t.time_since_update <= 90]

        # Ambil seluruh track yang memenuhi syarat aktif pada frame saat ini
        current_active_instances: Dict[int, Dict[str, Any]] = {}
        for track in self.tracks:
            is_active = (track.time_since_update == 0) or (track.time_since_update <= 2 and track.hits >= 2)
            if is_active:
                self.all_time_unique_ids.add(track.id)
                self.instance_trajectories[track.id] = list(track.history)

                if track.id not in self.instance_metadata:
                    self.instance_metadata[track.id] = {
                        "class_name": track.class_name,
                        "first_seen_frame": track.first_frame
                    }
                else:
                    self.instance_metadata[track.id]["class_name"] = track.class_name

                current_active_instances[track.id] = {
                    "id": track.id,
                    "bbox": [float(c) for c in track.bbox],
                    "center": ((track.bbox[0] + track.bbox[2]) / 2.0, (track.bbox[1] + track.bbox[3]) / 2.0),
                    "confidence": track.conf,
                    "class_name": track.class_name,
                    "class_id": track.class_id,
                    "last_seen_frame": frame_idx,
                }

        current_active_ids = set(current_active_instances.keys())

        # Hitung ID baru dan ID yang hilang (selisih himpunan)
        if len(self.history_records) == 0:
            new_ids = current_active_ids
            lost_ids = set()
        else:
            new_ids = current_active_ids - self.prev_active_ids
            lost_ids = self.prev_active_ids - current_active_ids

        self.prev_active_ids = current_active_ids

        # Ringkasan data frame
        frame_summary = {
            "frame_idx": frame_idx,
            "raw_det_count": raw_det_count,
            "active_count": len(current_active_ids),
            "active_ids": sorted(list(current_active_ids)),
            "new_count": len(new_ids),
            "new_ids": sorted(list(new_ids)),
            "lost_count": len(lost_ids),
            "lost_ids": sorted(list(lost_ids)),
            "reconnected_ids": reconnected_ids,
            "cumulative_unique_count": len(self.all_time_unique_ids),
            "active_instances": current_active_instances,
        }

        # Catat log baris untuk file CSV
        self.history_records.append({
            "frame_idx": frame_idx,
            "raw_det_count": raw_det_count,
            "active_count": len(current_active_ids),
            "new_count": len(new_ids),
            "lost_count": len(lost_ids),
            "lost_ids_str": ",".join(map(str, sorted(list(lost_ids)))),
            "cumulative_unique": len(self.all_time_unique_ids),
        })

        return frame_summary

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Menghitung ringkasan statistik agregat pelacakan video."""
        if not self.history_records:
            return {}

        counts = [r["active_count"] for r in self.history_records]
        total_lost_events = sum(r["lost_count"] for r in self.history_records)

        return {
            "total_frames_processed": len(self.history_records),
            "cumulative_unique_instances": len(self.all_time_unique_ids),
            "min_active_per_frame": int(np.min(counts)),
            "max_active_per_frame": int(np.max(counts)),
            "avg_active_per_frame": float(np.mean(counts)),
            "total_lost_events": int(total_lost_events),
            "active_ids_list": sorted(list(self.all_time_unique_ids)),
            "instances_summary": self.instance_metadata,
        }
