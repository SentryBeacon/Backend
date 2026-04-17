"""
=============================================================
    python wrong_lane_detector.py --draw
    python wrong_lane_detector.py --run
    python wrong_lane_detector.py --run --show
=============================================================
"""

import cv2
import json
import os
import argparse
import numpy as np
import threading
import queue
import time
from datetime import datetime
from collections import defaultdict
import re
import csv


#  CONFIG
ZONE_POLYGON = [
    (80, 1060), (400, 1060),
    (370,  380), (60,   380),
]
VLINE_A = (60,  370)
VLINE_B = (420, 370)

ENTER_CONFIRM_FRAMES     = 1

VIOLATION_CONFIRM_FRAMES = 1
COOLDOWN_FRAMES          = 60

GHOST_FRAMES             = 8
LOST_TTL                 = 30

VIDEO_PATH  = "video.mp4"
OUTPUT_DIR  = "../evidence/evidence_lane"
CONFIG_FILE = "config.json"

YOLO_MODEL  = "yolov8m.pt"

VEHICLE_CLASSES = {2, 3, 5}
VEHICLE_NAMES   = {2: "Car", 3: "Moto", 5: "Bus"}
CONF_THRESH     = 0.25
IOU_THRESH      = 0.45

INFER_W = 960

DRAG_THRESH = 16

SORT_MAX_AGE   = LOST_TTL
SORT_MIN_HITS  = 1
SORT_IOU_THRESH = 0.25

def _iou_batch(bb_test, bb_gt):
    """Tinh IoU giua mang det va mang track. Shape: (N,4) vs (M,4)."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    a1 = (bb_test[..., 2]-bb_test[..., 0])*(bb_test[..., 3]-bb_test[..., 1])
    a2 = (bb_gt[..., 2]-bb_gt[..., 0])*(bb_gt[..., 3]-bb_gt[..., 1])
    return inter / (a1 + a2 - inter + 1e-9)


def _linear_assignment(cost_matrix):
    """Hungarian algorithm don gian (scipy-free)."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost_matrix)
        return np.stack([r, c], axis=1)
    except ImportError:
        # Fallback greedy
        n, m = cost_matrix.shape
        assigned = []
        used_r, used_c = set(), set()
        idx = np.argsort(cost_matrix.ravel())
        for flat in idx:
            r, c = divmod(int(flat), m)
            if r not in used_r and c not in used_c:
                assigned.append([r, c])
                used_r.add(r); used_c.add(c)
        return np.array(assigned, dtype=int) if assigned else np.empty((0, 2), int)


class _KalmanBoxTracker:
    """Kalman filter theo doi 1 bbox [x1,y1,x2,y2]."""
    count = 0

    def __init__(self, bbox):
        # State: [cx, cy, s, r, dcx, dcy, ds]  (s=area, r=aspect)
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.
        meas = self._bbox_to_z(bbox)
        self.kf.statePost = np.array(
            [meas[0], meas[1], meas[2], meas[3], 0, 0, 0], np.float32
        ).reshape(7, 1)
        _KalmanBoxTracker.count += 1
        self.id          = _KalmanBoxTracker.count
        self.time_since_update = 0
        self.hit_streak  = 0
        self.hits        = 0
        self.age         = 0

    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        s  = (x2 - x1) * (y2 - y1)
        r  = (x2 - x1) / float(y2 - y1 + 1e-6)
        return np.array([cx, cy, s, r], np.float32)

    @staticmethod
    def _z_to_bbox(z):
        z = z.flatten()
        s, r = float(z[2]), float(z[3])
        s = max(s, 1.)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        cx, cy = float(z[0]), float(z[1])
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    def predict(self):
        if self.kf.statePost[6] + self.kf.statePost[2] <= 0:
            self.kf.statePost[6] = 0.
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.statePost)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.correct(self._bbox_to_z(bbox).reshape(4, 1))

    def get_state(self):
        return self._z_to_bbox(self.kf.statePost)


class SORTTracker:
    """SORT: Simple Online and Realtime Tracking."""

    def __init__(self, max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                 iou_threshold=SORT_IOU_THRESH):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def update(self, dets):
        """
        dets: np.array shape (N,5) [x1,y1,x2,y2,conf] hoac (0,5) neu khong co det.
        Tra ve: np.array shape (M,5) [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # Predict vi tri moi cho tat ca tracker hien co
        predicted = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                predicted.append(pos)
        for t in reversed(to_del):
            self.trackers.pop(t)
        predicted = np.array(predicted) if predicted else np.empty((0, 4))

        matched_t = set()
        matched_d = set()
        if len(predicted) > 0 and len(dets) > 0:
            iou_mat = _iou_batch(dets[:, :4], predicted)
            cost    = 1. - iou_mat
            pairs   = _linear_assignment(cost)
            for d_idx, t_idx in pairs:
                if iou_mat[d_idx, t_idx] >= self.iou_threshold:
                    self.trackers[t_idx].update(dets[d_idx, :4])
                    matched_t.add(t_idx)
                    matched_d.add(d_idx)

        # Tao tracker moi cho det chua duoc match
        for d_idx, det in enumerate(dets):
            if d_idx not in matched_d:
                self.trackers.append(_KalmanBoxTracker(det[:4]))

        # Thu thap ket qua, xoa tracker qua han
        ret = []
        keep = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = trk.get_state()
                    ret.append(np.append(bbox, trk.id))
                keep.append(trk)
        self.trackers = keep
        return np.array(ret) if ret else np.empty((0, 5))


#  ANALYZER
class Analyzer:
    S_UNSEEN   = 0
    S_ENTERING = 1
    S_TRACKING = 2
    S_VIOLATED = 3
    S_GHOST    = 4

    def __init__(self, polygon_pts, pa, pb,
                 enter_confirm=ENTER_CONFIRM_FRAMES,
                 vio_confirm=VIOLATION_CONFIRM_FRAMES,
                 cooldown=COOLDOWN_FRAMES):

        self.poly_np = np.array(
            [(int(p[0]), int(p[1])) for p in polygon_pts], dtype=np.int32
        )
        self.va = np.array([float(pa[0]), float(pa[1])])
        self.vb = np.array([float(pb[0]), float(pb[1])])

        self.enter_confirm = enter_confirm
        self.vio_confirm   = vio_confirm
        self.cooldown_max  = cooldown

        self.state      = defaultdict(lambda: self.S_UNSEEN)
        self.enter_cnt  = defaultdict(int)
        self.vio_cnt    = defaultdict(int)
        self.cd_cnt     = defaultdict(int)
        self.ghost_cnt  = defaultdict(int)
        self.prev_pt    = {}
        self.last_bbox  = {}

    def _in_zone(self, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx   = (x1 + x2) // 2

        bot  = y2
        mid  = (y1 + y2) // 2
        qbot = (mid + y2) // 2

        pts1 = [
            (cx,  bot),
            (x1,  bot),
            (x2,  bot),
            ((x1+cx)//2, bot),
            ((cx+x2)//2, bot),
            (cx,  qbot),
            (x1,  qbot),
            (x2,  qbot),
            (cx,  mid),
        ]
        for px, py in pts1:
            if cv2.pointPolygonTest(self.poly_np, (float(px), float(py)), False) >= 0:
                return True

        # Tang 2: dinh polygon nam trong bbox?
        for vx, vy in self.poly_np:
            if x1 <= vx <= x2 and y1 <= vy <= y2:
                return True

        # Tang 3: grid 3x3 nua duoi bbox
        step_x = max((x2 - x1) // 2, 1)
        step_y = max((y2 - mid) // 2, 1)
        for gx in range(x1, x2 + 1, step_x):
            for gy in range(mid, y2 + 1, step_y):
                if cv2.pointPolygonTest(self.poly_np, (float(gx), float(gy)), False) >= 0:
                    return True
        return False

    @staticmethod
    def _cross2d(ox, oy, ux, uy, vx, vy):
        return (ux - ox) * (vy - oy) - (uy - oy) * (vx - ox)

    def _seg_intersects(self, p1x, p1y, p2x, p2y):
        """Kiem tra doan p1Ã¢â€ â€™p2 co cat vach vaÃ¢â€ â€™vb khong (Shamos-Hoey)."""
        ax, ay = self.va[0], self.va[1]
        bx, by = self.vb[0], self.vb[1]
        EPS = 1e-9

        d1 = self._cross2d(ax, ay, bx, by, p1x, p1y)
        d2 = self._cross2d(ax, ay, bx, by, p2x, p2y)
        d3 = self._cross2d(p1x, p1y, p2x, p2y, ax, ay)
        d4 = self._cross2d(p1x, p1y, p2x, p2y, bx, by)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        def on_seg(ox, oy, ux, uy, vx, vy):
            return (min(ox,ux)-EPS <= vx <= max(ox,ux)+EPS and
                    min(oy,uy)-EPS <= vy <= max(oy,uy)+EPS)

        if abs(d1) < EPS and on_seg(ax, ay, bx, by, p1x, p1y): return True
        if abs(d2) < EPS and on_seg(ax, ay, bx, by, p2x, p2y): return True
        if abs(d3) < EPS and on_seg(p1x, p1y, p2x, p2y, ax, ay): return True
        if abs(d4) < EPS and on_seg(p1x, p1y, p2x, p2y, bx, by): return True
        return False

    def _crosses_vline(self, bbox, prev_bbox, cx, cy, prev):
        if prev is None or prev_bbox is None:
            return False

        # Tranh false positive khi xe dung yen
        dist = abs(cy - prev[1]) + abs(cx - prev[0])
        if dist < 1.0:
            return False

        if self._seg_intersects(prev[0], prev[1], cx, cy):
            return True

        x1, y1, x2, y2 = [int(v) for v in bbox]
        px1, py1, px2, py2 = [int(v) for v in prev_bbox]
        mx_cur  = (x1+x2)/2.;  my_cur  = (y1+y2)/2.
        mx_prev = (px1+px2)/2.; my_prev = (py1+py2)/2.
        if self._seg_intersects(mx_prev, my_prev, mx_cur, my_cur):
            return True

        for (cpx, cpy), (ccx, ccy) in [
            ((px1, py1), (x1, y1)),
            ((px2, py1), (x2, y1)),
            ((px1, py2), (x1, y2)),
            ((px2, py2), (x2, y2)),
        ]:
            if self._seg_intersects(float(cpx), float(cpy),
                                    float(ccx), float(ccy)):
                return True

        return False

    def update(self, tid, bbox):
        """
        Tra ve ('tracking'|'violation'|'ghost'|None, bbox_to_draw).
        bbox_to_draw co the la bbox hien tai hoac bbox cu (khi ghost).
        """
        tid = int(tid)
        self.last_bbox[tid]  = bbox
        self.ghost_cnt[tid]  = 0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx_cur = (x1 + x2) / 2.
        cy_cur = float(y2)
        prev      = self.prev_pt.get(tid)
        prev_bbox = self.last_bbox.get(tid)
        self.prev_pt[tid] = (cx_cur, cy_cur)

        if self.cd_cnt[tid] > 0:
            self.cd_cnt[tid] -= 1
            if self.cd_cnt[tid] == 0:
                self.state[tid]    = self.S_UNSEEN
                self.enter_cnt[tid] = 0
                self.vio_cnt[tid]   = 0
            return 'tracking', bbox

        inside = self._in_zone(bbox)

        # --- UNSEEN ---
        if self.state[tid] == self.S_UNSEEN:
            if inside:
                self.state[tid]    = self.S_ENTERING
                self.enter_cnt[tid] = 1
                if self.enter_confirm <= 1:
                    self.state[tid] = self.S_TRACKING
                return 'tracking', bbox
            return None, None

        # --- ENTERING ---
        if self.state[tid] == self.S_ENTERING:
            if inside:
                self.enter_cnt[tid] += 1
                if self.enter_cnt[tid] >= self.enter_confirm:
                    self.state[tid] = self.S_TRACKING
            else:
                self.enter_cnt[tid] = max(0, self.enter_cnt[tid] - 1)
                if self.enter_cnt[tid] == 0:
                    self.state[tid] = self.S_UNSEEN
                    return None, None
            return 'tracking', bbox

        # --- TRACKING ---
        if self.state[tid] == self.S_TRACKING:
            if self._crosses_vline(bbox, prev_bbox, cx_cur, cy_cur, prev):
                self.vio_cnt[tid] += 1
                if self.vio_cnt[tid] >= self.vio_confirm:
                    self.state[tid]   = self.S_VIOLATED
                    self.cd_cnt[tid]  = self.cooldown_max
                    self.vio_cnt[tid] = 0
                    return 'violation', bbox
            return 'tracking', bbox

        # --- VIOLATED (trong cooldown, xu ly o tren) ---
        if self.state[tid] == self.S_VIOLATED:
            self.state[tid]     = self.S_UNSEEN
            self.enter_cnt[tid] = 0
            self.vio_cnt[tid]   = 0
            return None, None

        return None, None

    def ghost_update(self, tid):

        tid = int(tid)
        if self.state[tid] not in (self.S_TRACKING, self.S_ENTERING,
                                   self.S_VIOLATED, self.S_GHOST):
            return None, None
        self.ghost_cnt[tid] += 1
        if self.ghost_cnt[tid] > GHOST_FRAMES:
            self._erase(tid)
            return None, None
        self.state[tid] = self.S_GHOST
        return 'ghost', self.last_bbox.get(tid)

    def cleanup(self, active_ids):
        """Goi sau moi frame voi tap ID thuc su con song."""
        active = {int(i) for i in active_ids}
        for tid in list(self.state.keys()):
            if tid not in active and self.state[tid] != self.S_GHOST:
                # Chuyen sang ghost thay vi xoa ngay
                if self.state[tid] in (self.S_TRACKING, self.S_ENTERING):
                    self.state[tid] = self.S_GHOST
                    self.ghost_cnt[tid] = 0
                elif self.state[tid] == self.S_UNSEEN:
                    self._erase(tid)

        # Gia tang ghost_cnt cho cac ID dang ghost nhung khong con active
        for tid in list(self.state.keys()):
            if tid not in active and self.state[tid] == self.S_GHOST:
                self.ghost_cnt[tid] += 1
                if self.ghost_cnt[tid] > GHOST_FRAMES:
                    self._erase(tid)

    def _erase(self, tid):
        for d in (self.state, self.enter_cnt, self.vio_cnt,
                  self.cd_cnt, self.ghost_cnt, self.prev_pt, self.last_bbox):
            d.pop(tid, None)

    def get_state(self, tid):
        return self.state.get(int(tid), self.S_UNSEEN)


# ── VIOLATION SAVER ────────────────────────────────────────────────────────────
class ViolationSaver:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.out    = output_dir
        self.count  = 0
        os.makedirs(output_dir, exist_ok=True)
        # Tao file log CSV
        self._log_path = os.path.join(output_dir, "violations.csv")
        with open(self._log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                ['#', 'track_id', 'frame', 'time_s', 'base'])

    def save(self, frame, track_id, bbox, frame_idx, fps, best_info=None):
        self.count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        pw = max(int((x2-x1)*0.3), 30)
        ph = max(int((y2-y1)*0.3), 30)
        crop  = frame[max(0,y1-ph):min(fh,y2+ph),
                      max(0,x1-pw):min(fw,x2+pw)]
        t_s = frame_idx / max(fps, 1)
        base = f"vp_{self.count:04d}_id{track_id}_{ts}"
        
        best_crop_path = None
        if best_info and len(best_info) > 0:
            # Lay anh lon nhat (dau danh sach) de lam anh 'best' dai dien
            bf, bb, bf_idx, _ = best_info[0]
            bx1, by1, bx2, by2 = [int(v) for v in bb]
            bh, bw = bf.shape[:2]
            bpw = max(int((bx2-bx1)*0.3), 30)
            bph = max(int((by2-by1)*0.3), 30)
            best_crop = bf[max(0,by1-bph):min(bh,by2+bph),
                           max(0,bx1-bpw):min(bw,bx2+bpw)]
            best_crop_path = os.path.join(self.out, base+"_best.jpg")
            cv2.imwrite(best_crop_path, best_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Luu crop tai frame vi pham
        cv2.imwrite(os.path.join(self.out, base+"_crop.jpg"),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Luu scene 
        scene = frame.copy()
        cv2.rectangle(scene, (x1,y1), (x2,y2), (0,0,220), 3)
        label = f"VI PHAM ID:{track_id}"
        cv2.rectangle(scene, (x1, max(y1-35, 0)), (x1+300, y1), (0,0,0), -1) 
        cv2.putText(scene, label, (x1, max(y1-12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,220), 2)
        cv2.putText(scene, f"t={t_s:.1f}s  frame={frame_idx}",
                    (x1, y2+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,220), 2)
        scene_path = os.path.join(self.out, base+"_scene.jpg")
        cv2.imwrite(scene_path, scene, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Ghi vao CSV
        with open(self._log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.count, track_id, frame_idx, f"{t_s:.2f}", base])
            
        print(f"  [VI PHAM #{self.count}] ID={track_id} frame={frame_idx} (Da ghi nhan)")



#  VIDEO READER (threaded)
class VideoReader(threading.Thread):
    def __init__(self, path, buf=16):
        super().__init__(daemon=True)
        # [OPTIMIZED] Use FFMPEG backend directly for better decoding parallelization
        self.cap   = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        self.q     = queue.Queue(maxsize=buf)
        self._stop = threading.Event()

    @property
    def fps(self):    return self.cap.get(cv2.CAP_PROP_FPS) or 25.0
    @property
    def total(self):  return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    @property
    def width(self):  return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                try: self.q.put(None, timeout=1)
                except queue.Full: pass
                break
            try:
                self.q.put(frame, timeout=0.05)
            except queue.Full:
                # [OPTIMIZED] Prevent thread from burning 100% CPU when queue is full
                time.sleep(0.01)
        self.cap.release()

    def read(self, timeout=5):
        return self.q.get(timeout=timeout)

    def stop(self):
        self._stop.set()
        while not self.q.empty():
            try: self.q.get_nowait()
            except queue.Empty: break


#  GUI DRAWER
class Drawer:
    _PC = (220, 80, 255)
    _VC = (0, 60, 230)
    _HR = 10

    def __init__(self, frame):
        self.frame = frame.copy()
        self.H, self.W = frame.shape[:2]
        self._reset()

    def _reset(self):
        self.poly   = [list(p) for p in ZONE_POLYGON]
        self.pa     = list(VLINE_A)
        self.pb     = list(VLINE_B)
        self.closed = True
        self._drag  = None

    @staticmethod
    def _dist(ax, ay, bx, by):
        return ((ax-bx)**2 + (ay-by)**2) ** 0.5

    def _hit(self, x, y):
        th = DRAG_THRESH
        best_d, best_h = float('inf'), None
        for key, pt in (('a', self.pa), ('b', self.pb)):
            d = self._dist(x, y, pt[0], pt[1])
            if d < th and d < best_d:
                best_d, best_h = d, ('vline', key)
        for i, p in enumerate(self.poly):
            d = self._dist(x, y, p[0], p[1])
            if d < th and d < best_d:
                best_d, best_h = d, ('poly', i)
        return best_h

    def _render(self):
        vis = self.frame.copy()
        if len(self.poly) >= 3:
            pts = np.array(self.poly, np.int32)
            ov  = vis.copy()
            cv2.fillPoly(ov, [pts], self._PC)
            cv2.addWeighted(ov, 0.18, vis, 0.82, 0, vis)
            cv2.polylines(vis, [pts], self.closed, self._PC, 2)
        elif len(self.poly) == 2:
            cv2.line(vis, tuple(self.poly[0]), tuple(self.poly[1]), self._PC, 2)
        for i, p in enumerate(self.poly):
            drag  = (self._drag == ('poly', i))
            r     = self._HR + 3 if drag else self._HR
            color = (0, 255, 120) if drag else self._PC
            cv2.circle(vis, tuple(p), r, color, -1)
            cv2.circle(vis, tuple(p), r, (255,255,255), 1)
            cv2.putText(vis, f"P{i+1}", (p[0]+r+3, p[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        if self.poly:
            cx = sum(p[0] for p in self.poly) // len(self.poly)
            cy = sum(p[1] for p in self.poly) // len(self.poly)
            cv2.putText(vis, "LAN CAM", (cx-36, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        ax, ay = self.pa; bx, by = self.pb
        cv2.line(vis, (ax,ay), (bx,by), self._VC, 3)
        for key, pt, lbl in (('a',self.pa,'A'), ('b',self.pb,'B')):
            drag  = (self._drag == ('vline', key))
            r     = self._HR + 3 if drag else self._HR
            color = (0, 200, 255) if drag else self._VC
            cv2.circle(vis, tuple(pt), r, color, -1)
            cv2.circle(vis, tuple(pt), r, (255,255,255), 2)
            cv2.putText(vis, lbl, (pt[0]+r+3, pt[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        mx = (ax+bx)//2 - 70; my = (ay+by)//2 - 14
        cv2.putText(vis, "VIOLATION LINE", (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,160,255), 2)
        guide = [
            "Trai-click them diem  |  Trai-drag di chuyen  |  Phai-click xoa diem",
            "C: dong/mo poly   Z: undo   R: reset   S: luu   Q: thoat",
            f"Polygon: {len(self.poly)} diem  |  {'DONG' if self.closed else 'MO'}",
        ]
        for i, g in enumerate(guide):
            cv2.putText(vis, g, (10, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220,220,50), 1)
        return vis

    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit(x, y)
            if hit: self._drag = hit
            else:
                self.poly.append([x, y])
                self._drag = ('poly', len(self.poly)-1)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag is None: return
            kind, ref = self._drag
            if kind == 'poly': self.poly[ref] = [x, y]
            elif ref == 'a':   self.pa = [x, y]
            else:              self.pb = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = self._hit(x, y)
            if hit and hit[0] == 'poly' and len(self.poly) > 1:
                self.poly.pop(hit[1])

    def run(self):
        cv2.namedWindow("Drawer - Wrong Lane Config", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Drawer - Wrong Lane Config", self._on_mouse)
        while True:
            cv2.imshow("Drawer - Wrong Lane Config", self._render())
            key = cv2.waitKey(20) & 0xFF
            if key == ord('c'):   self.closed = not self.closed
            elif key == ord('z') and self.poly: self.poly.pop()
            elif key == ord('r'): self._reset()
            elif key == ord('s'):
                if len(self.poly) < 3:
                    print("[WARN] Can it nhat 3 diem polygon!"); continue
                self._save()
                print(f"[SAVED] {CONFIG_FILE}")
                break
            elif key == ord('q'): break
        cv2.destroyAllWindows()
        return self.poly, self.pa, self.pb

    def _save(self):
        cfg = {
            "zone_polygon": self.poly,
            "vline_a": self.pa, "vline_b": self.pb,
            "enter_confirm_frames":     ENTER_CONFIRM_FRAMES,
            "violation_confirm_frames": VIOLATION_CONFIRM_FRAMES,
            "cooldown_frames":          COOLDOWN_FRAMES,
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)


#  HELPERS
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f: cfg = json.load(f)
        poly = cfg["zone_polygon"]
        pa   = cfg["vline_a"]; pb = cfg["vline_b"]
        ec   = cfg.get("enter_confirm_frames",     ENTER_CONFIRM_FRAMES)
        vc   = cfg.get("violation_confirm_frames", VIOLATION_CONFIRM_FRAMES)
        cd   = cfg.get("cooldown_frames",          COOLDOWN_FRAMES)
        print(f"[CONFIG] Loaded tu {CONFIG_FILE}")
    else:
        poly = list(ZONE_POLYGON)
        pa, pb = list(VLINE_A), list(VLINE_B)
        ec, vc, cd = ENTER_CONFIRM_FRAMES, VIOLATION_CONFIRM_FRAMES, COOLDOWN_FRAMES
        print("[CONFIG] Dung hardcode mac dinh Ã¢â‚¬â€ hay chay --draw de ve dung vi tri!")
    print(f"  Polygon : {len(poly)} diem | Vline: {pa} -> {pb}")
    return poly, pa, pb, ec, vc, cd


def nms(dets, iou_thr=0.50):
    """NMS don gian (x1y1x2y2 + conf)."""
    if len(dets) == 0:
        return dets
    x1, y1, x2, y2 = dets[:,0], dets[:,1], dets[:,2], dets[:,3]
    scores = dets[:,4]
    areas  = (x2-x1) * (y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0., xx2-xx1) * np.maximum(0., yy2-yy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return dets[keep]


def build_zone_overlay(shape, poly, pa, pb):
    ov  = np.zeros(shape, dtype=np.uint8)
    pts = np.array([(int(p[0]),int(p[1])) for p in poly], np.int32)
    cv2.fillPoly(ov, [pts], (180, 60, 220))
    cv2.polylines(ov, [pts], True, (220, 80, 255), 2)
    cv2.line(ov, tuple(pa), tuple(pb), (0, 50, 220), 3)
    cx = int(np.mean([p[0] for p in poly]))
    cy = int(np.mean([p[1] for p in poly]))
    cv2.putText(ov, "LAN CAM DI THANG", (cx-80, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    mx = (pa[0]+pb[0])//2 - 70; my = (pa[1]+pb[1])//2 - 14
    cv2.putText(ov, "VIOLATION LINE", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,160,255), 2)
    return ov


#  MAIN PIPELINE
def run_detection(show=False):
    from ultralytics import YOLO

    poly, pa, pb, ec, vc, cd = load_config()
    pa_t = tuple(int(v) for v in pa)
    pb_t = tuple(int(v) for v in pb)

    reader = VideoReader(VIDEO_PATH)
    if not reader.cap.isOpened():
        print(f"[ERROR] Khong mo duoc: {VIDEO_PATH}"); return

    fps   = reader.fps
    total = reader.total
    W, H  = reader.width, reader.height
    print(f"[VIDEO] {VIDEO_PATH}  {W}x{H}  {fps:.1f}fps  {total} frames")
    reader.start()

    # YOLO
    model = YOLO(YOLO_MODEL)

    infer_h = int(INFER_W * H / W)
    scale_x = W / INFER_W
    scale_y = H / infer_h

    import torch
    _dummy = np.zeros((infer_h, INFER_W, 3), dtype=np.uint8)
    model(_dummy, verbose=False)
    use_half = False
    if torch.cuda.is_available():
        try:
            model.model.half()
            use_half = True
            print("[MODEL] Half-precision (FP16) enabled (GPU)")
        except Exception:
            pass
    if not use_half:
        print("[MODEL] FP32 (CPU hoac GPU khong ho tro FP16)")

    # SORT tracker (tu tich hop, khong phu thuoc supervision)
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join(OUTPUT_DIR, f"run_{run_ts}")
    print(f"[OUTPUT] Luu ket qua vao: {run_dir}")

    tracker  = SORTTracker()
    analyzer = Analyzer(poly, pa_t, pb_t, ec, vc, cd)
    saver    = ViolationSaver(run_dir)

    S = Analyzer
    COLOR = {
        S.S_ENTERING: (0,   200, 200),
        S.S_TRACKING: (0,   200,  50),
        S.S_VIOLATED: (0,    40, 220),
        S.S_GHOST:    (150, 150,   0),
    }
    LABEL = {
        S.S_ENTERING: "ENTER",
        S.S_TRACKING: "TRACK",
        S.S_VIOLATED: "VI PHAM!",
        S.S_GHOST:    "GHOST",
    }

    if show:
        win = "Wrong Lane Detector"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, min(W, 1280), min(H, 720))
            zone_ov = build_zone_overlay((H, W, 3), poly, pa_t, pb_t)
        except cv2.error:
            print("\n[WARNING] OpenCV GUI khong duoc ho tro (Headless mode).")
            print("Vui long chay: pip install opencv-python --force-reinstall")
            print("System se tiep tuc chay o che do CLI (khong hien thi cua so).\n")
            show = False

    frame_idx  = 0
    spf        = 1.0 / max(fps, 1)
    track_cls    = {}
    best_frames  = {}   # {tid: (frame_copy, bbox, frame_idx, area)}
    pending_vio  = set() # IDs that have violated but not yet saved
    print(f"\n[RUN] Infer moi frame | nhan Q de dung\n")

    try:
        while True:
            t0 = time.perf_counter()

            try:
                frame = reader.read(timeout=5)
            except queue.Empty:
                break
            if frame is None:
                break

            small   = cv2.resize(frame, (INFER_W, infer_h), interpolation=cv2.INTER_LINEAR)
            
            # [OPTIMIZED] Push NMS & Filtering down to YOLO's native PyTorch/C++ backend (x3 Speedup)
            results = model(small, conf=CONF_THRESH, iou=IOU_THRESH, classes=list(VEHICLE_CLASSES), verbose=False)[0]

            raw_dets = []
            det_cls  = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clses = boxes.cls.cpu().numpy().astype(int)
                for j in range(len(xyxy)):
                    x1o = xyxy[j][0] * scale_x; y1o = xyxy[j][1] * scale_y
                    x2o = xyxy[j][2] * scale_x; y2o = xyxy[j][3] * scale_y
                    raw_dets.append([x1o, y1o, x2o, y2o, confs[j]])
                    det_cls.append(clses[j])

            dets_np = np.array(raw_dets, dtype=np.float32) if raw_dets else np.empty((0, 5), dtype=np.float32)
            
            # (Loại bỏ mảng python NMS thủ công vì YOLO đã tự lo liệu cực kì nhanh và chuẩn xác)


            tracks = tracker.update(dets_np)

            # Map track_id -> vehicle class by IoU matching
            if len(tracks) > 0 and len(dets_np) > 0:
                iou_mat = _iou_batch(tracks[:, :4], dets_np[:, :4])
                for i in range(len(tracks)):
                    best = np.argmax(iou_mat[i])
                    if iou_mat[i, best] > 0.3:
                        track_cls[int(tracks[i, 4])] = det_cls[best]

            active_ids = []
            draw_ids   = {}

            for row in tracks:
                x1, y1, x2, y2, tid_f = row
                tid  = int(tid_f)
                bbox = np.array([x1, y1, x2, y2])
                active_ids.append(tid)

                result, draw_bbox = analyzer.update(tid, bbox)

                # Cap nhat best_frame: Luu TOP 5 frame co area lon nhat
                if result in ('tracking', 'violation'):
                    area = (x2 - x1) * (y2 - y1)
                    if tid not in best_frames:
                        best_frames[tid] = []
                    
                    # Update list
                    best_frames[tid].append((frame.copy(), bbox.copy(), frame_idx, area))
                    best_frames[tid].sort(key=lambda x: x[3], reverse=True)
                    best_frames[tid] = best_frames[tid][:5] # Keep TOP 5

                if result == 'violation':
                    pending_vio.add(tid)
                    draw_ids[tid] = (draw_bbox, S.S_VIOLATED)
                elif result == 'tracking':
                    draw_ids[tid] = (draw_bbox, analyzer.get_state(tid))

            # [ULTRA-PRECISE] Delayed save: save when ID is lost
            for tid in list(pending_vio):
                if tid not in active_ids:
                    # Final save before discarding
                    b_info = best_frames.pop(tid, None)
                    # Get last known bbox for labels
                    l_bbox = analyzer.last_bbox.get(tid)
                    if l_bbox is not None:
                        saver.save(frame, tid, l_bbox, frame_idx, fps, b_info)
                    pending_vio.remove(tid)

            # Cleanup best_frames cho xe khong con tracking va khong pending
            for tid in list(best_frames.keys()):
                if tid not in active_ids and tid not in pending_vio:
                    best_frames.pop(tid, None)

            if show:
                vis = cv2.addWeighted(frame, 0.85, zone_ov, 0.15, 0)

                cv2.line(vis, pa_t, pb_t, (0, 40, 220), 3)
                cv2.circle(vis, pa_t, 7, (0, 180, 255), -1)
                cv2.circle(vis, pb_t, 7, (0,  40, 220), -1)

                for tid, (bbox, state) in draw_ids.items():
                    if bbox is None: continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = COLOR.get(state, (180, 180, 180))
                    lbl   = LABEL.get(state, "")
                    thick = 3 if state == S.S_VIOLATED else 2

                    # Ghost: ve bbox mo hon
                    if state == S.S_GHOST:
                        ghost_vis = vis.copy()
                        cv2.rectangle(ghost_vis, (x1,y1), (x2,y2), color, thick)
                        cv2.addWeighted(ghost_vis, 0.5, vis, 0.5, 0, vis)
                    else:
                        cv2.rectangle(vis, (x1,y1), (x2,y2), color, thick)

                    bc = ((x1+x2)//2, y2)
                    cv2.circle(vis, bc, 5, color, -1)

                    cls_name = VEHICLE_NAMES.get(track_cls.get(tid, -1), "")
                    tag = f"ID:{tid} {cls_name} {lbl}".strip()
                    (tw, th), _ = cv2.getTextSize(
                        tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    lx, ly = x1, max(y1-6, 16)
                    cv2.rectangle(vis, (lx, ly-th-4), (lx+tw+6, ly+2), (0,0,0), -1)
                    cv2.putText(vis, tag, (lx+3, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                # HUD
                n_track = sum(1 for _, s in draw_ids.values()
                              if s == S.S_TRACKING)
                hud = (f"Frame {frame_idx}/{total}  |  "
                       f"Trong lan: {n_track}  |  Vi pham: {saver.count}")
                (hw, hh), _ = cv2.getTextSize(
                    hud, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(vis, (0,0), (hw+16, 44), (0,0,0), -1)
                cv2.putText(vis, hud, (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,230,0), 2)

                try:
                    cv2.imshow(win, vis)
                    elapsed = time.perf_counter() - t0
                    wait_ms = max(1, int((spf - elapsed) * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # GUI not supported (headless)
                    if frame_idx == 0:
                        print("[WARNING] OpenCV GUI khong duoc ho tro (Headless mode).")
                    show = False # Tu dong tat show neu loi

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"  frame {frame_idx}/{total}"
                      f" | trong lan: {len(draw_ids)}"
                      f" | vi pham: {saver.count}")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Dung boi nguoi dung.")
    finally:
        reader.stop()
        
        #  POST-PROCESS
        # (Da loai bo ocr, khong can lam gi o day nua)

        try:
            cv2.destroyAllWindows()
        except:
            pass

    print(f"\n[DONE] Tong vi pham : {saver.count}")
    print(f"[DONE] Anh luu tai  : {os.path.abspath(saver.out)}/")


#  ENTRY POINT
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Wrong Lane Detector "
    )
    ap.add_argument("--draw", action="store_true",
                    help="Mo GUI ve polygon + violation line")
    ap.add_argument("--run",  action="store_true",
                    help="Chay detection")
    ap.add_argument("--show", action="store_true",
                    help="Hien thi preview khi chay")
    args = ap.parse_args()

    if args.draw:
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read(); cap.release()
        if not ret:
            print(f"[ERROR] Khong doc duoc video: {VIDEO_PATH}")
        else:
            d = Drawer(frame)
            poly, pa, pb = d.run()
            if poly:
                print(f"[DRAW] Polygon: {len(poly)} diem")
                print(f"[DRAW] Vline  : {pa} -> {pb}")
    elif args.run:
        run_detection(show=args.show)
    else:
        print(__doc__)