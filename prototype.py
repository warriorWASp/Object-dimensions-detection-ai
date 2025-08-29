# realtime_measure_fixed.py
# -------------------------------------------------------------
# Real-time object detection + monocular depth + size estimates
# with YOLOv11 (Ultralytics) and MiDaS (intel-isl).
#
# This version fixes transform type errors: MiDaS transforms returned
# by torch.hub may expect a NumPy float array (so `img/255.0` works).
# The DepthEstimator.infer method now supports:
#   - transforms that return a dict with key "image" (NumPy -> tensor)
#   - transforms that return a torch.Tensor directly
#
# Save as a single file and run with Python 3.9+ (or your existing Python).
# -------------------------------------------------------------

import os
import cv2
import time
import math
import queue
import torch
import warnings
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIG ----------------
YOLO_MODEL = "yolo11n.pt"   # Use yolo11x.pt for higher accuracy (slower)
MIDAS_MODEL_TYPE = "DPT_Hybrid"  # "DPT_Large", "DPT_Hybrid", "MiDaS_small"

# Camera
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
USE_DIRECTSHOW = True  # Windows-specific: helps certain webcams open reliably

# Approx camera horizontal field-of-view in degrees (set if you know it)
CAM_HFOV_DEG = 70.0

# If you know focal length in pixels, set this to override HFOV-based estimate.
FOCAL_LENGTH_PX: Optional[float] = None

# Calibration object and approximate physical width (meters)
KNOWN_OBJ_CLASS = "person"
KNOWN_OBJ_WIDTH_M = 0.45

# Depth and detection tuning
DEPTH_PERCENTILE_LOW = 10
DEPTH_PERCENTILE_HIGH = 90
EMA_ALPHA = 0.25
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Visualization toggles
DRAW_DEPTH_OVERLAY = True
SHOW_INFO_OVERLAY = True

# Threading queues
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Utilities ----------------
def compute_focal_from_hfov_px(img_width: int, hfov_deg: float) -> float:
    hfov_rad = math.radians(hfov_deg)
    return (img_width / 2.0) / math.tan(hfov_rad / 2.0)

def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1:
        x2 = min(x1 + 1, w - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, h - 1)
    return x1, y1, x2, y2

def robust_box_depth_stats(depth_map: np.ndarray, box: Tuple[int,int,int,int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return float("nan"), float("nan")
    d = patch.astype(np.float32)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    lo = np.percentile(d, DEPTH_PERCENTILE_LOW)
    hi = np.percentile(d, DEPTH_PERCENTILE_HIGH)
    d = d[(d >= lo) & (d <= hi)]
    if d.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(d))
    q1 = np.percentile(d, 25.0)
    q3 = np.percentile(d, 75.0)
    iqr = float(q3 - q1)
    return med, iqr

def make_depth_colormap(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(d == 0):
        d_norm = d
    else:
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d8 = (d_norm * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)
    return cm

@dataclass
class MeasureScale:
    depth_to_m: float = 1.0
    last_calib_time: float = 0.0
    def describe(self) -> str:
        return f"{self.depth_to_m:.4f} m/unit"

class DepthEstimator:
    def __init__(self, model_type: str, device: torch.device):
        self.device = device
        try:
            # Load model (this may download files on first run)
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MiDaS model '{model_type}' via torch.hub. "
                f"On first run this requires internet to download weights. Error: {e}"
            )
        self.model.to(device).eval()
        try:
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        except Exception as e:
            raise RuntimeError(f"Failed to load MiDaS transforms via torch.hub: {e}")

        if "DPT" in model_type:
            self.transform = self.transforms.dpt_transform
        else:
            self.transform = self.transforms.small_transform

        # Use autocast on CUDA for speed
        self.use_amp = device.type == "cuda"

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """
        Return depth map (H, W) as float32. This method accepts different
        transform behaviours:
          - transform returns a torch.Tensor (C,H,W) -> use directly
          - transform returns a dict like {"image": tensor} -> use that tensor
          - transform expects a NumPy float image array (so dividing by 255 works)
        """
        # Convert BGR -> RGB numpy array
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Many MiDaS transforms expect PIL.Image, some expect numpy arrays.
        # We'll try to call transform with a NumPy float32 array first,
        # then fall back to PIL image if needed.
        input_tensor = None
        transform_result = None

        # 1) Try numpy float array path (works if transform lambda uses img/255.0)
        try:
            img_np = np.asarray(rgb).astype(np.float32)
            transform_result = self.transform(img_np)
        except Exception:
            transform_result = None

        # 2) If previous didn't produce, try PIL.Image path
        if transform_result is None:
            try:
                pil_img = Image.fromarray(rgb)
                transform_result = self.transform(pil_img)
            except Exception:
                transform_result = None

        if transform_result is None:
            raise RuntimeError("MiDaS transform failed on both NumPy array and PIL image inputs. "
                               "Inspect the transform signature or torch.hub availability.")

        # Now interpret transform_result which can be:
        #  - torch.Tensor (C,H,W) or (1,C,H,W)
        #  - dict with key "image" that contains the tensor
        #  - dict with other structure (rare)
        if isinstance(transform_result, dict):
            # Common case: {"image": tensor}
            if "image" in transform_result:
                input_tensor = transform_result["image"]
            else:
                # Try to find a tensor value inside the dict
                found = None
                for v in transform_result.values():
                    if torch.is_tensor(v):
                        found = v
                        break
                if found is None:
                    raise RuntimeError("MiDaS transform returned a dict but no tensor found inside.")
                input_tensor = found
        elif torch.is_tensor(transform_result):
            input_tensor = transform_result
        else:
            raise RuntimeError(f"Unsupported transform_result type: {type(transform_result)}")

        # Ensure batch dim and correct device
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Forward
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(input_tensor)
            else:
                pred = self.model(input_tensor)

            # pred shape: (B, 1, H', W') or (B, H', W')
            # Convert to (H, W) in CPU and upsample to original image size
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            pred = torch.nn.functional.interpolate(
                pred,
                size=bgr.shape[:2],
                mode="bicubic",
                align_corners=False
            )
            pred = pred.squeeze(0).cpu().float().numpy()

        return pred

class YoloDetector:
    def __init__(self, model_path: str, device: torch.device):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_path}'. Confirm the file exists or allow Ultralytics to download. Error: {e}"
            )
        # Attempt to put model on device if possible
        try:
            if device.type == "cuda":
                self.model.to(device)
        except Exception:
            pass
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect_or_track(self, frame_bgr: np.ndarray):
        # Use track for stable ids where available
        # Ultralytics track API can accept numpy arrays directly
        results = self.model.track(
            source=frame_bgr,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            persist=True,
            verbose=False,
            stream=False
        )
        # model.track returns a sequence; return the first result
        return results[0]

# ---------------- Global state ----------------
g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_scale = MeasureScale()
g_focal_px = None
g_toggle_depth_overlay = DRAW_DEPTH_OVERLAY
g_toggle_info = SHOW_INFO_OVERLAY
track_smoothers: Dict[int, Dict[str, Any]] = {}

def ema_update(prev: Optional[float], new: float, alpha: float) -> float:
    if prev is None or prev != prev:  # check NaN using prev != prev
        return new
    return alpha * new + (1 - alpha) * prev

# ---------------- Worker ----------------
def processing_worker(yolo: YoloDetector, midas: DepthEstimator):
    global g_scale, g_focal_px, g_device, g_toggle_depth_overlay, g_toggle_info
    last_time = time.time()
    fps = 0.0

    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame = item

        start = time.time()
        try:
            det_result = yolo.detect_or_track(frame)
        except Exception as e:
            # detection failure; push empty result
            det_result = type("R", (), {"boxes": None})
            print(f"[Warn] YOLO detection failed: {e}")

        try:
            depth_map = midas.infer(frame)
        except Exception as e:
            print(f"[Warn] MiDaS inference failed: {e}")
            # create fallback zero-depth map
            depth_map = np.zeros(frame.shape[:2], dtype=np.float32)

        infer_time = time.time() - start

        H, W = frame.shape[:2]
        if g_focal_px is None:
            if FOCAL_LENGTH_PX is not None:
                g_focal_px = float(FOCAL_LENGTH_PX)
            elif CAM_HFOV_DEG is not None:
                g_focal_px = compute_focal_from_hfov_px(W, CAM_HFOV_DEG)
            else:
                g_focal_px = compute_focal_from_hfov_px(W, 70.0)

        annotated = frame.copy()
        if g_toggle_depth_overlay:
            cm = make_depth_colormap(depth_map)
            annotated = cv2.addWeighted(annotated, 0.6, cm, 0.4, 0)

        boxes = getattr(det_result, "boxes", None)
        calib_candidate = None

        if boxes is not None and len(boxes) > 0:
            try:
                # Ultralytics Boxes object: .xyxy, .conf, .cls, maybe .id
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                # In some versions, boxes.xyxy may already be numpy
                try:
                    xyxy = np.array(boxes.xyxy)
                except Exception:
                    xyxy = []

            conf = []
            cls = []
            ids = None
            try:
                conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            except Exception:
                conf = np.zeros(len(xyxy))
            try:
                cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            except Exception:
                cls = np.zeros(len(xyxy), dtype=int)
            try:
                if hasattr(boxes, "id") and boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                ids = None

            # find largest known candidate for calibration
            largest_known = (-1, None)
            for i, coords in enumerate(xyxy):
                x1, y1, x2, y2 = coords
                x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)
                area = (x2 - x1) * (y2 - y1)
                name = yolo.names.get(int(cls[i]), str(int(cls[i])))
                if name == KNOWN_OBJ_CLASS:
                    med_d, _ = robust_box_depth_stats(depth_map, (x1, y1, x2, y2))
                    if not math.isnan(med_d) and area > largest_known[0]:
                        largest_known = (area, (x1, y1, x2, y2, med_d))
            calib_candidate = largest_known[1]
            det_result.__dict__["_calib_candidate"] = calib_candidate

            # Draw boxes
            for i, coords in enumerate(xyxy):
                x1, y1, x2, y2 = coords
                x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)
                w_px = x2 - x1
                h_px = y2 - y1
                if w_px <= 1 or h_px <= 1:
                    continue

                med_depth, iqr_depth = robust_box_depth_stats(depth_map, (x1, y1, x2, y2))
                if math.isnan(med_depth):
                    continue

                tid = int(ids[i]) if ids is not None and i < len(ids) else -1
                sm = track_smoothers.get(tid, {"depth": None, "W": None, "H": None})
                med_depth_sm = ema_update(sm["depth"], med_depth, EMA_ALPHA)
                track_smoothers[tid] = {**sm, "depth": med_depth_sm}

                depth_m = med_depth_sm * g_scale.depth_to_m

                W_est = (w_px * depth_m) / (g_focal_px + 1e-9)
                H_est = (h_px * depth_m) / (g_focal_px + 1e-9)
                W_est_sm = ema_update(sm.get("W"), W_est, EMA_ALPHA)
                H_est_sm = ema_update(sm.get("H"), H_est, EMA_ALPHA)
                track_smoothers[tid].update({"W": W_est_sm, "H": H_est_sm})

                name = yolo.names.get(int(cls[i]), str(int(cls[i])))
                conf_i = float(conf[i]) if i < len(conf) else 0.0

                thickness = max(2, int(0.002 * (W + H)))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 128), thickness)

                id_part = f" id:{tid}" if tid >= 0 else ""
                scale_suffix = "m" if g_scale.depth_to_m != 1.0 else "u"
                label_top = f"{name}{id_part} {conf_i:.2f}"
                label_bot = f"W:{W_est_sm:.2f}{scale_suffix} H:{H_est_sm:.2f}{scale_suffix} D:{depth_m:.2f}{'m' if g_scale.depth_to_m != 1.0 else 'u'}"

                (tw, th), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 0, 0), -1)
                cv2.putText(annotated, label_top, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                (tw2, th2), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                y_text = y2 + th2 + 12
                y_text = min(y_text, H - 5)
                cv2.rectangle(annotated, (x1, y_text - th2 - 8), (x1 + tw2 + 8, y_text), (0, 0, 0), -1)
                cv2.putText(annotated, label_bot, (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
        last_time = now

        if g_toggle_info:
            info_lines = [
                f"Device: {g_device.type.upper()}  |  YOLO:{YOLO_MODEL}  MiDaS:{MIDAS_MODEL_TYPE}",
                f"FPS: {fps:5.1f}  |  Inference: {infer_time*1000:5.1f} ms",
                f"Focal(px): {g_focal_px:.1f}  |  Scale: {g_scale.describe()}",
                "Keys: [q]=quit  [d]=toggle depth  [i]=toggle info  [c]=calibrate"
            ]
            y0 = 28
            for line in info_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (10, y0 - th - 6), (12 + tw, y0 + 4), (0, 0, 0), -1)
                cv2.putText(annotated, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y0 += th + 12

        try:
            result_queue.get_nowait()
        except Exception:
            pass
        result_queue.put((annotated, depth_map, calib_candidate))

# ---------------- Main ----------------
def main():
    global g_toggle_depth_overlay, g_toggle_info, g_scale, g_device

    print(f"Using device: {g_device}")
    print("Loading YOLO model...")
    yolo = YoloDetector(YOLO_MODEL, g_device)
    print("Loading MiDaS model...")
    midas = DepthEstimator(MIDAS_MODEL_TYPE, g_device)
    print("Models loaded.")

    open_flags = cv2.CAP_DSHOW if (USE_DIRECTSHOW and hasattr(cv2, "CAP_DSHOW")) else 0
    cap = cv2.VideoCapture(CAM_INDEX + open_flags)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera. Try a different CAM_INDEX or backend.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    time.sleep(0.2)

    worker = threading.Thread(target=processing_worker, args=(yolo, midas), daemon=True)
    worker.start()

    calib_tuple_cache = None
    print("Starting stream. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera frame not received. Exiting.")
            break

        if frame_queue.empty():
            try:
                frame_queue.put_nowait(frame)
            except Exception:
                pass

        if not result_queue.empty():
            annotated, depth_map, calib_candidate = result_queue.get()
            calib_tuple_cache = calib_candidate
            cv2.imshow("3D Object Measurement (fixed)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            g_toggle_depth_overlay = not g_toggle_depth_overlay
        elif key == ord('i'):
            g_toggle_info = not g_toggle_info
        elif key == ord('c'):
            if calib_tuple_cache is not None:
                x1, y1, x2, y2, med_depth = calib_tuple_cache
                if not math.isnan(med_depth) and (x2 > x1) and (y2 > y1):
                    w_px = x2 - x1
                    focal = (FOCAL_LENGTH_PX if FOCAL_LENGTH_PX is not None
                             else compute_focal_from_hfov_px(frame.shape[1], CAM_HFOV_DEG))
                    new_scale = (KNOWN_OBJ_WIDTH_M * focal) / (max(1.0, w_px) * med_depth)
                    if 1e-4 < new_scale < 10.0:
                        g_scale.depth_to_m = float(new_scale)
                        g_scale.last_calib_time = time.time()
                        print(f"[Calibration] Scale set to {g_scale.describe()} using {KNOWN_OBJ_CLASS}")
                    else:
                        print("[Calibration] Ignored absurd scale estimate. Try again.")
                else:
                    print("[Calibration] No valid candidate depth; try again.")
            else:
                print(f"[Calibration] No {KNOWN_OBJ_CLASS} found. Make sure it's visible and try again.")

    try:
        frame_queue.put_nowait(None)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
