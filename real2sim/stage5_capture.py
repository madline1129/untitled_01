"""Stage 5: camera capture, image similarity metrics, and snapshot.

Receives the pre-spawned ego actor from Stage 4, mounts six RGB cameras
(CAM_FRONT, CAM_FRONT_LEFT, …, CAM_BACK_RIGHT) plus a top-down BEV camera,
ticks the world, saves the rendered images, and computes SSIM/PSNR/LPIPS
against the ground-truth nuScenes images.
"""
import logging
import math
import time

import carla
import cv2
import lpips
import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from scipy.ndimage import rotate as ndi_rotate
from scipy.spatial.transform import Rotation as Rot
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from real2sim.core import (
    EGO_REAR_AXLE_TO_CENTER,
    NuscMeta,
    get_cam_front,
    log_json,
    nusc_ego_centre,
    nusc_quaternion_yaw,
)
from real2sim.stage3_local_map import LAYER_NAMES
from real2sim.stage4_spawn import (
    EGO_BLUEPRINT,
    IMAGE_H,
    IMAGE_W,
    NUSC_TO_CARLA,
    SKIP_CATEGORIES,
)

# BEV rendering resolution — matches the CARLA BEV camera (z=50, fov=90, 1600×1600 → ~100m × 100m, ~16 px/m).
_NUSC_BEV_PATCH_SIZE = (100, 100)
_NUSC_BEV_CANVAS_SIZE = (1600, 1600)


logger = logging.getLogger("real2sim")
CAMERA_CHANNELS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAMERA_VIEWS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
_lpips_model = None


def quat_mult(q1, q2):
    """Multiply two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_to_rot_matrix(q):
    """Convert (w, x, y, z) quaternion to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def get_box_corners(translation, size, rotation):
    """Return 8 corners of a 3D box given centre *translation*, *size* (w, l, h), and *rotation* quaternion."""
    w, l, h = size
    corners = np.array([
        [w / 2, l / 2, h / 2], [-w / 2, l / 2, h / 2],
        [-w / 2, -l / 2, h / 2], [w / 2, -l / 2, h / 2],
        [w / 2, l / 2, -h / 2], [-w / 2, l / 2, -h / 2],
        [-w / 2, -l / 2, -h / 2], [w / 2, -l / 2, -h / 2],
    ])
    R = quat_to_rot_matrix(rotation)
    return (R @ corners.T).T + np.array(translation)


def inverse_transform(points, translation, rotation):
    """Transform *points* from global frame to a local frame defined by *translation* + *rotation* quaternion."""
    R = quat_to_rot_matrix(rotation)
    return (R.T @ (points - np.array(translation)).T).T


def project_to_image(points_3d, intr):
    """Project 3D camera-frame points to 2D pixel coordinates using intrinsic matrix *intr*."""
    fx, fy = intr[0][0], intr[1][1]
    cx, cy = intr[0][2], intr[1][2]
    u = fx * points_3d[:, 0] / points_3d[:, 2] + cx
    v = fy * points_3d[:, 1] / points_3d[:, 2] + cy
    return np.column_stack([u, v])


def get_lidar_calibration(meta: NuscMeta):
    """Return the calibrated_sensor record for LIDAR_TOP."""
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    for cs in meta.calibrated_sensors:
        if sensor_to_channel.get(cal_to_sensor.get(cs["token"], ""), "") == "LIDAR_TOP":
            return cs
    raise RuntimeError("LIDAR_TOP calibration not found")


def _build_sample_cameras(meta, target_sample):
    """Build a dict {channel_name → sample_data record} for all 6 cameras of the target sample."""
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    result = {}
    for sd in meta.sample_data:
        if sd["sample_token"] != target_sample or not sd.get("is_key_frame"):
            continue
        ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
        if ch in CAMERA_CHANNELS:
            result[ch] = sd
    return result


def _compute_camera_spec(cs, nusc_ego_pos, nusc_ego_rot):
    """Convert a nuScenes calibrated sensor record to a CARLA camera transform + intrinsics."""
    t = cs["translation"]
    q = cs["rotation"]
    t_carla = np.array([t[0] - EGO_REAR_AXLE_TO_CENTER, -t[1], t[2]])
    r_nu = Rot.from_quat([q[1], q[2], q[3], q[0]])
    fwd_nu = r_nu.apply([0, 0, 1])
    fwd_carla = np.array([fwd_nu[0], -fwd_nu[1], fwd_nu[2]])
    fx = cs["camera_intrinsic"][0][0]
    fov = 2 * math.degrees(math.atan(IMAGE_W / (2 * fx)))
    return {
        "x": float(t_carla[0]),
        "y": float(t_carla[1]),
        "z": float(t_carla[2]),
        "pitch": float(np.degrees(np.arctan2(-fwd_carla[2], np.linalg.norm(fwd_carla[:2])))),
        "yaw": float(np.degrees(np.arctan2(fwd_carla[1], fwd_carla[0]))),
        "fov": float(fov),
        "intrinsic": cs["camera_intrinsic"],
        "translation_nusc": t,
        "rotation_nusc": q,
    }


def _find_nusc_image_paths(meta, target_sample, nusc_root):
    """Return {view_name → filesystem_path} for all 6 camera images of the target sample."""
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    result = {}
    for sd in meta.sample_data:
        if sd["sample_token"] != target_sample or not sd.get("is_key_frame"):
            continue
        ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
        if ch in CAMERA_CHANNELS:
            result[ch.replace("CAM_", "")] = nusc_root / sd["filename"]
    return result


def _compute_view_metrics(arr_a: np.ndarray, arr_b: np.ndarray, downsample: int = 2) -> dict:
    """SSIM, PSNR, and LPIPS between two downsampled RGB images."""
    global _lpips_model
    h = min(arr_a.shape[0], arr_b.shape[0])
    w = min(arr_a.shape[1], arr_b.shape[1])
    a = arr_a[:h, :w].astype(np.float32)
    b = arr_b[:h, :w].astype(np.float32)
    h_small = h // downsample
    w_small = w // downsample
    a = np.array(Image.fromarray(a.astype(np.uint8)).resize((w_small, h_small), Image.BILINEAR), dtype=np.float32)
    b = np.array(Image.fromarray(b.astype(np.uint8)).resize((w_small, h_small), Image.BILINEAR), dtype=np.float32)
    result = {"resolution": [h_small, w_small], "downsample": downsample}
    try:
        result["ssim"] = round(float(ssim(a, b, channel_axis=-1, data_range=255)), 6)
    except Exception:
        result["ssim"] = None
    try:
        result["psnr"] = round(float(psnr(a, b, data_range=255)), 4)
    except Exception:
        result["psnr"] = None
    try:
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex", verbose=False)
            if torch.cuda.is_available():
                _lpips_model = _lpips_model.to("cuda")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ta = torch.from_numpy(a).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        tb = torch.from_numpy(b).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        ta, tb = ta.to(device), tb.to(device)
        with torch.no_grad():
            dist = _lpips_model(ta, tb, normalize=True).item()
        result["lpips"] = round(float(dist), 6)
    except Exception:
        result["lpips"] = None
    return result


def _compute_similarity_metrics(meta, target_sample, nusc_root, carla_images, out_dir=None):
    """Compute SSIM/PSNR/LPIPS per view between nuScenes ground truth and CARLA renders."""
    nusc_paths = _find_nusc_image_paths(meta, target_sample, nusc_root)
    results = {}
    for view in CAMERA_VIEWS:
        nusc_path = nusc_paths.get(view)
        carla_arr = carla_images.get(view)
        if not nusc_path or not nusc_path.exists() or carla_arr is None:
            continue
        img_a = np.array(Image.open(nusc_path).convert("RGB"), dtype=np.uint8)
        results[view] = _compute_view_metrics(img_a, carla_arr)
    if results:
        results["_summary"] = {
            "ssim_mean": round(float(np.mean([r["ssim"] for r in results.values() if r.get("ssim") is not None])), 6),
            "psnr_mean": round(float(np.mean([r["psnr"] for r in results.values() if r.get("psnr") is not None])), 4),
            "lpips_mean": round(float(np.mean([r["lpips"] for r in results.values() if r.get("lpips") is not None])), 6),
        }
    if out_dir is not None:
        log_json(results, out_dir / "similarity_metrics.json", "image similarity metrics")
    return results


def _project_boxes_on_image(image_rgb, meta, target_sample, calib_token, ego_pose_token):
    """Draw nuScenes 3D annotation boxes projected onto a camera image."""
    cs = next(c for c in meta.calibrated_sensors if c["token"] == calib_token)
    ep = next(e for e in meta.ego_poses if e["token"] == ego_pose_token)
    intrinsic = cs["camera_intrinsic"]
    lidar_cal = get_lidar_calibration(meta)
    Q_LIDAR = lidar_cal["rotation"]
    cat_dict = meta.cat_dict()
    inst_dict = meta.inst_dict()
    sample_anns = [a for a in meta.annotations if a["sample_token"] == target_sample]

    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    box_colors = {
        "human.pedestrian": (255, 0, 0),
        "vehicle": (0, 255, 0),
        "movable_object": (255, 255, 0),
        "static_object": (255, 165, 0),
    }
    BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    for ann in sample_anns:
        inst = inst_dict.get(ann["instance_token"], {})
        cat_name = cat_dict.get(inst.get("category_token", ""), "unknown")
        if cat_name in SKIP_CATEGORIES:
            continue
        rotation = quat_mult(Q_LIDAR, ann["rotation"])
        corners_global = get_box_corners(ann["translation"], ann["size"], rotation)
        corners_ego = inverse_transform(corners_global, ep["translation"], ep["rotation"])
        corners_cam = inverse_transform(corners_ego, cs["translation"], cs["rotation"])
        if np.all(corners_cam[:, 2] <= 0):
            continue
        pts_2d = project_to_image(corners_cam, intrinsic)
        w, h = img.size
        visible = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h))
        if np.sum(visible) < 2:
            continue
        color = (0, 255, 0)
        for prefix, col in box_colors.items():
            if cat_name.startswith(prefix):
                color = col
                break
        for i, j in BOX_EDGES:
            draw.line([(pts_2d[i][0], pts_2d[i][1]), (pts_2d[j][0], pts_2d[j][1])], fill=color, width=2)
        front_mask = corners_cam[:, 2] > 0
        if np.any(front_mask):
            closest = np.where(front_mask)[0][np.argmin(corners_cam[front_mask, 2])]
            bp_id = NUSC_TO_CARLA.get(cat_name, cat_name.split(".")[-1])
            draw.text((pts_2d[closest][0] + 3, pts_2d[closest][1] - 10), bp_id.split(".")[-1], fill=color)
    return np.array(img)


# def _render_nusc_bev(meta: NuscMeta, target_sample: str, nusc_root):
#     """Render a bird's-eye view of the nuScenes local map with annotation dots."""
#     cam_front = get_cam_front(meta, target_sample)
#     ep = meta.ego_dict()[cam_front["ego_pose_token"]]
#     s = next(s for s in meta.samples if s["token"] == target_sample)
#     scene = next(sc for sc in meta.scenes if sc["token"] == s["scene_token"])
#     log = next(l for l in meta.logs if l["token"] == scene["log_token"])
#     location = log["location"]
#     ego_x, ego_y, _ = ep["translation"]
#     q = Quaternion(ep["rotation"])
#     ego_yaw = nusc_quaternion_yaw(q)
#     fwd_nusc = (math.cos(ego_yaw), math.sin(ego_yaw))
#     nusc_map = NuScenesMap(dataroot=str(nusc_root), map_name=location)
#     M = nusc_map.get_map_mask([ego_x, ego_y, PATCH_SIZE[0], PATCH_SIZE[1]], 0, LAYER_NAMES, CANVAS_SIZE)
#     M = np.flip(M, axis=1)  # image Y-down → global Y-up
#     h, w = M.shape[1:]
#     img = np.full((h, w, 3), 50, dtype=np.uint8)
#     img[M[0] > 0] = (60, 80, 50)  # drivable_area → olive
#     img[M[1] > 0] = (200, 200, 200)  # road_divider → light gray
#     img[M[2] > 0] = (150, 150, 150)  # lane_divider → mid gray
#     m_per_px = PATCH_SIZE[0] / CANVAS_SIZE[0]
#     inst_dict = meta.inst_dict()
#     cat_dict = meta.cat_dict()
#     sample_anns = [a for a in meta.annotations if a["sample_token"] == target_sample]
#     for ann in sample_anns:
#         inst = inst_dict.get(ann["instance_token"], {})
#         cat_name = cat_dict.get(inst.get("category_token", ""), "unknown")
#         if cat_name in SKIP_CATEGORIES:
#             continue
#         ax, ay, _ = ann["translation"]
#         dx, dy = ax - ego_x, ay - ego_y
#         px = int(w / 2 + dx / m_per_px)
#         py = int(h / 2 - dy / m_per_px)
#         if not (0 <= px < w and 0 <= py < h):
#             continue
#         color = (0, 255, 0)  # vehicle → green
#         if cat_name.startswith("human.pedestrian"):
#             color = (0, 0, 255)  # pedestrian → blue
#         elif cat_name.startswith("movable_object"):
#             color = (0, 255, 255)  # static obstacles → cyan
#         r = max(2, int(ann["size"][0] / m_per_px / 2))
#         cv2.circle(img, (px, py), r, color, -1)
#     cx, cy = w // 2, h // 2
#     cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)  # ego dot → red
#     fwd_x, fwd_y = fwd_nusc[0], fwd_nusc[1]
#     line_len_px = int(15 / m_per_px)
#     end_x = int(cx + line_len_px * fwd_x)
#     end_y = int(cy - line_len_px * fwd_y)
#     cv2.line(img, (cx, cy), (end_x, end_y), (0, 0, 255), 2)  # forward arrow → blue
#     return img


import math
import numpy as np
import cv2
from pyquaternion import Quaternion

def _render_nusc_bev(meta: NuscMeta, target_sample: str, nusc_root):
    """Axis-aligned (north-up) nuScenes BEV rendering centred on the ego vehicle centre.

    Renders at the same resolution as the CARLA BEV camera (~16 px/m, 100×100 m → 1600×1600).
    """
    cam_front = get_cam_front(meta, target_sample)
    ep = meta.ego_dict()[cam_front["ego_pose_token"]]
    
    s = next(s for s in meta.samples if s["token"] == target_sample)
    scene = next(sc for sc in meta.scenes if sc["token"] == s["scene_token"])
    log = next(l for l in meta.logs if l["token"] == scene["log_token"])
    location = log["location"]
    
    rear_x, rear_y, rear_z = ep["translation"]
    q = Quaternion(ep["rotation"])
    
    cx_global, cy_global, _ = nusc_ego_centre((rear_x, rear_y, rear_z), q)
    
    fwd_nusc = q.rotate([1, 0, 0])
    
    nusc_map = NuScenesMap(dataroot=str(nusc_root), map_name=location)
    
    ps = _NUSC_BEV_PATCH_SIZE
    cs = _NUSC_BEV_CANVAS_SIZE
    M = nusc_map.get_map_mask([cx_global, cy_global, ps[0], ps[1]], 0, LAYER_NAMES, cs)
    M = np.flip(M, axis=1)
    
    h, w = M.shape[1:]
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    img[M[0] > 0] = (60, 80, 50)
    img[M[1] > 0] = (200, 200, 200)
    img[M[2] > 0] = (150, 150, 150)
    
    m_per_px = ps[0] / cs[0]
    cx, cy = w // 2, h // 2
    
    inst_dict = meta.inst_dict()
    cat_dict = meta.cat_dict()
    sample_anns = [a for a in meta.annotations if a["sample_token"] == target_sample]
    
    # Fixed-size box (render pixels) — stays visible after 1600→400 resize
    _BEV_BOX_HALF = 12
    
    for ann in sample_anns:
        inst = inst_dict.get(ann["instance_token"], {})
        cat_name = cat_dict.get(inst.get("category_token", ""), "unknown")
        if cat_name in SKIP_CATEGORIES:
            continue
            
        ax, ay, _ = ann["translation"]
        dx, dy = ax - cx_global, ay - cy_global
        
        px = int(cx + dx / m_per_px)
        py = int(cy - dy / m_per_px)
        
        if not (0 <= px < w and 0 <= py < h):
            continue
            
        # Colour map: pedestrian=red, vehicle=blue, movable=yellow
        if cat_name.startswith("human.pedestrian"):
            color = (0, 0, 255)
        elif cat_name.startswith("movable_object"):
            color = (0, 255, 255)
        else:
            color = (255, 0, 0)
            
        pt1 = (px - _BEV_BOX_HALF, py - _BEV_BOX_HALF)
        pt2 = (px + _BEV_BOX_HALF, py + _BEV_BOX_HALF)
        cv2.rectangle(img, pt1, pt2, color, -1)
        
    # Ego at centre (green rectangle)
    pt1 = (cx - _BEV_BOX_HALF, cy - _BEV_BOX_HALF)
    pt2 = (cx + _BEV_BOX_HALF, cy + _BEV_BOX_HALF)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), -1)
    
    # Forward yaw indicator (green line, ¼ canvas width)
    fwd_x, fwd_y = fwd_nusc[0], fwd_nusc[1]
    yaw_len = w // 4
    end_x = int(cx + yaw_len * fwd_x)
    end_y = int(cy - yaw_len * fwd_y)
    
    cv2.line(img, (cx, cy), (end_x, end_y), (0, 255, 0), 4)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _create_comparison_montage(meta, target_sample, nusc_root, carla_images, out_dir, best_rotation: float = 0):
    """Save a 6-view side-by-side montage (nuScenes | CARLA) plus BEV comparison."""
    try:
        nusc_paths = _find_nusc_image_paths(meta, target_sample, nusc_root)
        sample_cams = _build_sample_cameras(meta, target_sample)
        tile_w, tile_h = IMAGE_W // 4, IMAGE_H // 4
        col_tiles = []
        for view in CAMERA_VIEWS:
            ch = f"CAM_{view}"
            sd = sample_cams.get(ch)
            path = nusc_paths.get(view)
            if path and path.exists():
                img_nusc = np.array(Image.open(path))
            else:
                img_nusc = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8) + 64
            if sd is not None:
                nusc_tile = _project_boxes_on_image(img_nusc, meta, target_sample, sd["calibrated_sensor_token"], sd["ego_pose_token"])
            else:
                nusc_tile = img_nusc
            nusc_tile = np.array(Image.fromarray(nusc_tile).resize((tile_w, tile_h)))

            carla_arr = carla_images.get(view)
            if carla_arr is not None and sd is not None:
                carla_tile = _project_boxes_on_image(carla_arr, meta, target_sample, sd["calibrated_sensor_token"], sd["ego_pose_token"])
                carla_tile = np.array(Image.fromarray(carla_tile).resize((tile_w, tile_h)))
            elif carla_arr is not None:
                carla_tile = np.array(Image.fromarray(carla_arr).resize((tile_w, tile_h)))
            else:
                carla_tile = np.full((tile_h, tile_w, 3), 64, dtype=np.uint8)
            col_tiles.append((nusc_tile, carla_tile))

        row_nusc = np.hstack([col_tiles[i][0] for i in range(6)])
        row_carla = np.hstack([col_tiles[i][1] for i in range(6)])
        col_bar_h = 20
        col_bar = np.ones((col_bar_h, row_nusc.shape[1], 3), dtype=np.uint8) * 30
        for i, view in enumerate(CAMERA_VIEWS):
            x = i * tile_w + 8
            cv2.putText(col_bar, view, (x, col_bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        nusc_label = np.ones((24, row_nusc.shape[1], 3), dtype=np.uint8) * 50
        carla_label = np.ones((24, row_carla.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(nusc_label, "nuScenes", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(carla_label, "CARLA", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        montage = np.vstack([col_bar, nusc_label, row_nusc, carla_label, row_carla])
        Image.fromarray(montage).save(out_dir / "comparison_montage.png")

        nusc_bev = _render_nusc_bev(meta, target_sample, nusc_root)
        display_h = 400
        nusc_disp = cv2.resize(nusc_bev, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        nusc_rot = ndi_rotate(nusc_bev, best_rotation, order=0, reshape=False, mode="constant", cval=50)
        nusc_rot_disp = cv2.resize(nusc_rot, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        gap = np.full((display_h, 10, 3), 40, dtype=np.uint8)
        carla_bev = carla_images.get("BEV")
        if carla_bev is not None:
            carla_disp = cv2.resize(carla_bev, (display_h, display_h))
            img_row = np.hstack([nusc_disp, gap, nusc_rot_disp, gap, carla_disp])
            total_w = display_h * 3 + 20
            labels = ["nusc (axis)", f"nusc (rot={best_rotation})", "carla BEV"]
        else:
            img_row = np.hstack([nusc_disp, gap, nusc_rot_disp])
            total_w = display_h * 2 + 10
            labels = ["nusc (axis)", f"nusc (rot={best_rotation})"]
        col_bar = np.full((24, total_w, 3), 40, dtype=np.uint8)
        for i, text in enumerate(labels):
            x = 6 + i * (display_h + 10)
            cv2.putText(col_bar, text, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        bev_combined = np.vstack([col_bar, img_row])
        Image.fromarray(bev_combined).save(out_dir / "bev_comparison.png")
    except Exception as exc:
        logger.warning("Montage creation failed: %s", exc)


def _save_carla_snapshot(carla_world, ego, out_dir, vel_map=None):
    """Save the CARLA world snapshot (ego transform, velocity, actor list) as JSON.

    When *vel_map* (``{actor_id_str: {"velocity": [vx,vy,vz],
    "angular_velocity": [ax,ay,az]}}``) is provided, its values override whatever
    CARLA's API returns — necessary because actors have ``set_simulate_physics(False)``
    so the physics engine does not update their velocity.
    """
    if vel_map is None:
        vel_map = {}

    def _stored_vel(actor_id):
        v = vel_map.get(str(actor_id))
        if v is None:
            return None
        return {"x": v["velocity"][0], "y": v["velocity"][1], "z": v["velocity"][2]}

    def _stored_angular_vel(actor_id):
        v = vel_map.get(str(actor_id))
        if v is None:
            return None
        return {"x": v["angular_velocity"][0], "y": v["angular_velocity"][1], "z": v["angular_velocity"][2]}

    ws = carla_world.get_snapshot()
    ego_snap = ws.find(ego.id)

    def vec3d(v):
        return {"x": v.x, "y": v.y, "z": v.z}

    def transform(t):
        return {
            "location": vec3d(t.location),
            "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
        }

    def velocity_for(actor_id, snap):
        stored = _stored_vel(actor_id)
        if stored is not None:
            return stored
        return vec3d(snap.get_velocity())

    def angular_velocity_for(actor_id, snap):
        stored = _stored_angular_vel(actor_id)
        if stored is not None:
            return stored
        return vec3d(snap.get_angular_velocity())

    data = {
        "frame": ws.frame,
        "town": carla_world.get_map().name.split("/")[-1].split(".")[0],
        "timestamp": {
            "elapsed_seconds": ws.timestamp.elapsed_seconds,
            "delta_seconds": ws.timestamp.delta_seconds,
            "platform_timestamp": ws.timestamp.platform_timestamp,
        },
        "ego": {
            "id": ego.id,
            "type_id": ego.type_id,
            "transform": transform(ego_snap.get_transform()),
            "velocity": velocity_for(ego.id, ego_snap),
            "angular_velocity": angular_velocity_for(ego.id, ego_snap),
            "acceleration": vec3d(ego_snap.get_acceleration()),
        },
        "weather": {
            "cloudiness": carla_world.get_weather().cloudiness,
            "precipitation": carla_world.get_weather().precipitation,
            "sun_altitude_angle": carla_world.get_weather().sun_altitude_angle,
        },
        "actors": [],
    }
    type_map = {a.id: a.type_id for a in carla_world.get_actors()}
    for actor_id in {a.id for a in carla_world.get_actors()} - {ego.id}:
        snap = ws.find(actor_id)
        if snap is None:
            continue
        type_id = type_map.get(actor_id, "unknown")
        if type_id.startswith("sensor."):
            continue
        data["actors"].append({
            "id": actor_id, "type_id": type_id,
            "transform": transform(snap.get_transform()),
            "velocity": velocity_for(actor_id, snap),
            "angular_velocity": angular_velocity_for(actor_id, snap),
        })
    log_json(data, out_dir / "carla_snapshot.json", "CARLA world snapshot")
    return data


def capture(meta: NuscMeta, target_sample: str, carla_world, weather_params: dict, nusc_root, ego):
    """Set weather, mount 6 RGB cameras + BEV on *ego*, tick, return images + similarity metrics."""
    if ego is None:
        raise RuntimeError("ego actor is None — stage 4 failed to spawn the ego (see actor_spawn_log.json for details)")
    bp_lib = carla_world.get_blueprint_library()
    cwp = carla.WeatherParameters(
        cloudiness=weather_params.get("cloudiness", 5.0),
        precipitation=weather_params.get("precipitation", 0.0),
        precipitation_deposits=weather_params.get("precipitation_deposits", 0.0),
        wind_intensity=weather_params.get("wind_intensity", 10.0),
        sun_azimuth_angle=weather_params.get("sun_azimuth_angle", -1.0),
        sun_altitude_angle=weather_params.get("sun_altitude_angle", 45.0),
        fog_density=weather_params.get("fog_density", 2.0),
        fog_distance=weather_params.get("fog_distance", 0.75),
        wetness=weather_params.get("wetness", 0.0),
    )
    carla_world.set_weather(cwp)

    sample_cams = _build_sample_cameras(meta, target_sample)
    cal_dict = meta.calib_dict()
    ego_dict = meta.ego_dict()
    nusc_ep = ego_dict[sample_cams["CAM_FRONT"]["ego_pose_token"]]
    nusc_ego_pos = np.array(nusc_ep["translation"])
    nusc_ego_rot = nusc_ep["rotation"]
    camera_specs = {}
    for ch in CAMERA_CHANNELS:
        sd = sample_cams[ch]
        cs = cal_dict[sd["calibrated_sensor_token"]]
        camera_specs[ch.replace("CAM_", "")] = _compute_camera_spec(cs, nusc_ego_pos, nusc_ego_rot)

    captured = {}
    sensor_actors = {}
    for name, spec in camera_specs.items():
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(IMAGE_W))
        bp.set_attribute("image_size_y", str(IMAGE_H))
        bp.set_attribute("fov", str(spec["fov"]))
        tf = carla.Transform(carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]), carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=0.0))
        sensor = carla_world.spawn_actor(bp, tf, attach_to=ego)
        sensor.listen(lambda img, n=name: captured.update({n: img}))
        sensor_actors[name] = sensor
    for _ in range(30):
        carla_world.tick()
        time.sleep(0.05)

    carla_images = {}
    for name in camera_specs:
        img = captured.get(name)
        if img is None:
            continue
        array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        carla_images[name] = array[:, :, [2, 1, 0]].copy()

    bev_bp = bp_lib.find("sensor.camera.rgb")
    bev_bp.set_attribute("image_size_x", "1600")
    bev_bp.set_attribute("image_size_y", "1600")
    bev_bp.set_attribute("fov", "90")
    loc = ego.get_location()
    bev_sensor = carla_world.spawn_actor(bev_bp, carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 50), carla.Rotation(pitch=-90, yaw=-90, roll=0)))
    bev_captured = {}
    bev_sensor.listen(lambda img: bev_captured.update({"bev": img}))
    for _ in range(10):
        carla_world.tick()
        time.sleep(0.05)
    bev_sensor.stop()
    bev_sensor.destroy()
    bev_img = bev_captured.get("bev")
    if bev_img is not None:
        array = np.frombuffer(bev_img.raw_data, dtype=np.uint8).reshape((bev_img.height, bev_img.width, 4))
        carla_images["BEV"] = array[:, :, [2, 1, 0]].copy()

    sim_results = _compute_similarity_metrics(meta, target_sample, nusc_root, carla_images)
    return carla_images, sim_results, camera_specs, sensor_actors, ego


def stage5_capture(meta: NuscMeta, target_sample: str, carla_world, weather_params: dict, nusc_root, out_dir, ego, vel_map=None, best_rotation=0):
    """Orchestrate capture: set weather → capture six views + BEV → compute metrics → save montage + snapshot."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Camera Capture & Snapshot")
    logger.info("=" * 60)
    carla_images, sim_results, camera_specs, sensor_actors, _ = capture(meta, target_sample, carla_world, weather_params, nusc_root, ego)

    log_json(camera_specs, out_dir / "camera_calibration.json", "camera intrinsics and extrinsics")
    _create_comparison_montage(meta, target_sample, nusc_root, carla_images, out_dir, best_rotation=best_rotation)
    for s in sensor_actors.values():
        s.stop()
        s.destroy()
    carla_world.tick()
    _save_carla_snapshot(carla_world, ego, out_dir, vel_map=vel_map)
    log_json(sim_results, out_dir / "similarity_metrics.json", "image similarity metrics")
    return camera_specs
