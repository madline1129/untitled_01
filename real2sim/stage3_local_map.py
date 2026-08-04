"""Stage 3: rotation-invariant local-map matching.

Extracts the nuScenes local map around the ego, pre-computes CCW rotations,
then searches CARLA spawn points for the best visual match via LPIPS.
Returns the top-ranked spawn point and the rotation angle D that aligns
the two coordinate frames for downstream 3-step scene transform.
"""
import logging
import math
import time

import cv2
import lpips
import numpy as np
import torch
import carla
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from scipy.ndimage import rotate as ndi_rotate
from scipy.spatial import KDTree

from real2sim.core import EGO_REAR_AXLE_TO_CENTER, NuscMeta, log_json, nusc_ego_centre, nusc_quaternion_yaw


logger = logging.getLogger("real2sim")

PATCH_SIZE = (60, 60)
CANVAS_SIZE = (120, 120)
LAYER_NAMES = ["drivable_area", "road_divider", "lane_divider"]
NUM_CANDIDATES = 1000
TOP_K = 1

_lpips_model = None
_rotated_lpips_cache = {}


def angle_diff(a, b):
    """Absolute angular difference |a−b| normalised to [0, 180] degrees."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def _get_cam_front(meta, target_sample):
    """Return the CAM_FRONT sample_data record for a given sample token."""
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    for sd in meta.sample_data:
        if sd["sample_token"] != target_sample or not sd.get("is_key_frame"):
            continue
        ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
        if ch == "CAM_FRONT":
            return sd
    raise RuntimeError("CAM_FRONT not found for sample")


def extract_nusc_local_map(meta: NuscMeta, target_sample: str, nusc_root, out_dir):
    """Extract and save an axis-aligned nuScenes local map patch centred on the ego vehicle (no yaw rotation)."""
    cam_front = _get_cam_front(meta, target_sample)
    ep = meta.ego_dict()[cam_front["ego_pose_token"]]
    s = next(s for s in meta.samples if s["token"] == target_sample)
    scene = next(sc for sc in meta.scenes if sc["token"] == s["scene_token"])
    log = next(l for l in meta.logs if l["token"] == scene["log_token"])
    location = log["location"]

    ego_x, ego_y, ego_z = ep["translation"]
    q = Quaternion(ep["rotation"])
    ego_yaw = nusc_quaternion_yaw(q)
    ego_yaw_deg = math.degrees(ego_yaw)
    fwd = (math.cos(ego_yaw), math.sin(ego_yaw))

    cx, cy, _ = nusc_ego_centre((ego_x, ego_y, ego_z), q)

    logger.info("nuScenes ego: rear_axle=(%.2f, %.2f) center=(%.2f, %.2f) yaw=%.1fdeg fwd=(%.4f, %.4f) location=%s",
                ego_x, ego_y, cx, cy, ego_yaw_deg, fwd[0], fwd[1], location)

    nusc_map = NuScenesMap(dataroot=str(nusc_root), map_name=location)
    M = nusc_map.get_map_mask([cx, cy, PATCH_SIZE[0], PATCH_SIZE[1]], 0, LAYER_NAMES, CANVAS_SIZE)
    M = np.flip(M, axis=1)  # image Y-down → global Y-up

    log_json({
        "ego_center_x": cx,
        "ego_center_y": cy,
        "ego_rear_x": ego_x,
        "ego_rear_y": ego_y,
        "ego_yaw_deg": ego_yaw_deg,
        "center_offset_m": EGO_REAR_AXLE_TO_CENTER,
        "location": location,
        "mask_shape": list(M.shape),
        "nonzeros": [int(np.count_nonzero(M[i])) for i in range(len(LAYER_NAMES))],
    }, out_dir / "nusc_local_map.json", "nusc local map metadata")
    return M, (ego_x, ego_y, ego_yaw_deg), location, (fwd[0], fwd[1])


def render_carla_local_map(waypoint, waypoints_xy, waypoint_objs, kdtree):
    """Render an axis-aligned local map patch from CARLA waypoints centred at *waypoint*."""
    if hasattr(waypoint, "road_id"):
        cx, cy = waypoint.transform.location.x, waypoint.transform.location.y
    else:
        cx, cy = waypoint.location.x, waypoint.location.y
    canvas = np.zeros((len(LAYER_NAMES), CANVAS_SIZE[0], CANVAS_SIZE[1]), dtype=np.uint8)
    m_per_px_h = PATCH_SIZE[0] / CANVAS_SIZE[0]
    m_per_px_w = PATCH_SIZE[1] / CANVAS_SIZE[1]
    radius = max(PATCH_SIZE) / 2 * 1.1

    for idx in kdtree.query_ball_point([cx, cy], radius):
        wp = waypoint_objs[idx]
        lx, ly = waypoints_xy[idx]
        dx, dy = lx - cx, ly - cy
        rx, ry = dx, dy
        px = int(CANVAS_SIZE[1] / 2 + rx / m_per_px_w)
        py = int(CANVAS_SIZE[0] / 2 + ry / m_per_px_h)
        if not (0 <= px < CANVAS_SIZE[1] and 0 <= py < CANVAS_SIZE[0]):
            continue
        r = max(1, int(2.0 / min(m_per_px_h, m_per_px_w)))
        cv2.circle(canvas[0], (px, py), r, 1, -1)
        lt = wp.left_lane_marking.type if wp.left_lane_marking else None
        rt = wp.right_lane_marking.type if wp.right_lane_marking else None
        if lt == carla.LaneMarkingType.SolidSolid or rt == carla.LaneMarkingType.SolidSolid:
            cv2.circle(canvas[1], (px, py), 1, 1, -1)
        if lt == carla.LaneMarkingType.Broken:
            cv2.circle(canvas[2], (px, py), 1, 1, -1)
        if rt == carla.LaneMarkingType.Broken:
            cv2.circle(canvas[2], (px, py), 1, 1, -1)
    return canvas


def _precompute_rotated_maps(M, step=1):
    """Pre-compute CCW rotations of a map mask in 1-degree steps."""
    n_layers = M.shape[0]
    rotated = {}
    for angle in range(0, 360, step):
        layer_rot = np.array([
            ndi_rotate(M[c], angle, order=0, reshape=False, mode="constant", cval=0)
            for c in range(n_layers)
        ])
        rotated[angle] = layer_rot
    return rotated


def _map_mask_to_bgr(mask):
    """Convert a 3-layer boolean mask to an RGB image.
    
    Layer 0 — drivable_area: filled waypoint circles → BGR(0,0,200) = red
    Layer 1 — road_divider:  SolidSolid marking dot  → BGR(0,255,0)  = green
    Layer 2 — lane_divider:  Broken marking dot      → BGR(255,0,0)  = blue
    """
    h, w = mask.shape[1:]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask[0] > 0] = (0, 0, 200)
    out[mask[1] > 0] = (0, 255, 0)
    out[mask[2] > 0] = (255, 0, 0)
    return out


def _build_rotated_lpips_tensor(rotated_Ms, rotation_angles, device):
    """Build a batched RGB tensor from pre-computed rotated maps for LPIPS."""
    cache_key = (id(rotated_Ms), tuple(rotation_angles), str(device))
    cached = _rotated_lpips_cache.get(cache_key)
    if cached is not None:
        return cached

    rotated_rgb = [_map_mask_to_bgr(rotated_Ms[angle]) for angle in rotation_angles]
    tensor = torch.from_numpy(np.stack(rotated_rgb, axis=0)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    tensor = tensor.to(device)
    _rotated_lpips_cache[cache_key] = tensor
    return tensor


def _select_best_angles_from_similarity(similarity_matrix, rotation_angles):
    """Argmax over LPIPS similarity scores per candidate → best score + angle."""
    best_scores_t, best_idx_t = torch.max(similarity_matrix, dim=1)
    best_scores = [float(v) for v in best_scores_t.detach().cpu().tolist()]
    best_idx = best_idx_t.detach().cpu().tolist()
    best_angles = [rotation_angles[i] for i in best_idx]
    return best_scores, best_angles


def _batch_best_lpips_scores(m_primes, rotated_Ms, rotation_angles, lpips_candidate_batch=2):
    """Batch LPIPS: compare each candidate map against all 360 rotated references, return best per candidate."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_rotations = _build_rotated_lpips_tensor(rotated_Ms, rotation_angles, device)
    num_rot = ref_rotations.shape[0]
    all_scores = []
    all_angles = []

    for start in range(0, len(m_primes), lpips_candidate_batch):
        chunk = m_primes[start:start + lpips_candidate_batch]
        q_rgb = [_map_mask_to_bgr(m) for m in chunk]
        q_tensor = torch.from_numpy(np.stack(q_rgb, axis=0)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        q_tensor = q_tensor.to(device)

        bsz = q_tensor.shape[0]
        ref = ref_rotations.unsqueeze(0).expand(bsz, num_rot, -1, -1, -1).reshape(bsz * num_rot, 3, q_tensor.shape[2], q_tensor.shape[3])
        qry = q_tensor.unsqueeze(1).expand(bsz, num_rot, -1, -1, -1).reshape(bsz * num_rot, 3, q_tensor.shape[2], q_tensor.shape[3])

        with torch.no_grad():
            dist = _lpips_model(ref, qry, normalize=True).view(bsz, num_rot)
        sim = 1.0 - dist
        best_scores, best_angles = _select_best_angles_from_similarity(sim, rotation_angles)
        all_scores.extend(best_scores)
        all_angles.extend(best_angles)

    return all_scores, all_angles


def _save_match_strip(M_nusc, rotated_Ms, best_angle, M_carla, path, label, fwd_nusc=(1.0, 0.0),
                      col_labels=None):
    """Save a three-column comparison: nusc(axis-aligned) | rotated | carla(axis-aligned).

    Args:
        col_labels: 3-element list of column header labels.  Defaults to
                    ``["nusc (axis)", "nusc (rot=N°)", "carla (axis)"]``
                    (real2sim convention).  real2sb overrides with
                    ``["nusc (axis)", "carla (rot=N°)", "carla (axis)"]``.
    """
    if col_labels is None:
        col_labels = ["nusc (axis)", f"nusc (rot={best_angle})", "carla (axis)"]

    DISP_H = 200
    nusc_raw = _map_mask_to_bgr(M_nusc)
    mid = _map_mask_to_bgr(rotated_Ms[best_angle])
    carla_disp = _map_mask_to_bgr(M_carla)
    nusc_raw = cv2.resize(nusc_raw, (DISP_H, DISP_H), interpolation=cv2.INTER_NEAREST)
    mid = cv2.resize(mid, (DISP_H, DISP_H), interpolation=cv2.INTER_NEAREST)
    carla_disp = cv2.resize(carla_disp, (DISP_H, DISP_H), interpolation=cv2.INTER_NEAREST)

    cx = cy = DISP_H // 2
    line_len = 30
    fx, fy = fwd_nusc
    ex = int(cx + line_len * fx)
    ey = int(cy - line_len * fy)
    cv2.line(nusc_raw, (cx, cy), (ex, ey), (255, 128, 0), 2)

    D_rad = math.radians(best_angle)
    fx_rot = fx * math.cos(D_rad) - fy * math.sin(D_rad)
    fy_rot = fx * math.sin(D_rad) + fy * math.cos(D_rad)
    ex = int(cx + line_len * fx_rot)
    ey = int(cy - line_len * fy_rot)
    cv2.line(mid, (cx, cy), (ex, ey), (255, 128, 0), 2)

    cv2.circle(carla_disp, (cx, cy), 4, (255, 191, 0), -1)

    gap = np.full((DISP_H, 8, 3), 40, dtype=np.uint8)
    strip = np.hstack([nusc_raw, gap, mid, gap, carla_disp])
    bar = np.zeros((40, strip.shape[1], 3), dtype=np.uint8)
    gap_px = 8
    for i, text in enumerate(col_labels):
        x = 6 + i * (DISP_H + gap_px)
        cv2.putText(bar, text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(bar, label, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.vstack([bar, strip]))


def _sanity_check_spawn(top, world):
    # BUG: actor.destroy is async. we spawn here to test, but block the later actual spawn in stage 4.
    return

    """Try spawning ego at each top-k candidate; raise RuntimeError if any position is blocked."""
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.citroen.c3")
    for rank, (score, best_angle, tf) in enumerate(top, 1):
        ego_tf = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.05),
            tf.rotation,
        )
        actor = world.try_spawn_actor(ego_bp, ego_tf)
        if actor is None:
            raise RuntimeError(
                f"Top-{rank} spawn point at ({tf.location.x:.1f}, {tf.location.y:.1f}, {tf.location.z:.1f}) "
                f"score={score:.4f} rot={best_angle}deg — failed to spawn ego (collision)"
            )
        actor.destroy()
        logger.info("  spawn sanity OK: top-%d (%.1f, %.1f, %.1f, yaw=%.1f) score=%.4f", 
                    rank, tf.location.x, tf.location.y, tf.location.z + 0.05, tf.rotation.yaw, score)


def match_carla_town(rotated_Ms, M_nusc, world, out_dir, top_k=TOP_K, fwd_nusc=(1.0, 0.0)):
    """Score CARLA spawn points against rotated nuScenes maps, return top-k ranked candidates with best_rotation D."""
    # TODO: use waypoint insteal of spawnponts for better coverage.
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(1.0)
    waypoints_xy = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in waypoints])
    kdtree = KDTree(waypoints_xy)
    rotation_angles = sorted(rotated_Ms.keys())

    spawn_points = carla_map.get_spawn_points()
    step = max(1, len(spawn_points) // NUM_CANDIDATES)
    # TODO: maybe spawn_points is less than NUM_CANDIDATES??
    # TODO: maybe waypoint is more dense and can yield better matches than spawn_points? but more expensive to render all waypoints.
    candidates = spawn_points[::step] if step > 1 else spawn_points
    logger.info("Total spawn points: %d, using step=%d → %d candidates", len(spawn_points), step, len(candidates))

    cand_maps = [render_carla_local_map(tf, waypoints_xy, waypoints, kdtree) for tf in candidates]
    cand_to_map = dict(zip(candidates, cand_maps))

    results = []
    t0 = time.time()
    
    chunk_size = 8
    for start in range(0, len(candidates), chunk_size):
        cand_chunk = candidates[start:start + chunk_size]
        m_primes = [cand_to_map[tf] for tf in cand_chunk]
        best_scores, best_angles = _batch_best_lpips_scores(m_primes, rotated_Ms, rotation_angles)
        for tf, best_score, best_angle in zip(cand_chunk, best_scores, best_angles):
            results.append((best_score, best_angle, tf))
        done = start + len(cand_chunk)
        if done % 200 == 0 or done == len(candidates):
            elapsed = time.time() - t0
            best = max(results, key=lambda r: r[0])
            logger.info("    [%d/%d] best=%.4f (rot=%ddeg) (%.1fs)", done, len(candidates), best[0], best[1], elapsed)

    results.sort(key=lambda x: -x[0])
    top = results[:top_k]
    _sanity_check_spawn(top, world)

    top_dir = out_dir / "top_maps"
    top_dir.mkdir(parents=True, exist_ok=True)
    
    for rank, (score, best_angle, tf) in enumerate(top, 1):
        M_carla = cand_to_map[tf]
        _save_match_strip(M_nusc, rotated_Ms, best_angle, M_carla, top_dir / f"carla_top_{rank:02d}_strip.png", f"#{rank} score={score:.4f}", fwd_nusc=fwd_nusc)

    # refined = []
    # for idx, (seed_score, best_angle, seed_tf) in enumerate(top, start=1):
    #     target_yaw = 90 - (nusc_yaw_deg)
    #     target_yaw = ((target_yaw + 180) % 360) - 180
    #     bx, by = seed_tf.location.x, seed_tf.location.y
    #     sp_indices = kdtree.query_ball_point([bx, by], r=refine_radius)
    #     logger.info(
    #         "  refine seed #%d: target_yaw=%.1f neighbors=%d at (%.2f, %.2f)",
    #         idx, target_yaw, len(sp_indices), bx, by,
    #     )
    #     best_tf = seed_tf
    #     best_yaw_diff = angle_diff(best_tf.rotation.yaw, target_yaw)
    #     best_dist = 0.0
    #     for i in sp_indices:
    #         wx, wy = waypoints_xy[i]
    #         if abs(wx - bx) <= 0.01 and abs(wy - by) <= 0.01:
    #             continue
    #         wp_n = waypoints[i]
    #         yaw_diff = angle_diff(wp_n.transform.rotation.yaw, target_yaw)
    #         dist = math.hypot(wx - bx, wy - by)
    #         improved = yaw_diff < best_yaw_diff - 1e-6
    #         tied = abs(yaw_diff - best_yaw_diff) <= 1e-6
    #         if improved or (tied and dist < best_dist):
    #             if improved:
    #                 logger.info(
    #                     "    refine seed #%d: yaw_diff %.1f -> %.1f at (%.2f, %.2f) yaw=%.1f dist=%.2f",
    #                     idx, best_yaw_diff, yaw_diff, wx, wy, wp_n.transform.rotation.yaw, dist,
    #                 )
    #             best_tf = wp_n.transform
    #             best_yaw_diff = yaw_diff
    #             best_dist = dist

    #     logger.info(
    #         "  refine seed #%d done: yaw_diff=%.1f dist=%.2f",
    #         idx, best_yaw_diff, best_dist,
    #     )
    #     refined.append((seed_score, best_angle, best_tf, seed_tf))

    #     top_refined = refined[:top_k]

    #     for rank, (score, best_angle, tf, seed_tf) in enumerate(top_refined, 1):
    #         M_init = cand_to_map[seed_tf]
    #         M_ref = cand_to_map.get(tf) or render_carla_local_map(tf, waypoints_xy, waypoints, kdtree)
    #         _save_map_mask_image(M_ref, top_dir / f"carla_top_{rank:02d}_refined.png", f"refined{rank}")
    #         _save_match_strip(rotated_Ms, best_angle, M_init, M_ref, top_dir / f"carla_top_{rank:02d}_strip.png", f"#{rank}")

    #     top = [(s, a, tf) for s, a, tf, _ in top_refined]

    matches = []
    for rank, (score, best_angle, tf) in enumerate(top, 1):
        matches.append({
            "rank": rank,
            "score": round(float(score), 6),
            "best_rotation": best_angle,
            "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
            "yaw": tf.rotation.yaw,
        })
    log_json(matches, out_dir / f"map_matches.json", f"map matches")
    return matches


def destroy_all_actors(carla_client, carla_world):
    """Destroy every actor in the CARLA world (used between stages)."""
    actors = [a for a in carla_world.get_actors()]
    if not actors:
        return
    carla_client.apply_batch([carla.command.DestroyActor(a.id) for a in actors])
    carla_world.tick()


def stage3_local_map(
    meta: NuscMeta,
    target_sample: str,
    nusc_root,
    carla_world,
    out_dir,
    top_k: int = TOP_K,
):
    """Run rotation-invariant map matching and return the top spawn-candidate dict + rotated maps."""
    M, ego_info, _, fwd_nusc = extract_nusc_local_map(meta, target_sample, nusc_root, out_dir)
    rotated_Ms = _precompute_rotated_maps(M, step=1)
    log_json({"num_rotations": len(rotated_Ms), "step_deg": 1}, out_dir / "rotation_enum.json", "rotation enumeration")

    candidates = match_carla_town(rotated_Ms, M, carla_world, out_dir, top_k=top_k, fwd_nusc=fwd_nusc)

    log_json(candidates, out_dir / "geo_candidates.json", "geo-ranked candidates")

    final_spawn = candidates[0] if candidates else None
    log_json(final_spawn, out_dir / "spawn_point.json", "selected spawn point")
    return final_spawn, rotated_Ms
