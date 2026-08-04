import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import socket

import numpy as np
from dotenv import load_dotenv
import carla

logger = logging.getLogger("real2sim")

EGO_BLUEPRINT = "vehicle.citroen.c3"


REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_json(obj, path, desc="data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Saved %s -> %s", desc, path)


class NuscMeta:
    """Lazy-loaded nuScenes metadata tables with cross-reference helpers."""

    def __init__(self, meta_dir: Path):
        self._dir = meta_dir
        self._tables = {}

    def _load(self, name):
        if name not in self._tables:
            with open(self._dir / name) as f:
                self._tables[name] = json.load(f)
        return self._tables[name]

    @property
    def samples(self):
        return self._load("sample.json")

    @property
    def sample_data(self):
        return self._load("sample_data.json")

    @property
    def calibrated_sensors(self):
        return self._load("calibrated_sensor.json")

    @property
    def sensors(self):
        return self._load("sensor.json")

    @property
    def ego_poses(self):
        return self._load("ego_pose.json")

    @property
    def annotations(self):
        return self._load("sample_annotation.json")

    @property
    def categories(self):
        return self._load("category.json")

    @property
    def instances(self):
        return self._load("instance.json")

    @property
    def scenes(self):
        return self._load("scene.json")

    @property
    def logs(self):
        return self._load("log.json")

    def sensor_channel(self, calib_token):
        cs = next(c for c in self.calibrated_sensors if c["token"] == calib_token)
        s = next(s for s in self.sensors if s["token"] == cs["sensor_token"])
        return s["channel"]

    def calib_dict(self):
        return {c["token"]: c for c in self.calibrated_sensors}

    def ego_dict(self):
        return {e["token"]: e for e in self.ego_poses}

    def cat_dict(self):
        return {c["token"]: c["name"] for c in self.categories}

    def inst_dict(self):
        return {i["token"]: i for i in self.instances}


EGO_REAR_AXLE_TO_CENTER = 1.386   # metres from rear axle to vehicle centre (Citroen C3)
EGO_CENTER_Z = 0.75               # approx half the Citroen C3 height, geometric centre above ground


def nusc_ego_centre(rear_axle_pos, rotation):
    """Return vehicle centre (rear axle + 1.386 m forward + 0.75 m up) in nuScenes global coordinates."""
    q = Quaternion(rotation) if not isinstance(rotation, Quaternion) else rotation
    offset = q.rotate([EGO_REAR_AXLE_TO_CENTER, 0.0, 0.0])
    return (rear_axle_pos[0] + offset[0],
            rear_axle_pos[1] + offset[1],
            rear_axle_pos[2] + EGO_CENTER_Z)


def nusc_quaternion_yaw(q):
    """Extract ego yaw (radians, CCW from global +X) from a nuScenes quaternion [w,x,y,z] or pyquaternion Quaternion.

    nuScenes vehicle frame: +X = forward, +Y = left, +Z = up.
    The ego-pose quaternion rotates from vehicle frame to global frame
    (global: +X = East, +Y = North, +Z = up).
    """
    if not isinstance(q, Quaternion):
        q = Quaternion(q)
    v = q.rotate([1, 0, 0])
    return math.atan2(v[1], v[0])


def get_cam_front(meta: NuscMeta, target_sample: str):
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


def nusc_velocity_to_carla(vx, vy, vz, best_rotation_D_deg):
    """Transform nuScenes linear velocity (global frame, RHS) to CARLA (global frame, LHS).

    Step 1 — rotate by D CCW (same as position transform).
    Step 2 — flip Y for LHS (same as position transform).
    Step 3 — translation has no effect on velocity.
    """
    D_rad = math.radians(best_rotation_D_deg)
    cosD, sinD = math.cos(D_rad), math.sin(D_rad)
    vx1 = vx * cosD - vy * sinD
    vy1 = vx * sinD + vy * cosD
    vz1 = vz
    return (vx1, -vy1, vz1)


def nusc_yaw_rate_to_carla(yaw_rate_rad_s):
    """Transform nuScenes yaw rate (CCW+, rad/s) to CARLA (CW+, deg/s).

    Translation and coordinate-frame rotation do NOT affect global angular velocity.
    Only the handedness change (RHS→LHS) negates the sign, and CARLA uses deg/s.
    """
    return -math.degrees(yaw_rate_rad_s)


def check_velocity_invariants(nusc_vel_list, carla_vel_list, nusc_yaw_list, carla_yaw_list, eps=1e-4):
    """Sanity: velocity norms and pairwise relative directions are preserved under transform.

    *nusc_vel_list* and *carla_vel_list* each contain (vx, vy, vz) tuples.
    *nusc_yaw_list* and *carla_yaw_list* contain yaw rates (rad/s and deg/s).
    """
    ok = True
    n = len(nusc_vel_list)
    for i in range(n):
        n_i = np.array(nusc_vel_list[i][:3], dtype=float)
        c_i = np.array(carla_vel_list[i][:3], dtype=float)
        nn_i = float(np.linalg.norm(n_i))
        cn_i = float(np.linalg.norm(c_i))
        if abs(nn_i - cn_i) > eps:
            logger.warning("Velocity sanity: norm changed (idx %d): %.6f → %.6f", i, nn_i, cn_i)
            ok = False
        for j in range(i + 1, n):
            n_j = np.array(nusc_vel_list[j][:3], dtype=float)
            c_j = np.array(carla_vel_list[j][:3], dtype=float)
            nn_j = float(np.linalg.norm(n_j))
            cn_j = float(np.linalg.norm(c_j))
            if nn_i > eps and nn_j > eps and cn_i > eps and cn_j > eps:
                cos_before = float(np.dot(n_i, n_j) / (nn_i * nn_j))
                cos_after = float(np.dot(c_i, c_j) / (cn_i * cn_j))
                if abs(cos_before - cos_after) > eps:
                    logger.warning(
                        "Velocity sanity: relative direction changed (pair %d,%d): cos=%.6f → %.6f",
                        i, j, cos_before, cos_after,
                    )
                    ok = False
    if n > 0:
        yaw_ok = all(
            abs(carla_yaw_list[i] - nusc_yaw_rate_to_carla(nusc_yaw_list[i])) < max(eps, 1e-3)
            for i in range(n)
        )
        if not yaw_ok:
            logger.warning("Velocity sanity: yaw rate mismatch in some actors")
            ok = False
    if ok:
        logger.info("Velocity sanity: %d actors preserve norms, relative directions, and yaw rates", n)
    return ok


def get_robust_world(client, target_town: str, clean_residuals: bool = True) -> object:
    """
    Robustly get or load a CARLA world, avoiding redundant load_world calls,
    and cleaning residual actors when reusing an already-loaded map.

    Args:
        client: Connected carla.Client.
        target_town: Town name e.g. "Town03".
        clean_residuals: If True and the map is already loaded, destroy
                         leftover vehicles / walkers / sensors.

    Returns:
        carla.World
    """
    import carla

    world = client.get_world()
    full_map_name = world.get_map().name
    current_map_clean = full_map_name.split("/")[-1]
    # Strip trailing _Opt suffix so Town03_Opt matches target "Town03"
    if current_map_clean.lower().endswith("_opt"):
        current_map_clean = current_map_clean[:-4]

    logger.debug("Raw map string from CARLA: '%s' -> Cleaned: '%s'", full_map_name, current_map_clean)

    if target_town.lower() != current_map_clean.lower():
        logger.info("Map mismatch. Loading town: %s (current was %s)", target_town, current_map_clean)
        world = client.load_world(target_town)
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick()
        logger.info("Successfully loaded new town: %s", target_town)
    else:
        logger.info("Town '%s' is already loaded. Skipping load_world.", target_town)
        if clean_residuals:
            logger.info("Cleaning up residual actors from previous runs ...")
            actor_list = world.get_actors()
            residuals = (
                list(actor_list.filter("vehicle.*")) +
                list(actor_list.filter("walker.*")) +
                list(actor_list.filter("sensor.*"))
            )
            destroyed = sum(1 for a in residuals if a.is_alive and a.destroy())
            if destroyed:
                logger.info("Removed %d residual actors.", destroyed)
                if world.get_settings().synchronous_mode:
                    world.tick()

    return world


def compute_carla_trajectory(meta, target_sample, final_spawn, weather_params, town, num_frames, ego_blueprint=EGO_BLUEPRINT):
    """Walk the nusc sample chain and compute CARLA ego transforms via the 3-step transform.

    Used by ``real2sim.pipeline`` and ``real2sb.offline_snapshots``.
    """
    best_rotation_D = final_spawn.get("best_rotation", 0)
    carla_ego_loc = (
        final_spawn["location"]["x"],
        final_spawn["location"]["y"],
        final_spawn["location"]["z"],
    )

    D_rad = math.radians(best_rotation_D)
    cosD, sinD = math.cos(D_rad), math.sin(D_rad)

    def _step1_rotate(x, y, z, yaw_rad):
        return (x * cosD - y * sinD, x * sinD + y * cosD, z), yaw_rad + D_rad

    def _step2_to_carla(x, y, z, yaw_rad):
        return (x, -y, z), -yaw_rad

    def _process_nusc_pose(translation, rotation):
        x, y, z = translation
        q = Quaternion(rotation)
        yaw = nusc_quaternion_yaw(q)
        (x, y, z), yaw = _step1_rotate(x, y, z, yaw)
        (x, y, z), yaw = _step2_to_carla(x, y, z, yaw)
        return (x, y, z), yaw

    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    sample_by_token = {s["token"]: s for s in meta.samples}
    ego_dict = meta.ego_dict()

    sample_cam_map = {}
    for sd in meta.sample_data:
        if not sd.get("is_key_frame"):
            continue
        ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
        if ch == "CAM_FRONT":
            sample_cam_map[sd["sample_token"]] = sd

    frames = []
    sample_tok = target_sample
    prev_carla_loc = None
    prev_carla_yaw = None
    prev_ts = None
    Tx = Ty = Tz = 0.0

    for fi in range(num_frames):
        if sample_tok is None:
            if fi == 0:
                logger.warning("No samples to walk for trajectory")
            break

        cam_front = sample_cam_map.get(sample_tok)
        if cam_front is None:
            if fi == 0:
                logger.warning("Sample %s has no CAM_FRONT", sample_tok)
            break

        nusc_ep = ego_dict[cam_front["ego_pose_token"]]
        nusc_pos = np.array(nusc_ep["translation"])
        q_ego = Quaternion(nusc_ep["rotation"])
        nusc_center = nusc_ego_centre(nusc_pos, q_ego)

        ego_carla_pos, ego_carla_yaw = _process_nusc_pose(
            list(nusc_center), nusc_ep["rotation"])

        if fi == 0:
            Tx = carla_ego_loc[0] - ego_carla_pos[0]
            Ty = carla_ego_loc[1] - ego_carla_pos[1]
            Tz = carla_ego_loc[2] - ego_carla_pos[2]

        ego_carla = {
            "location": (ego_carla_pos[0] + Tx, ego_carla_pos[1] + Ty, ego_carla_pos[2] + Tz),
            "yaw_deg": ((math.degrees(ego_carla_yaw) + 180) % 360) - 180,
        }

        ts = sample_by_token[sample_tok]["timestamp"]

        if prev_carla_loc is not None and prev_ts is not None:
            dt = abs(ts - prev_ts) / 1e6
            if dt > 0:
                vx = (ego_carla["location"][0] - prev_carla_loc[0]) / dt
                vy = (ego_carla["location"][1] - prev_carla_loc[1]) / dt
                vz = (ego_carla["location"][2] - prev_carla_loc[2]) / dt
                dyaw = (ego_carla["yaw_deg"] - prev_carla_yaw + 180) % 360 - 180
                wz = dyaw / dt
            else:
                vx, vy, vz, wz = 0.0, 0.0, 0.0, 0.0
        else:
            vx, vy, vz, wz = 0.0, 0.0, 0.0, 0.0

        frame = {
            "sample_token": sample_tok,
            "ego": {
                "type_id": ego_blueprint,
                "transform": {
                    "location": {"x": ego_carla["location"][0], "y": ego_carla["location"][1], "z": ego_carla["location"][2]},
                    "rotation": {"pitch": 0.0, "yaw": ego_carla["yaw_deg"], "roll": 0.0},
                },
                "velocity": {"x": vx, "y": vy, "z": vz},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": wz},
            },
        }
        frames.append(frame)

        prev_carla_loc = ego_carla["location"]
        prev_carla_yaw = ego_carla["yaw_deg"]
        prev_ts = ts

        s = sample_by_token.get(sample_tok)
        sample_tok = s["next"] if s else None

    delta_seconds = 0.5
    if len(frames) >= 2:
        s0 = sample_by_token[frames[0]["sample_token"]]
        s1 = sample_by_token[frames[1]["sample_token"]]
        delta_seconds = abs(s1["timestamp"] - s0["timestamp"]) / 1e6

    return {
        "town": town,
        "weather": {
            "cloudiness": weather_params.get("cloudiness"),
            "precipitation": weather_params.get("precipitation"),
            "sun_altitude_angle": weather_params.get("sun_altitude_angle"),
        },
        "delta_seconds": delta_seconds,
        "frames": frames,
    }


def filter_barely_visible_annotations(meta, target_sample, min_pixel_area=50):
    """Return set of annotation tokens whose max projected pixel area across all 6 cameras is below *min_pixel_area*."""
    from real2sim.stage5_capture import (
        CAMERA_CHANNELS, _build_sample_cameras, get_box_corners,
        get_lidar_calibration, inverse_transform, project_to_image, quat_mult,
    )
    from real2sim.stage4_spawn import IMAGE_H, IMAGE_W

    sample_cams = _build_sample_cameras(meta, target_sample)
    cal_dict = meta.calib_dict()
    ego_dict = meta.ego_dict()
    lidar_cal = get_lidar_calibration(meta)
    Q_LIDAR = lidar_cal["rotation"]
    sample_anns = [a for a in meta.annotations if a["sample_token"] == target_sample]

    barely_visible = set()
    for ann in sample_anns:
        max_area = 0
        for ch in CAMERA_CHANNELS:
            sd = sample_cams.get(ch)
            if sd is None:
                continue
            cs = cal_dict[sd["calibrated_sensor_token"]]
            ep = ego_dict[sd["ego_pose_token"]]

            rotation = quat_mult(Q_LIDAR, ann["rotation"])
            corners_global = get_box_corners(ann["translation"], ann["size"], rotation)
            corners_ego = inverse_transform(corners_global, ep["translation"], ep["rotation"])
            corners_cam = inverse_transform(corners_ego, cs["translation"], cs["rotation"])

            if np.all(corners_cam[:, 2] <= 0):
                continue

            pts_2d = project_to_image(corners_cam, cs["camera_intrinsic"])
            x1, y1 = pts_2d.min(axis=0)
            x2, y2 = pts_2d.max(axis=0)

            vis_w = max(0, min(x2, IMAGE_W) - max(x1, 0))
            vis_h = max(0, min(y2, IMAGE_H) - max(y1, 0))
            area = vis_w * vis_h
            if area > max_area:
                max_area = area

        if max_area < min_pixel_area:
            barely_visible.add(ann["token"])

    if barely_visible:
        logger.info("filter_barely_visible_annotations: %d/%d annotations below %d-pixel threshold, excluded",
                     len(barely_visible), len(sample_anns), min_pixel_area)
    return barely_visible


def ensure_carla_running(port: int, logger: logging.Logger):
    """Ensure CARLA is running on *port* (detached, persists across runs).

    Tries a lightweight connect first; if nothing is listening, kills anything
    on the port and starts a detached CARLA server.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10):
            logger.info("CARLA already running on port %d", port)
            return
    except (ConnectionRefusedError, OSError, socket.timeout):
        logger.info("No CARLA on port %d — starting one", port)

    load_dotenv(REPO_ROOT / ".env")
    carla_root = os.environ.get("CARLA_ROOT", "")
    if not carla_root:
        raise RuntimeError(
            "CARLA_ROOT not set. Add CARLA_ROOT=/path/to/CARLA_0.9.13 to a .env file "
            f"at {REPO_ROOT / '.env'} or export it in your shell."
        )
    carla_sh = Path(carla_root) / "CarlaUE4.sh"
    if not carla_sh.exists():
        raise RuntimeError(
            f"CarlaUE4.sh not found at {carla_sh}. Check CARLA_ROOT."
        )

    # Kill anything on this port
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            logger.info("Killed existing process on port %d", port)
            time.sleep(2)
        elif result.returncode != 1:
            logger.warning("fuser returned %d: %s", result.returncode, result.stderr.decode().strip())
    except FileNotFoundError:
        logger.warning("fuser not found — cannot kill existing process on port %d", port)
    except Exception as exc:
        logger.warning("Failed to kill existing process on port %d: %s", port, exc)

    # Start completely detached (survives Python exit)
    logger.info("Starting CARLA on port %d (detached): %s -RenderOffScreen -carla-port=%d",
                port, carla_sh, port)
    subprocess.Popen(
        [str(carla_sh), "-RenderOffScreen", f"-carla-port={port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Poll until CARLA is ready (up to 120 s)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                logger.info("CARLA ready on port %d", port)
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(3)
    raise RuntimeError("CARLA did not become ready on port %d within 120 s", port)
