#!/usr/bin/env python3
"""
verify_snapshot.py — Verify CARLA snapshot save/load fidelity.

Loads a CARLA snapshot saved by pipeline.py Stage 5, re-spawns the 6 cameras
using the saved calibration, captures fresh images, and compares them to the
originals via L1 error.

Usage:
  python real2sim/verify_snapshot.py <stage5_output_dir> [--port PORT]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

from real2sim.core import ensure_carla_running

IMAGE_W, IMAGE_H = 1600, 900
MAX_CLEAR_ATTEMPTS = 20

logger = logging.getLogger("real2sim.verify")


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)


def log_json(obj, path, desc="data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Saved %s → %s", desc, path)


def load_snapshot(snapshot_path: Path, world):
    """Load a saved CARLA snapshot JSON and spawn all actors into *world*.

    Returns (ego, actor_count, failed_count).
    """
    with open(snapshot_path) as f:
        data = json.load(f)

    weather = data.get("weather", {})
    world.set_weather(carla.WeatherParameters(
        cloudiness=weather.get("cloudiness", 5.0),
        precipitation=weather.get("precipitation", 0.0),
        sun_altitude_angle=weather.get("sun_altitude_angle", 45.0),
    ))
    logger.info("Weather set from snapshot")

    def clear_world(world):
        """Destroy all actors synchronously with retries.

        CARLA's destroy() is async — actors linger until the next tick.
        This helper ticks repeatedly until all non-spectator actors are gone.
        """
        for attempt in range(MAX_CLEAR_ATTEMPTS):
            for a in world.get_actors():
                if a.is_alive and "spectator" not in a.type_id:
                    a.destroy()
            world.tick()
            remaining = [a for a in world.get_actors() if a.is_alive and "spectator" not in a.type_id]
            if not remaining:
                logger.debug("World cleared after %d tick(s)", attempt + 1)
                return
        raise RuntimeError(
            f"Failed to clear world after {MAX_CLEAR_ATTEMPTS} ticks; "
            f"{len(remaining)} actors still alive: {[a.type_id for a in remaining[:5]]}"
        )
    clear_world(world)

    bp_lib = world.get_blueprint_library()

    def loc(d):
        return carla.Location(x=d["x"], y=d["y"], z=d.get("z", 0.0))

    def rot(d):
        r = d.get("rotation", d)
        return carla.Rotation(pitch=r.get("pitch", 0.0), yaw=r.get("yaw", 0.0), roll=r.get("roll", 0.0))

    def tf_from_dict(d):
        return carla.Transform(loc(d.get("location", d)), rot(d.get("rotation", d)))

    ego_info = data["ego"]
    ego_bp = bp_lib.find(ego_info["type_id"])
    ego_tf = tf_from_dict(ego_info["transform"])
    ego = world.try_spawn_actor(ego_bp, ego_tf)
    if ego is None:
        ego = world.spawn_actor(ego_bp, ego_tf)
    ego.set_simulate_physics(False)
    ego.set_target_velocity(carla.Vector3D(**ego_info["velocity"]))
    ego.set_target_angular_velocity(carla.Vector3D(**ego_info.get("angular_velocity", {"x": 0, "y": 0, "z": 0})))
    logger.info("Ego spawned: %s at (%.1f, %.1f, %.1f)",
                ego_info["type_id"],
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z)

    spawned = 0
    failed = 0
    for actor_data in data.get("actors", []):
        try:
            bp = bp_lib.find(actor_data["type_id"])
        except KeyError:
            logger.debug("  SKIP %s: blueprint not found", actor_data["type_id"])
            continue
        actor_tf = tf_from_dict(actor_data["transform"])
        a = world.try_spawn_actor(bp, actor_tf)
        if a is not None:
            a.set_simulate_physics(False)
            a.set_target_velocity(carla.Vector3D(**actor_data["velocity"]))
            a.set_target_angular_velocity(carla.Vector3D(**actor_data.get("angular_velocity", {"x": 0, "y": 0, "z": 0})))
            spawned += 1
        else:
            failed += 1

    world.tick()
    logger.info("Snapshot loaded: %d actors spawned, %d failed", spawned, failed)
    return ego, spawned, failed


def verify_snapshot(s5_out: Path, s6_out: Path, client, world):
    """Load the Stage 5 snapshot, re-spawn cameras, capture, compare with originals via L1."""
    logger.info("=" * 60)
    logger.info("Verify Snapshot Save/Load")
    logger.info("=" * 60)

    snapshot_path = s5_out / "carla_snapshot.json"
    if not snapshot_path.exists():
        raise RuntimeError(f"Snapshot not found at {snapshot_path}")
    if not (s5_out / "camera_calibration.json").exists():
        raise RuntimeError(f"Camera calibration not found at {s5_out / 'camera_calibration.json'}")
    with open(s5_out / "camera_calibration.json") as f:
        camera_specs = json.load(f)

    ego, _, _ = load_snapshot(snapshot_path, world)

    # Re-spawn cameras from saved calibration specs
    bp_lib = world.get_blueprint_library()
    re_captured = {}
    sensor_actors = {}
    for name, spec in sorted(camera_specs.items()):
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(IMAGE_W))
        bp.set_attribute("image_size_y", str(IMAGE_H))
        bp.set_attribute("fov", str(spec["fov"]))
        tf = carla.Transform(
            carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
            carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=0.0),
        )
        sensor = world.spawn_actor(bp, tf, attach_to=ego)
        sensor.listen(lambda img, n=name: re_captured.update({n: img}))
        sensor_actors[name] = sensor
        logger.info("  %s cam re-spawned: (%.3f, %.3f, %.3f) fov=%.1f",
                    name, spec["x"], spec["y"], spec["z"], spec["fov"])
    logger.info("Re-spawned %d camera sensors", len(sensor_actors))

    for _ in range(30):
        world.tick()
        time.sleep(0.05)

    # Compare with originals via L1
    l1_results = {}
    for name in sorted(camera_specs.keys()):
        orig_path = s5_out / f"{name}_carla.jpg"
        if not orig_path.exists():
            logger.warning("Original image not found: %s", orig_path)
            continue
        new_img = re_captured.get(name)
        if new_img is None:
            logger.warning("No re-captured image for %s", name)
            continue

        orig_arr = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32)
        new_arr = np.frombuffer(new_img.raw_data, dtype=np.uint8)
        new_arr = new_arr.reshape((new_img.height, new_img.width, 4))
        new_rgb = new_arr[:, :, [2, 1, 0]].astype(np.float32)

        h = min(orig_arr.shape[0], new_rgb.shape[0])
        w = min(orig_arr.shape[1], new_rgb.shape[1])
        diff = np.abs(orig_arr[:h, :w] - new_rgb[:h, :w])
        l1 = float(np.mean(diff))
        l1_results[str(name)] = round(l1, 4)
        logger.info("  L1 error for %s: %.4f", name, l1)

        re_path = s6_out / f"{name}_carla_reloaded.jpg"
        Image.fromarray(new_rgb.astype(np.uint8)).save(re_path, quality=95)
        logger.info("Saved re-captured %s", re_path)

    # Destroy cameras after verification
    for s in sensor_actors.values():
        s.stop()
        s.destroy()
    world.tick()
    logger.info("Destroyed %d camera sensors", len(sensor_actors))

    if l1_results:
        l1_results["mean"] = round(
            float(np.mean(list(l1_results.values()))), 4)
        log_json(l1_results, s6_out / "snapshot_verify_l1.json",
                 "snapshot verification L1 errors")
        logger.info("  Mean L1 error: %.4f", l1_results["mean"])


def main():
    parser = argparse.ArgumentParser(
        description="Verify CARLA snapshot save/load fidelity")
    parser.add_argument("input_dir", type=Path,
                        help="Stage 5 output directory (contains snapshot + calibration)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <input_dir>/../stage6_verify)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-hz", type=float, default=20.0,
                        help="Simulation frequency (default: 20)")
    args = parser.parse_args()

    s5_out = args.input_dir.resolve()
    s6_out = args.output_dir.resolve() if args.output_dir else s5_out.parent / "stage6_verify"
    s6_out.mkdir(parents=True, exist_ok=True)

    setup_logging(s6_out / "verify.log")

    ensure_carla_running(args.port, logger)

    client = carla.Client(args.host, args.port)
    client.set_timeout(100.0)
    world = client.get_world()

    # Load the correct town if the snapshot specifies one
    snapshot_path = s5_out / "carla_snapshot.json"
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            snap_data = json.load(f)
        town = snap_data.get("town")
    if town is not None:
        current_map = world.get_map().name.split("/")[-1].split(".")[0]
        town = town.split("/")[-1].split(".")[0]
        if town != current_map:
            logger.info("Loading town '%s' (was '%s')", town, current_map)
            world = client.load_world(town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.sim_hz
    world.apply_settings(settings)
    world.tick()

    try:
        verify_snapshot(s5_out, s6_out, client, world)
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        logger.info("CARLA returned to asynchronous mode")


if __name__ == "__main__":
    main()
