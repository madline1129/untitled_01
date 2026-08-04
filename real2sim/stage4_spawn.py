"""Stage 4: 3-step scene transform + batch spawning of all actors (annotations + ego).

Transforms every nuScenes pose (ego + annotation boxes) into CARLA world
coordinates via the 3-step procedure (rotate by D, flip Y/negate yaw,
translate to spawn point), then spawns all actors in a single CARLA batch.
"""
import json
import logging
import math
import random
from pathlib import Path
import carla
import numpy as np
from pyquaternion import Quaternion

from real2sim.core import (
    EGO_BLUEPRINT,
    EGO_REAR_AXLE_TO_CENTER,
    NuscMeta,
    get_cam_front,
    log_json,
    nusc_ego_centre,
    nusc_quaternion_yaw,
    nusc_velocity_to_carla,
    nusc_yaw_rate_to_carla,
    check_velocity_invariants,
)


logger = logging.getLogger("real2sim")


def nusc_scene_to_carla(nusc_ego_translation, nusc_ego_rotation,
                        nusc_box_translations, nusc_box_rotations,
                        best_rotation_D, carla_ego_location):
    """
    3-step transform of a nuScenes scene to CARLA world coordinates.
    Step 1 — Rotate all poses by best_rotation_D (CCW, degrees).
    Step 2 — Convert right-handed (nuScenes) to left-handed (CARLA).
    Step 3 — Translate so the ego matches carla_ego_location.
    """
    D_rad = math.radians(best_rotation_D)
    cosD, sinD = math.cos(D_rad), math.sin(D_rad)

    def step1_rotate(x, y, z, yaw_rad):
        return (x * cosD - y * sinD,
                x * sinD + y * cosD,
                z), yaw_rad + D_rad

    def step2_to_carla(x, y, z, yaw_rad):
        return (x, -y, z), -yaw_rad

    def process_pose(translation, rotation):
        x, y, z = translation
        q = Quaternion(rotation)
        yaw = nusc_quaternion_yaw(q)
        (x, y, z), yaw = step1_rotate(x, y, z, yaw)
        (x, y, z), yaw = step2_to_carla(x, y, z, yaw)
        return (x, y, z), yaw

    ego_carla_pos, ego_carla_yaw = process_pose(
        nusc_ego_translation, nusc_ego_rotation)

    Tx = carla_ego_location[0] - ego_carla_pos[0]
    Ty = carla_ego_location[1] - ego_carla_pos[1]
    Tz = carla_ego_location[2] - ego_carla_pos[2]

    def final_pose(translation, yaw_rad):
        x, y, z = translation
        yaw_deg = math.degrees(yaw_rad)
        yaw_deg = ((yaw_deg + 180) % 360) - 180
        return {"location": (x + Tx, y + Ty, z + Tz), "yaw_deg": yaw_deg}

    ego_result = final_pose(ego_carla_pos, ego_carla_yaw)

    box_results = []
    for t, r in zip(nusc_box_translations, nusc_box_rotations):
        pos, yaw = process_pose(t, r)
        box_results.append(final_pose(pos, yaw))

    return ego_result, box_results


def check_scene_invariants(nusc_poses, carla_poses, nusc_yaws_deg, carla_yaws_deg, eps=1e-4):
    """Assert all pairwise distances and relative-yaw magnitudes are unchanged after the 3-step transform."""
    n = len(nusc_poses)
    ok = True
    for i in range(n):
        n_i = np.array(nusc_poses[i])
        c_i = np.array(carla_poses[i])
        for j in range(i + 1, n):
            n_j = np.array(nusc_poses[j])
            c_j = np.array(carla_poses[j])
            d_before = float(np.linalg.norm(n_i - n_j))
            d_after = float(np.linalg.norm(c_i - c_j))
            if abs(d_before - d_after) > eps:
                logger.warning("Sanity: distance (pose %d, %d) changed: %.6f -> %.6f", i, j, d_before, d_after)
                ok = False
            yaw_diff_before = (nusc_yaws_deg[i] - nusc_yaws_deg[j] + 180) % 360 - 180
            yaw_diff_after = (carla_yaws_deg[i] - carla_yaws_deg[j] + 180) % 360 - 180
            if abs(abs(yaw_diff_before) - abs(yaw_diff_after)) > eps:
                logger.warning("Sanity: relative yaw (pose %d, %d) changed: |%.1f| -> |%.1f|", i, j, yaw_diff_before, yaw_diff_after)
                ok = False
    if ok:
        logger.info("Sanity: 3-step transform preserves all pairwise distances and relative yaws (%d poses)", n)
    return ok

NUSC_TO_CARLA = {
    "movable_object.barrier": "static.prop.streetbarrier",
    "movable_object.debris": "static.prop.dirtdebris01",
    "movable_object.pushable_pullable": "static.prop.bin",
    "movable_object.trafficcone": "static.prop.trafficcone01",
}

_VEHICLE_BLUEPRINT_CANDIDATES = {
    "vehicle.car": [
        "vehicle.audi.a2", "vehicle.audi.etron", "vehicle.audi.tt",
        "vehicle.bmw.grandtourer", "vehicle.chevrolet.impala", "vehicle.citroen.c3",
        "vehicle.dodge.charger_2020", "vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020",
        "vehicle.ford.ambulance", "vehicle.ford.crown", "vehicle.ford.mustang",
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe", "vehicle.mercedes.coupe_2020", "vehicle.mercedes.sprinter",
        "vehicle.micro.microlino", "vehicle.mini.cooper_s", "vehicle.mini.cooper_s_2021",
        "vehicle.nissan.micra", "vehicle.nissan.patrol", "vehicle.nissan.patrol_2021",
        "vehicle.seat.leon", "vehicle.tesla.cybertruck", "vehicle.tesla.model3",
        "vehicle.toyota.prius", "vehicle.volkswagen.t2", "vehicle.volkswagen.t2_2021",
    ],
    "vehicle.truck": [
        "vehicle.carlamotors.european_hgv", "vehicle.carlamotors.firetruck",
        "vehicle.mitsubishi.fusorosa",
    ],
    "vehicle.bus.bendy": ["vehicle.carlamotors.european_hgv"],
    "vehicle.bus.rigid": ["vehicle.carlamotors.european_hgv"],
    "vehicle.construction": ["vehicle.carlamotors.firetruck"],
    "vehicle.emergency.ambulance": ["vehicle.ford.ambulance"],
    "vehicle.emergency.police": ["vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020"],
    "vehicle.motorcycle": [
        "vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja",
        "vehicle.vespa.zx125", "vehicle.yamaha.yzf",
    ],
    "vehicle.bicycle": [
        "vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets",
    ],
    "vehicle.trailer": ["vehicle.carlamotors.carlacola"],
}

_PEDESTRIAN_CACHE = {"adult": [], "child": []}
SKIP_CATEGORIES = {
    "noise", "animal",
    "static_object.bicycle_rack",
    "flat.driveable_surface", "flat.other", "flat.sidewalk", "flat.terrain",
    "static.manmade", "static.other", "static.vegetation",
}
IMAGE_W, IMAGE_H = 1600, 900

_MEASURED_DIMS = {}

_GROUND_OFFSET_CACHE = {}


def _build_ground_offset_cache(carla_world, bp_lib):
    """Pre-measure ground offset (pivot type + half-height) for every blueprint we might spawn.

    Uses the bbox.location.z heuristic from notes/spawn-actor-ground.md:
      - |bbox.location.z| < 0.1  → pivot at geometric center (walker/prop)
      - otherwise                → pivot at bottom (vehicle)

    Results are cached to ``bp-z.json`` to avoid re-measuring across runs.
    """
    cache_path = Path(__file__).parent / "bp-z.json"
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
            _GROUND_OFFSET_CACHE.clear()
            _GROUND_OFFSET_CACHE.update(raw)
        logger.info("Loaded %d cached ground offsets from %s", len(_GROUND_OFFSET_CACHE), cache_path)
        return

    all_ids = set()
    for ids in _VEHICLE_BLUEPRINT_CANDIDATES.values():
        all_ids.update(ids)
    for age in _PEDESTRIAN_CACHE.values():
        all_ids.update(age)
    for bp_id in NUSC_TO_CARLA.values():
        all_ids.add(bp_id)
    all_ids.add(EGO_BLUEPRINT)

    logger.info("Measuring ground offsets for %d blueprints (spawn @ z=200, measure, destroy) ...", len(all_ids))
    for bp_id in sorted(all_ids):
        try:
            bp = bp_lib.find(bp_id)
        except KeyError:
            continue
        temp_tf = carla.Transform(carla.Location(x=0, y=0, z=200))
        actor = carla_world.try_spawn_actor(bp, temp_tf)
        if actor is None:
            continue
        bbox = actor.bounding_box
        pivot_to_center = bbox.location.z
        half_height = bbox.extent.z
        if abs(pivot_to_center) < 0.1:
            _GROUND_OFFSET_CACHE[bp_id] = {"type": "center", "half_height": half_height}
        else:
            _GROUND_OFFSET_CACHE[bp_id] = {"type": "bottom"}
        actor.destroy()
    logger.info("Cached ground offsets for %d blueprints", len(_GROUND_OFFSET_CACHE))

    with open(cache_path, "w") as f:
        json.dump(_GROUND_OFFSET_CACHE, f, indent=2)
    logger.info("Saved ground offsets to %s", cache_path)


def _ground_z(carla_map, x, y):
    """Return the road-surface Z at (x, y), or 0.0 if no waypoint found."""
    loc = carla.Location(x=x, y=y, z=0.0)
    wp = carla_map.get_waypoint(loc, project_to_road=True)
    return wp.transform.location.z if wp else 0.0


def _final_z(carla_map, x, y, bp_id):
    """Return the Z that places the actor's bottom on the ground at (x, y).

    Walkers/props (center pivot) are lifted by half their height so their
    feet/base touch the ground.  Vehicles (bottom pivot) sit on their wheels
    with a 5 cm buffer.
    """
    ground_z = _ground_z(carla_map, x, y)
    info = _GROUND_OFFSET_CACHE.get(bp_id, {"type": "bottom"})
    if info["type"] == "center":
        return ground_z + info["half_height"]
    return ground_z + 0.1


_NUSC_PEDESTRIAN_AGE = {
    "human.pedestrian.adult": "adult",
    "human.pedestrian.child": "child",
    "human.pedestrian.construction_worker": "adult",
    "human.pedestrian.personal_mobility": "adult",
    "human.pedestrian.police_officer": "adult",
    "human.pedestrian.stroller": "adult",
    "human.pedestrian.wheelchair": "adult",
}


def measure_blueprint_dims(bp, carla_world):
    """Spawn a blueprint temporarily at z=200, measure its bounding-box extent, destroy, return (L, W, H)."""
    temp_tf = carla.Transform(carla.Location(x=0, y=0, z=200))
    actor = carla_world.try_spawn_actor(bp, temp_tf)
    if actor is None:
        return None
    ext = actor.bounding_box.extent
    dims = (round(ext.x * 2, 2), round(ext.y * 2, 2), round(ext.z * 2, 2))
    actor.destroy()
    return dims


def _build_dim_cache(carla_world, bp_lib):
    """Measure bounding-box extents of all blueprints we might spawn and cache to ``bp-size.json``.

    Skips measurement when ``bp-size.json`` already exists — load that instead.
    """
    cache_path = Path(__file__).parent / "bp-size.json"
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
            _MEASURED_DIMS.update({k: tuple(v) for k, v in raw.items()})
        logger.info("Loaded %d cached blueprint dimensions from %s", len(_MEASURED_DIMS), cache_path)
        return

    all_ids = set()
    for ids in _VEHICLE_BLUEPRINT_CANDIDATES.values():
        all_ids.update(ids)
    for age in _PEDESTRIAN_CACHE.values():
        all_ids.update(age)
    for bp_id in NUSC_TO_CARLA.values():
        all_ids.add(bp_id)
    all_ids.add(EGO_BLUEPRINT)

    logger.info("Measuring %d blueprints (spawn @ z=200, measure, destroy) ...", len(all_ids))
    for bp_id in sorted(all_ids):
        try:
            bp = bp_lib.find(bp_id)
        except KeyError:
            continue
        dims = measure_blueprint_dims(bp, carla_world)
        if dims:
            _MEASURED_DIMS[bp_id] = dims
    logger.info("Cached dimensions for %d blueprints", len(_MEASURED_DIMS))

    with open(cache_path, "w") as f:
        json.dump({k: list(v) for k, v in _MEASURED_DIMS.items()}, f, indent=2)
    logger.info("Saved blueprint dimensions to %s", cache_path)


def _build_pedestrian_cache(bp_lib):
    """Cache pedestrian blueprints split by age group (adult / child)."""
    for bp in bp_lib.filter("walker.pedestrian.*"):
        age = bp.get_attribute("age").as_str()
        if age == "child":
            _PEDESTRIAN_CACHE["child"].append(bp.id)
        else:
            _PEDESTRIAN_CACHE["adult"].append(bp.id)


def _select_blueprint(nusc_cat, bp_lib, rng):
    """Pick a CARLA blueprint matching a nuScenes annotation category (pedestrian / vehicle / static)."""
    if nusc_cat.startswith("human.pedestrian"):
        pool = _PEDESTRIAN_CACHE.get(_NUSC_PEDESTRIAN_AGE.get(nusc_cat, "adult"), [])
        if not pool:
            raise KeyError(nusc_cat)
        return bp_lib.find(rng.choice(pool))

    if not nusc_cat.startswith("vehicle."):
        bp_id = NUSC_TO_CARLA.get(nusc_cat)
        if bp_id:
            return bp_lib.find(bp_id)
        raise KeyError(nusc_cat)

    candidates = _VEHICLE_BLUEPRINT_CANDIDATES.get(nusc_cat, [])
    if not candidates:
        raise KeyError(nusc_cat)
    return bp_lib.find(rng.choice(candidates))


def _select_blueprint_best_fit(nusc_size, candidates, bp_lib):
    """Pick the candidate whose measured CARLA dims have the smallest L2 error vs *nusc_size*.

    When multiple candidates tie (same dimensions), one is chosen at random
    so that not all pedestrians collapse to the same blueprint.

    *nusc_size* is ``[w, l, h]`` from the nuScenes annotation.
    *candidates* is a list of CARLA blueprint ID strings.
    """
    best_candidates = []
    best_err = float("inf")
    nusc_w, nusc_l, nusc_h = nusc_size
    for bp_id in candidates:
        dims = _MEASURED_DIMS.get(bp_id)
        if dims is None:
            continue
        carla_l, carla_w, carla_h = dims
        err = math.sqrt((nusc_w - carla_w) ** 2 + (nusc_l - carla_l) ** 2 + (nusc_h - carla_h) ** 2)
        if err < best_err - 1e-6:
            best_candidates = [bp_id]
            best_err = err
        elif abs(err - best_err) < 1e-6:
            best_candidates.append(bp_id)
    if not best_candidates:
        raise KeyError(f"no candidate with measured dims among {candidates}")
    best = random.choice(best_candidates)
    logger.debug("  best-fit for nusc_size %s: %s (L2=%.3f)", nusc_size, best, best_err)
    return bp_lib.find(best)


def _compute_ego_velocity(meta, target_sample):
    """Compute ego linear velocity (vx, vy, vz) and yaw rate (rad/s) from adjacent frames."""
    sd_curr = get_cam_front(meta, target_sample)
    ep_curr = meta.ego_dict()[sd_curr["ego_pose_token"]]
    curr_rec = next(s for s in meta.samples if s["token"] == target_sample)
    has_next = curr_rec["next"] != ""
    adj_token = curr_rec["next"] if has_next else curr_rec["prev"]
    if not adj_token:
        return (0., 0., 0.), 0.
    try:
        sd_adj = get_cam_front(meta, adj_token)
    except RuntimeError:
        return (0., 0., 0.), 0.
    ep_adj = meta.ego_dict()[sd_adj["ego_pose_token"]]
    dt = abs(ep_adj["timestamp"] - ep_curr["timestamp"]) / 1e6
    if dt == 0:
        return (0., 0., 0.), 0.
    curr_p = np.array(ep_curr["translation"])
    adj_p = np.array(ep_adj["translation"])
    vel = tuple((adj_p - curr_p) / dt if has_next else (curr_p - adj_p) / dt)
    q_curr = Quaternion(ep_curr["rotation"])
    yaw_curr = math.atan2(q_curr.rotate([1, 0, 0])[1], q_curr.rotate([1, 0, 0])[0])
    q_adj = Quaternion(ep_adj["rotation"])
    yaw_adj = math.atan2(q_adj.rotate([1, 0, 0])[1], q_adj.rotate([1, 0, 0])[0])
    yaw_diff = (yaw_adj - yaw_curr + math.pi) % (2 * math.pi) - math.pi
    if not has_next:
        yaw_diff = -yaw_diff
    return vel, yaw_diff / dt


def _compute_ann_velocity(meta, ann):
    """Compute annotation linear velocity (vx, vy, 0) and yaw rate (rad/s) from adjacent frames.

    Raises RuntimeError with a detailed message when velocity cannot be computed.
    """
    ann_token = ann.get("token", "?")
    has_next = ann["next"] != ""
    target_token = ann["next"] if has_next else ann["prev"]
    if not target_token:
        raise RuntimeError(
            f"Annotation {ann_token} has neither 'next' nor 'prev' — "
            f"cannot compute velocity from adjacent frames"
        )
    target_ann = next((a for a in meta.annotations if a["token"] == target_token), None)
    if target_ann is None:
        raise RuntimeError(
            f"Annotation {ann_token}: adjacent frame token {target_token} "
            f"not found in meta.annotations"
        )
    cs = next((s for s in meta.samples if s["token"] == ann["sample_token"]), None)
    ts = next((s for s in meta.samples if s["token"] == target_ann["sample_token"]), None)
    if cs is None or ts is None:
        raise RuntimeError(
            f"Annotation {ann_token}: sample not found for "
            f"current token={ann['sample_token']} or target token={target_ann['sample_token']}"
        )
    dt = abs(ts["timestamp"] - cs["timestamp"]) / 1e6
    if dt == 0:
        raise RuntimeError(
            f"Annotation {ann_token}: zero time delta between "
            f"sample {ann['sample_token']} and {target_ann['sample_token']}"
        )
    if has_next:
        vel = ((target_ann["translation"][0] - ann["translation"][0]) / dt,
               (target_ann["translation"][1] - ann["translation"][1]) / dt, 0.0)
    else:
        vel = ((ann["translation"][0] - target_ann["translation"][0]) / dt,
               (ann["translation"][1] - target_ann["translation"][1]) / dt, 0.0)
    q_curr = Quaternion(ann["rotation"])
    yaw_curr = math.atan2(q_curr.rotate([1, 0, 0])[1], q_curr.rotate([1, 0, 0])[0])
    q_target = Quaternion(target_ann["rotation"])
    yaw_target = math.atan2(q_target.rotate([1, 0, 0])[1], q_target.rotate([1, 0, 0])[0])
    yaw_diff = (yaw_target - yaw_curr + math.pi) % (2 * math.pi) - math.pi
    if not has_next:
        yaw_diff = -yaw_diff
    return vel, yaw_diff / dt


def _set_actor_velocity(actor, carla_vel, carla_yaw_rate):
    """Set linear and angular velocity on a CARLA actor (global frame)."""
    actor.set_target_velocity(carla.Vector3D(x=carla_vel[0], y=carla_vel[1], z=carla_vel[2]))
    actor.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=carla_yaw_rate))


def _velocity_log_entry(nusc_vel, carla_vel, nusc_yaw_rate, carla_yaw_rate):
    if nusc_vel is None:
        return {}
    return {
        "nusc_velocity": [round(v, 4) for v in nusc_vel],
        "carla_velocity": [round(v, 4) for v in carla_vel],
        "nusc_yaw_rate": round(nusc_yaw_rate, 4),
        "carla_yaw_rate": round(carla_yaw_rate, 4),
    }


def _build_annotation_commands(sample_anns, box_carla_list, cat_dict, inst_dict,
                                bp_lib, carla_map, color_cache, spawn_log,
                                ann_vels=None):
    """Build spawn commands for every non-skipped annotation.

    Appends skip entries to *spawn_log*.  Returns ``(commands, skip_count)``.
    """
    commands = []
    skip_count = 0
    all_vehicle_ids = [k for k in _MEASURED_DIMS if k.startswith("vehicle.")]
    for i, ann in enumerate(sample_anns):
        inst = inst_dict.get(ann["instance_token"], {})
        cat_name = cat_dict.get(inst.get("category_token", ""), "unknown")

        if cat_name in SKIP_CATEGORIES:
            logger.info("  SKIP %-20s reason=in SKIP_CATEGORIES", cat_name)
            skip_count += 1
            spawn_log.append({"status": "skipped", "category": cat_name, "reason": "in SKIP_CATEGORIES", "ann_token": ann["token"]})
            continue

        try:
            ann_size = list(ann["size"])
            if cat_name.startswith("human.pedestrian"):
                pool = _PEDESTRIAN_CACHE.get(_NUSC_PEDESTRIAN_AGE.get(cat_name, "adult"), [])
                if not pool:
                    raise KeyError(cat_name)
                bp = _select_blueprint_best_fit(ann_size, pool, bp_lib)
            elif cat_name.startswith("vehicle."):
                if not all_vehicle_ids:
                    raise KeyError(cat_name)
                bp = _select_blueprint_best_fit(ann_size, all_vehicle_ids, bp_lib)
            else:
                bp_id = NUSC_TO_CARLA.get(cat_name)
                if bp_id is None:
                    raise KeyError(cat_name)
                bp = bp_lib.find(bp_id)
        except KeyError:
            logger.info("  SKIP %-20s reason=no CARLA blueprint mapping", cat_name)
            skip_count += 1
            spawn_log.append({"status": "skipped", "category": cat_name, "reason": "no CARLA blueprint mapping", "ann_token": ann["token"]})
            continue

        bp_id = bp.id if hasattr(bp, "id") else str(bp)
        if color_cache and bp.has_attribute("color"):
            c = color_cache.get(ann["token"])
            if c:
                bp.set_attribute("color", c)

        box = box_carla_list[i]
        bx, by = box["location"][0], box["location"][1]
        final_z = _final_z(carla_map, bx, by, bp_id)
        global_yaw = box["yaw_deg"]
        spawn_loc = carla.Location(x=bx, y=by, z=final_z)
        logger.debug("  spawn_pos=(%.3f, %.3f, %.3f) yaw=%.1f",
                     spawn_loc.x, spawn_loc.y, spawn_loc.z, global_yaw)
        tf = carla.Transform(spawn_loc, carla.Rotation(yaw=global_yaw))
        cmd = {
            "bp": bp, "bp_id": bp_id, "tf": tf,
            "cat_name": cat_name, "ann_token": ann["token"],
            "nusc_size": ann_size,
        }
        if ann_vels:
            cmd["nusc_vel"] = ann_vels[i]["nusc_vel"]
            cmd["carla_vel"] = ann_vels[i]["carla_vel"]
            cmd["nusc_yaw_rate"] = ann_vels[i]["nusc_yaw_rate"]
            cmd["carla_yaw_rate"] = ann_vels[i]["carla_yaw_rate"]
        commands.append(cmd)

    return commands, skip_count


def _spawn_ego_actor(bp_lib, ego_carla, carla_client, carla_world, spawn_log,
                     carla_vel=None, carla_yaw_rate=0.0, nusc_vel=None, nusc_yaw_rate=0.0):
    """Spawn the ego vehicle using the original spawn-point Z.  Returns the actor (or ``None``).

    Tries ``apply_batch_sync`` first; falls back to ``try_spawn_actor`` when the
    batch spawn fails with a collision (which happens occasionally due to
    non-determinism in CARLA's batch spawning).
    """
    ego_bp = bp_lib.find(EGO_BLUEPRINT)
    ego_bp.set_attribute("role_name", "ego_vehicle")
    ego_bp.set_attribute("color", "255,255,255")
    ego_x, ego_y, ego_z = ego_carla["location"]
    ego_tf = carla.Transform(
        carla.Location(x=ego_x, y=ego_y, z=ego_z + 0.1),
        carla.Rotation(yaw=ego_carla["yaw_deg"]),
    )

    ego_cmd = carla.command.SpawnActor(ego_bp, ego_tf)
    result = carla_client.apply_batch_sync([ego_cmd], True)[0]
    if not result.error:
        ego_actor = carla_world.get_actor(result.actor_id)
        if ego_actor is not None and carla_vel is not None:
            _set_actor_velocity(ego_actor, carla_vel, carla_yaw_rate)
            ego_actor.set_simulate_physics(False)
        _log_spawned_ego(ego_tf, spawn_log, carla_vel, carla_yaw_rate, nusc_vel, nusc_yaw_rate)
        return ego_actor

    # Fallback: batch spawn failed (likely collision) — try individually.
    logger.warning("  EGO batch spawn failed (%s); retrying with try_spawn_actor ...", result.error)
    ego_actor = carla_world.try_spawn_actor(ego_bp, ego_tf)
    if ego_actor is None:
        logger.error("  FAILED ego at (%.1f, %.1f, %.1f): try_spawn_actor also failed",
                     ego_tf.location.x, ego_tf.location.y, ego_tf.location.z)
        entry = {
            "status": "failed", "error": "try_spawn_actor returned None",
            "category": "ego", "blueprint": EGO_BLUEPRINT,
            "location": [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z],
            "ann_token": None,
        }
        entry.update(_size_log_entry(None, EGO_BLUEPRINT))
        if nusc_vel is not None:
            entry.update(_velocity_log_entry(nusc_vel, carla_vel, nusc_yaw_rate, carla_yaw_rate))
        spawn_log.append(entry)
        return None

    if carla_vel is not None:
        _set_actor_velocity(ego_actor, carla_vel, carla_yaw_rate)
    ego_actor.set_simulate_physics(False)
    logger.info("  SPAWNED ego (fallback) at (%.1f, %.1f, %.1f) yaw=%.0f",
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z, ego_tf.rotation.yaw)
    entry = {
        "status": "spawned",
        "category": "ego", "blueprint": EGO_BLUEPRINT,
        "location": [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z],
        "rotation": [ego_tf.rotation.pitch, ego_tf.rotation.yaw, ego_tf.rotation.roll],
        "ann_token": None,
    }
    entry.update(_size_log_entry(None, EGO_BLUEPRINT))
    if nusc_vel is not None:
        entry.update(_velocity_log_entry(nusc_vel, carla_vel, nusc_yaw_rate, carla_yaw_rate))
    spawn_log.append(entry)
    return ego_actor


def _log_spawned_ego(ego_tf, spawn_log,
                     carla_vel=None, carla_yaw_rate=0.0,
                     nusc_vel=None, nusc_yaw_rate=0.0):
    """Log a successful ego spawn with blueprint size info."""
    size_entry = _size_log_entry(None, EGO_BLUEPRINT)
    logger.info("  SPAWNED ego at (%.1f, %.1f, %.1f) yaw=%.0f",
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z, ego_tf.rotation.yaw)
    entry = {
        "status": "spawned",
        "category": "ego", "blueprint": EGO_BLUEPRINT,
        "location": [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z],
        "rotation": [ego_tf.rotation.pitch, ego_tf.rotation.yaw, ego_tf.rotation.roll],
        "ann_token": None,
    }
    entry.update(size_entry)
    if nusc_vel is not None:
        entry.update(_velocity_log_entry(nusc_vel, carla_vel, nusc_yaw_rate, carla_yaw_rate))
    spawn_log.append(entry)


def _size_log_entry(nusc_size, bp_id):
    """Return dict with nusc bbox size, CARLA blueprint size, and their L2 error, or empty dict if unknown."""
    carla_dims = _MEASURED_DIMS.get(bp_id)
    if carla_dims is None:
        return {}
    carla_l, carla_w, carla_h = carla_dims
    if nusc_size is None:
        return {"carla_size": [carla_l, carla_w, carla_h]}
    nusc_w, nusc_l, nusc_h = nusc_size
    err = math.sqrt((nusc_w - carla_w) ** 2 + (nusc_l - carla_l) ** 2 + (nusc_h - carla_h) ** 2)
    return {
        "nusc_size": [nusc_w, nusc_l, nusc_h],
        "carla_size": [carla_l, carla_w, carla_h],
        "size_l2": round(err, 3),
    }


def _spawn_annotations_sequential(commands, carla_world, spawn_log):
    """Spawn all annotation actors individually via ``world.try_spawn_actor``.

    When the initial spawn at the target position fails, up to
    ``_MAX_SNAP_ATTEMPTS`` retries are performed at small random offsets
    (within ``_SNAP_RADIUS``) so that actors that would otherwise be blocked
    by a minor collision can still be placed.

    Returns ``(spawned, snapped, failed, vel_map)`` counts and appends entries to *spawn_log*.
    """
    spawned = 0
    snapped_count = 0
    failed = 0
    vel_map = {}
    for c in commands:
        actor = carla_world.try_spawn_actor(c["bp"], c["tf"])
        if actor is None:
            actor = try_spawn_with_nudge(carla_world, c["bp"], c["tf"])
            if actor is not None:
                snapped_count += 1
                _set_actor_velocity(actor, c["carla_vel"], c["carla_yaw_rate"])
                actor.set_simulate_physics(False)
                vel_map[str(actor.id)] = {
                    "velocity": list(c.get("carla_vel", [0.0, 0.0, 0.0])),
                    "angular_velocity": [0.0, 0.0, c.get("carla_yaw_rate", 0.0)],
                }
                _log_snapped(c, actor.get_location(), spawn_log)
            else:
                # Try smaller vehicle blueprints before force-spawning
                actor = _try_smaller_blueprint(c, carla_world, spawn_log)
                if actor is not None:
                    snapped_count += 1
                    vel_map[str(actor.id)] = {
                        "velocity": list(c.get("carla_vel", [0.0, 0.0, 0.0])),
                        "angular_velocity": [0.0, 0.0, c.get("carla_yaw_rate", 0.0)],
                    }
                    spawned += 1
                    continue

                # Last resort: force-spawn at the original position with physics off.
                try:
                    actor = spawn_actor_forced(carla_world, c["bp"], c["tf"])
                    snapped_count += 1
                    spawned += 1
                    vel_map[str(actor.id)] = {
                        "velocity": list(c.get("carla_vel", [0.0, 0.0, 0.0])),
                        "angular_velocity": [0.0, 0.0, c.get("carla_yaw_rate", 0.0)],
                    }
                    _log_forced(c, spawn_log)
                    continue
                except RuntimeError:
                    failed += 1
                    short = c["cat_name"].split(".")[-1]
                    logger.warning("  FAILED %-20s at (%.1f, %.1f, %.1f): spawn_actor also failed",
                                   short, c["tf"].location.x, c["tf"].location.y, c["tf"].location.z)
                    entry = {
                        "status": "failed", "error": "spawn_actor raised RuntimeError",
                        "category": c["cat_name"], "blueprint": c["bp_id"],
                        "location": [c["tf"].location.x, c["tf"].location.y, c["tf"].location.z],
                        "ann_token": c["ann_token"],
                    }
                    entry.update(_size_log_entry(c.get("nusc_size"), c["bp_id"]))
                    entry.update(_velocity_log_entry(
                        c.get("nusc_vel"), c.get("carla_vel"),
                        c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
                    spawn_log.append(entry)
                    continue
        _set_actor_velocity(actor, c["carla_vel"], c["carla_yaw_rate"])
        actor.set_simulate_physics(False)
        vel_map[str(actor.id)] = {
            "velocity": list(c.get("carla_vel", [0.0, 0.0, 0.0])),
            "angular_velocity": [0.0, 0.0, c.get("carla_yaw_rate", 0.0)],
        }
        spawned += 1
        short = c["cat_name"].split(".")[-1]
        logger.debug("  SPAWNED %-20s at (%.1f, %.1f, %.1f) yaw=%.0f",
                     short, c["tf"].location.x, c["tf"].location.y, c["tf"].location.z, c["tf"].rotation.yaw)
        entry = {
            "status": "spawned",
            "category": c["cat_name"], "blueprint": c["bp_id"],
            "location": [c["tf"].location.x, c["tf"].location.y, c["tf"].location.z],
            "rotation": [c["tf"].rotation.pitch, c["tf"].rotation.yaw, c["tf"].rotation.roll],
            "ann_token": c["ann_token"],
        }
        entry.update(_size_log_entry(c.get("nusc_size"), c["bp_id"]))
        entry.update(_velocity_log_entry(
            c.get("nusc_vel"), c.get("carla_vel"),
            c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
        spawn_log.append(entry)
    return spawned, snapped_count, failed, vel_map


def _build_vehicles_by_volume():
    """Return list of (bp_id, volume) sorted ascending by volume (L*W*H).

    Includes all vehicle blueprints with measured dimensions so the fallback
    chain tries close-in-size matches first (any vehicle that is strictly
    smaller than the original), eventually reaching tiny blueprints like
    bicycles only when nothing larger fits.
    """
    vehicles = []
    for bp_id, dims in _MEASURED_DIMS.items():
        if bp_id.startswith("vehicle."):
            l, w, h = dims
            vehicles.append((bp_id, l * w * h))
    vehicles.sort(key=lambda x: x[1])
    return vehicles


_VEHICLES_BY_VOLUME = None


def _try_smaller_blueprint(c, carla_world, spawn_log):
    """Try spawning with a smaller vehicle blueprint when the original collides.

    Iterates through car/truck blueprints in *descending* volume order
    (closest to original size first) and attempts ``try_spawn_actor`` at the
    original position.  Returns the actor on first success, or ``None`` if
    none fits.
    """
    global _VEHICLES_BY_VOLUME
    if _VEHICLES_BY_VOLUME is None:
        _VEHICLES_BY_VOLUME = _build_vehicles_by_volume()

    if not c["cat_name"].startswith("vehicle."):
        return None

    orig_dims = _MEASURED_DIMS.get(c["bp_id"])
    if orig_dims is None:
        return None
    orig_vol = orig_dims[0] * orig_dims[1] * orig_dims[2]

    bp_lib = carla_world.get_blueprint_library()
    # Reverse iterate: largest (closest to original) first
    for smaller_bp_id, vol in reversed(_VEHICLES_BY_VOLUME):
        if smaller_bp_id == c["bp_id"] or vol >= orig_vol:
            continue
        try:
            smaller_bp = bp_lib.find(smaller_bp_id)
        except KeyError:
            continue
        actor = carla_world.try_spawn_actor(smaller_bp, c["tf"])
        if actor is not None:
            _set_actor_velocity(actor, c["carla_vel"], c["carla_yaw_rate"])
            actor.set_simulate_physics(False)
            short = c["cat_name"].split(".")[-1]
            logger.debug("  FALLBACK %-20s %s -> %s at (%.1f, %.1f, %.1f)",
                         short, c["bp_id"], smaller_bp_id,
                         c["tf"].location.x, c["tf"].location.y, c["tf"].location.z)
            entry = {
                "status": "fallback",
                "category": c["cat_name"],
                "original_blueprint": c["bp_id"],
                "fallback_blueprint": smaller_bp_id,
                "location": [c["tf"].location.x, c["tf"].location.y, c["tf"].location.z],
                "ann_token": c["ann_token"],
            }
            entry.update(_velocity_log_entry(
                c.get("nusc_vel"), c.get("carla_vel"),
                c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
            spawn_log.append(entry)
            return actor
    return None


_SNAP_RADIUS = 1.0
_MAX_SNAP_ATTEMPTS = 8


def try_spawn_with_nudge(world, blueprint, transform, max_tries=8):
    """Tries to spawn a vehicle; if it fails, nudges it laterally away from the curb.

    Nudge distances increase progressively so that actors sitting on a sidewalk
    (typically 1-3 m wide) get pushed onto the road surface.
    """
    current_transform = carla.Transform(transform.location, transform.rotation)
    right_vector = transform.get_right_vector()
    # Progressively larger lateral nudges: 0.3, 0.3, 0.8, 0.8, 1.5, 1.5, 2.5, 2.5 m
    nudge_steps = [0.3, 0.3, 0.8, 0.8, 1.5, 1.5, 2.5, 2.5]

    for i in range(min(max_tries, len(nudge_steps))):
        actor = world.try_spawn_actor(blueprint, current_transform)
        if actor is not None:
            return actor

        # Alternate left (+) / right (-) within each distance pair
        direction = 1 if i % 2 == 0 else -1
        nudge = nudge_steps[i] * direction

        current_transform.location.x = transform.location.x + right_vector.x * nudge
        current_transform.location.y = transform.location.y + right_vector.y * nudge
        current_transform.location.z = transform.location.z + 0.02

    return None


def spawn_actor_forced(world, blueprint, transform):
    """Force-spawn exactly at *transform*, bypassing collision checks.

    Physics is disabled immediately to prevent the actor from exploding or
    glitching against surrounding geometry.
    """
    actor = world.spawn_actor(blueprint, transform)
    actor.set_simulate_physics(False)
    return actor


def _log_snapped(c, snapped_location, spawn_log):
    """Log a successful nudge spawn."""
    orig = c["tf"].location
    short = c["cat_name"].split(".")[-1]
    logger.debug("  SNAPPED %-20s from (%.1f,%.1f) to (%.1f,%.1f)",
                 short, orig.x, orig.y, snapped_location.x, snapped_location.y)
    entry = {
        "status": "snapped",
        "category": c["cat_name"], "blueprint": c["bp_id"],
        "original_location": [orig.x, orig.y, orig.z],
        "snapped_location": [snapped_location.x, snapped_location.y, snapped_location.z],
        "ann_token": c["ann_token"],
    }
    entry.update(_size_log_entry(c.get("nusc_size"), c["bp_id"]))
    entry.update(_velocity_log_entry(
        c.get("nusc_vel"), c.get("carla_vel"),
        c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
    spawn_log.append(entry)


def _log_forced(c, spawn_log):
    """Log a force-spawned actor."""
    loc = c["tf"].location
    short = c["cat_name"].split(".")[-1]
    logger.debug("  FORCED %-20s at (%.1f, %.1f, %.1f) yaw=%.0f",
                 short, loc.x, loc.y, loc.z, c["tf"].rotation.yaw)
    entry = {
        "status": "forced",
        "category": c["cat_name"], "blueprint": c["bp_id"],
        "location": [loc.x, loc.y, loc.z],
        "rotation": [c["tf"].rotation.pitch, c["tf"].rotation.yaw, c["tf"].rotation.roll],
        "ann_token": c["ann_token"],
    }
    entry.update(_size_log_entry(c.get("nusc_size"), c["bp_id"]))
    entry.update(_velocity_log_entry(
        c.get("nusc_vel"), c.get("carla_vel"),
        c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
    spawn_log.append(entry)


def _try_snap_spawn(c, carla_world, rng, spawn_log):
    """Try spawning *c* at small random offsets around its original position.

    Returns the actor on success, or ``None`` if all attempts fail.
    """
    orig = c["tf"].location
    for _ in range(_MAX_SNAP_ATTEMPTS):
        dx = rng.uniform(-_SNAP_RADIUS, _SNAP_RADIUS)
        dy = rng.uniform(-_SNAP_RADIUS, _SNAP_RADIUS)
        snap_loc = carla.Location(x=orig.x + dx, y=orig.y + dy, z=orig.z)
        snap_tf = carla.Transform(snap_loc, c["tf"].rotation)
        actor = carla_world.try_spawn_actor(c["bp"], snap_tf)
        if actor is not None:
            short = c["cat_name"].split(".")[-1]
            logger.debug("  SNAPPED %-20s from (%.1f,%.1f) to (%.1f,%.1f)",
                         short, orig.x, orig.y, snap_loc.x, snap_loc.y)
            entry = {
                "status": "snapped",
                "category": c["cat_name"], "blueprint": c["bp_id"],
                "original_location": [orig.x, orig.y, orig.z],
                "snapped_location": [snap_loc.x, snap_loc.y, snap_loc.z],
                "ann_token": c["ann_token"],
            }
            entry.update(_size_log_entry(c.get("nusc_size"), c["bp_id"]))
            spawn_log.append(entry)
            return actor
    return None


def _batch_spawn_annotations(commands, carla_client, carla_world, spawn_log):
    """Batch-spawn all annotation actors and process results.

    Returns ``(spawned, failed, vel_map)`` counts and appends entries to *spawn_log*.
    """
    cmds = [carla.command.SpawnActor(c["bp"], c["tf"]) for c in commands]
    results = carla_client.apply_batch_sync(cmds, True)
    spawned = 0
    failed = 0
    vel_map = {}
    for c, r in zip(commands, results):
        if r.error:
            failed += 1
            short = c["cat_name"].split(".")[-1]
            logger.warning("  FAILED %-20s at (%.1f, %.1f, %.1f): %s",
                           short, c["tf"].location.x, c["tf"].location.y, c["tf"].location.z, r.error)
            entry = {
                "status": "failed", "error": str(r.error),
                "category": c["cat_name"], "blueprint": c["bp_id"],
                "location": [c["tf"].location.x, c["tf"].location.y, c["tf"].location.z],
                "ann_token": c["ann_token"],
            }
            entry.update(_velocity_log_entry(
                c.get("nusc_vel"), c.get("carla_vel"),
                c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
            spawn_log.append(entry)
        else:
            actor = carla_world.get_actor(r.actor_id)
            if actor is not None:
                _set_actor_velocity(actor, c["carla_vel"], c["carla_yaw_rate"])
                actor.set_simulate_physics(False)
                vel_map[str(actor.id)] = {
                    "velocity": list(c.get("carla_vel", [0.0, 0.0, 0.0])),
                    "angular_velocity": [0.0, 0.0, c.get("carla_yaw_rate", 0.0)],
                }
            spawned += 1
            short = c["cat_name"].split(".")[-1]
            logger.debug("  SPAWNED %-20s at (%.1f, %.1f, %.1f) yaw=%.0f",
                         short, c["tf"].location.x, c["tf"].location.y, c["tf"].location.z, c["tf"].rotation.yaw)
            entry = {
                "status": "spawned",
                "category": c["cat_name"], "blueprint": c["bp_id"],
                "location": [c["tf"].location.x, c["tf"].location.y, c["tf"].location.z],
                "rotation": [c["tf"].rotation.pitch, c["tf"].rotation.yaw, c["tf"].rotation.roll],
                "ann_token": c["ann_token"],
            }
            entry.update(_velocity_log_entry(
                c.get("nusc_vel"), c.get("carla_vel"),
                c.get("nusc_yaw_rate"), c.get("carla_yaw_rate")))
            spawn_log.append(entry)
    return spawned, failed, vel_map


def stage4_spawn_actors(meta: NuscMeta, target_sample: str,
                        spawn_point: dict, carla_world, carla_client, out_dir=None,
                        color_cache: dict = None, seed: int = 0,
                        exclude_ann_tokens: set = None):
    """Transform the nuScenes scene to CARLA (3-step) and batch-spawn all actors including the ego.

    Args:
        exclude_ann_tokens: Optional set of annotation tokens to skip (e.g. barely visible).
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Spawn Annotation Actors")
    logger.info("=" * 60)

    # ── 1. Caches & helpers ──
    bp_lib = carla_world.get_blueprint_library()
    _build_pedestrian_cache(bp_lib)
    carla_map = carla_world.get_map()
    _build_ground_offset_cache(carla_world, bp_lib)
    _build_dim_cache(carla_world, bp_lib)

    cat_dict = meta.cat_dict()
    inst_dict = meta.inst_dict()
    sample_anns = [a for a in meta.annotations if a["sample_token"] == target_sample]
    if exclude_ann_tokens:
        filtered = [a for a in sample_anns if a["token"] not in exclude_ann_tokens]
        logger.info("Excluded %d/%d barely visible annotations", len(sample_anns) - len(filtered), len(sample_anns))
        sample_anns = filtered

    best_rotation_D = spawn_point.get("best_rotation", 0)

    # ── 2. 3-step scene transform (nusc → CARLA) ──
    nusc_ep = meta.ego_dict()[get_cam_front(meta, target_sample)["ego_pose_token"]]
    nusc_ego_pos = np.array(nusc_ep["translation"])
    q_ego = Quaternion(nusc_ep["rotation"])
    nusc_ego_center = nusc_ego_centre(nusc_ego_pos, q_ego)
    carla_ego_loc = (spawn_point["location"]["x"],
                     spawn_point["location"]["y"],
                     spawn_point["location"]["z"])

    box_translations = [ann["translation"] for ann in sample_anns]
    box_rotations = [ann["rotation"] for ann in sample_anns]
    ego_carla, box_carla_list = nusc_scene_to_carla(
        list(nusc_ego_center), nusc_ep["rotation"],
        box_translations, box_rotations,
        best_rotation_D, carla_ego_loc,
    )
    # Use the CARLA XY + yaw from the 3-step transform; Z is recomputed per-actor.
    ego_carla["location"] = (spawn_point["location"]["x"],
                             spawn_point["location"]["y"],
                             spawn_point["location"]["z"])

    # ── 3. Sanity: verify geometry preserved ──
    nusc_yaw_deg = math.degrees(nusc_quaternion_yaw(q_ego))
    nusc_poses = [nusc_ego_center] + [tuple(t) for t in box_translations]
    nusc_yaws = [nusc_yaw_deg] + [math.degrees(nusc_quaternion_yaw(Quaternion(r))) for r in box_rotations]
    carla_poses = [ego_carla["location"]] + [b["location"] for b in box_carla_list]
    carla_yaws = [ego_carla["yaw_deg"]] + [b["yaw_deg"] for b in box_carla_list]
    check_scene_invariants(nusc_poses, carla_poses, nusc_yaws, carla_yaws)

    # ── 4. Compute and transform velocities ──
    logger.info("Computing nuScenes velocities for ego + %d annotations ...", len(sample_anns))
    ego_nusc_vel, ego_nusc_yaw_rate = _compute_ego_velocity(meta, target_sample)
    ego_carla_vel = nusc_velocity_to_carla(*ego_nusc_vel, best_rotation_D)
    ego_carla_yaw_rate = nusc_yaw_rate_to_carla(ego_nusc_yaw_rate)
    logger.info("  Ego: nusc_vel=%s nusc_yaw_rate=%.4f rad/s → carla_vel=%s carla_yaw_rate=%.2f deg/s",
                [round(v, 3) for v in ego_nusc_vel], ego_nusc_yaw_rate,
                [round(v, 3) for v in ego_carla_vel], ego_carla_yaw_rate)

    ann_vels = []
    nusc_vel_list = [ego_nusc_vel]
    carla_vel_list = [ego_carla_vel]
    nusc_yaw_list = [ego_nusc_yaw_rate]
    carla_yaw_list = [ego_carla_yaw_rate]
    for ann in sample_anns:
        try:
            nv, ny = _compute_ann_velocity(meta, ann)
        except RuntimeError as e:
            logger.warning("Skipping velocity for annotation %s: %s", ann.get("token", "?"), e)
            nv, ny = (0., 0., 0.), 0.
        cv = nusc_velocity_to_carla(*nv, best_rotation_D)
        cy = nusc_yaw_rate_to_carla(ny)
        ann_vels.append({
            "nusc_vel": nv, "carla_vel": cv,
            "nusc_yaw_rate": ny, "carla_yaw_rate": cy,
        })
        nusc_vel_list.append(nv)
        carla_vel_list.append(cv)
        nusc_yaw_list.append(ny)
        carla_yaw_list.append(cy)

    check_velocity_invariants(nusc_vel_list, carla_vel_list, nusc_yaw_list, carla_yaw_list)

    # ── 5. Prepare annotation spawn commands ──
    spawn_log = []
    commands, skipped = _build_annotation_commands(
        sample_anns, box_carla_list, cat_dict, inst_dict,
        bp_lib, carla_map, color_cache, spawn_log,
        ann_vels=ann_vels,
    )

    # ── 6. Spawn ego (isolated batch, before annotations) ──
    ego_actor = _spawn_ego_actor(
        bp_lib, ego_carla, carla_client, carla_world, spawn_log,
        carla_vel=ego_carla_vel, carla_yaw_rate=ego_carla_yaw_rate,
        nusc_vel=ego_nusc_vel, nusc_yaw_rate=ego_nusc_yaw_rate,
    )
    ego_spawned = 1 if ego_actor is not None else 0

    # ── 7. Spawn all annotation actors (sequential try_spawn_actor) ──
    spawned, snapped, failed, ann_vel_map = _spawn_annotations_sequential(commands, carla_world, spawn_log)

    # ── 8. Build full velocity map (ego + annotations) ──
    vel_map = {}
    if ego_actor is not None:
        vel_map[str(ego_actor.id)] = {
            "velocity": list(ego_carla_vel),
            "angular_velocity": [0.0, 0.0, ego_carla_yaw_rate],
        }
    for actor_id_str, vel_data in ann_vel_map.items():
        vel_map[actor_id_str] = vel_data
    if out_dir is not None:
        log_json(vel_map, out_dir / "actor_velocities.json", "actor velocity map")

    logger.info("Spawn result: %d spawned, %d snapped, %d failed, %d skipped (of %d total)",
                spawned + ego_spawned, snapped, failed, skipped, len(sample_anns))
    result = {
        "total_annotations": len(sample_anns),
        "spawned": spawned + ego_spawned,
        "spawned_snap": snapped,
        "failed": failed,
        "skipped": skipped,
        "details": spawn_log,
    }
    if out_dir is not None:
        log_json(result, out_dir / "actor_spawn_log.json", "actor spawn log")
    return result, ego_actor, vel_map
