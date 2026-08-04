import logging
import random

from real2sim.core import NuscMeta


logger = logging.getLogger("real2sim")


def find_sample_by_scene_name(meta: NuscMeta, scene_name: str):
    cam_channels = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}

    scene = next((sc for sc in meta.scenes if sc["name"] == scene_name), None)
    if scene is None:
        raise RuntimeError(f"Scene '{scene_name}' not found. Available: " +
                           ", ".join(sorted(set(s["name"] for s in meta.scenes))[:20]))

    logger.info("Found scene: %s (description: %s)", scene["name"],
                scene.get("description", "")[:120])

    sample_tok = scene["first_sample_token"]
    while sample_tok:
        s = next(s for s in meta.samples if s["token"] == sample_tok)
        cams = {}
        for sd in meta.sample_data:
            if sd["sample_token"] != sample_tok or not sd.get("is_key_frame"):
                continue
            ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
            if ch in cam_channels:
                cams[ch] = sd
        if all(ch in cams for ch in cam_channels):
            logger.info("Selected sample %s from scene %s (timestamp %d)",
                        sample_tok, scene_name, s["timestamp"])
            return sample_tok, cams
        sample_tok = s.get("next")

    raise RuntimeError(
        f"No sample with all 6 cameras found in scene '{scene_name}' "
        f"(checked all {scene.get('nbr_samples', '?')} samples)"
    )


def find_random_sample(meta: NuscMeta, seed: int = 0, allowed_scene_names: set = None):
    rng = random.Random(seed)
    cam_channels = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}

    scene_names = sorted(set(s["name"] for s in meta.scenes))
    if allowed_scene_names:
        scene_names = [n for n in scene_names if n in allowed_scene_names]
    rng.shuffle(scene_names)

    scene_by_name = {}
    for sc in meta.scenes:
        scene_by_name.setdefault(sc["name"], []).append(sc)
    sample_by_token = {s["token"]: s for s in meta.samples}

    for name in scene_names:
        for scene in scene_by_name[name]:
            sample_tok = scene["first_sample_token"]
            while sample_tok:
                s = sample_by_token.get(sample_tok)
                if s is None:
                    break
                cams = {}
                for sd in meta.sample_data:
                    if sd["sample_token"] != sample_tok or not sd.get("is_key_frame"):
                        continue
                    ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
                    if ch in cam_channels:
                        cams[ch] = sd
                if all(ch in cams for ch in cam_channels):
                    logger.info("Random scene (seed=%d): %s -> sample %s", seed, name, sample_tok)
                    return sample_tok, cams
                sample_tok = s.get("next")

    raise RuntimeError("No valid sample found in any scene")
