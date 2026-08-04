"""Stage 1–2: weather determination and CARLA town selection from nuScenes metadata.

Requires only nuScenes metadata (no CARLA server).
"""
import logging
import re

from real2sim.core import NuscMeta, log_json


logger = logging.getLogger("real2sim")


WEATHER_PRESETS = {
    "ClearNoon": {"cloudiness": 5.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 45.0, "fog_density": 2.0, "fog_distance": 0.75, "wetness": 0.0},
    "CloudyNoon": {"cloudiness": 60.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 45.0, "fog_density": 3.0, "fog_distance": 0.75, "wetness": 0.0},
    "MidRainyNoon": {"cloudiness": 60.0, "precipitation": 60.0, "precipitation_deposits": 60.0, "wind_intensity": 60.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 45.0, "fog_density": 3.0, "fog_distance": 0.75, "wetness": 0.0},
    "HardRainNoon": {"cloudiness": 100.0, "precipitation": 100.0, "precipitation_deposits": 90.0, "wind_intensity": 100.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 45.0, "fog_density": 7.0, "fog_distance": 0.75, "wetness": 0.0},
    "ClearNight": {"cloudiness": 5.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": -90.0, "fog_density": 60.0, "fog_distance": 75.0, "wetness": 0.0},
    "CloudyNight": {"cloudiness": 60.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": -90.0, "fog_density": 60.0, "fog_distance": 0.75, "wetness": 0.0},
    "MidRainyNight": {"cloudiness": 80.0, "precipitation": 60.0, "precipitation_deposits": 60.0, "wind_intensity": 60.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": -90.0, "fog_density": 60.0, "fog_distance": 0.75, "wetness": 80.0},
    "HardRainNight": {"cloudiness": 100.0, "precipitation": 100.0, "precipitation_deposits": 90.0, "wind_intensity": 100.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": -90.0, "fog_density": 100.0, "fog_distance": 0.75, "wetness": 100.0},
    "WetCloudyNoon": {"cloudiness": 60.0, "precipitation": 0.0, "precipitation_deposits": 50.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 45.0, "fog_density": 3.0, "fog_distance": 0.75, "wetness": 0.0},
    "ClearSunset": {"cloudiness": 5.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 10.0, "sun_azimuth_angle": -1.0, "sun_altitude_angle": 15.0, "fog_density": 2.0, "fog_distance": 0.75, "wetness": 0.0},
}

_WEATHER_RULES = [
    (["rain", "night"], "MidRainyNight"),
    (["rain"], "HardRainNoon"),
    (["night", "cloud"], "CloudyNight"),
    (["night", "wet"], "WetCloudyNoon"),
    (["night"], "ClearNight"),
    (["cloud", "wet"], "WetCloudyNoon"),
    (["cloud"], "CloudyNoon"),
    (["wet"], "HardRainNoon"),
    (["sun"], "ClearNoon"),
]


def extract_nusc_weather(description: str) -> str:
    """Map a nuScenes scene description string to a CARLA weather preset name."""
    text = description.lower()
    for keywords, preset in _WEATHER_RULES:
        if all(re.search(rf"\b{k}\b", text) for k in keywords):
            return preset
    return "ClearNoon"


def stage1_determine_weather(meta: NuscMeta, target_sample: str, out_dir) -> dict:
    """Look up the scene description, match to a weather preset via keyword rules, return CARLA weather params."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Weather Determination")
    logger.info("=" * 60)

    s = next(s for s in meta.samples if s["token"] == target_sample)
    scene = next(sc for sc in meta.scenes if sc["token"] == s["scene_token"])
    log = next(l for l in meta.logs if l["token"] == scene["log_token"])

    scene_info = {
        "scene_name": scene.get("name", ""),
        "description": scene.get("description", ""),
        "location": log.get("location", ""),
        "nbr_samples": scene.get("nbr_samples", 0),
        "sample_token": target_sample,
    }

    preset = extract_nusc_weather(scene.get("description", ""))
    scene_info["weather_preset"] = preset
    log_json(scene_info, out_dir / "scene_info.json", "scene info")
    logger.info("Matched description to CARLA preset: %s", preset)

    weather_params = dict(WEATHER_PRESETS[preset])
    weather_params["preset"] = preset
    log_json(weather_params, out_dir / "weather_params.json", "weather params")
    logger.info("Weather parameters: %s", weather_params)
    return weather_params


def stage2_determine_town(meta: NuscMeta, target_sample: str, out_dir) -> dict:
    """Select the CARLA town based on nuScenes scene location (Boston→Town05, Singapore→Town03)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Town Selection")
    logger.info("=" * 60)

    s = next(s for s in meta.samples if s["token"] == target_sample)
    scene = next(sc for sc in meta.scenes if sc["token"] == s["scene_token"])
    log = next(l for l in meta.logs if l["token"] == scene["log_token"])
    location = log["location"]
    logger.info("Scene location: %s", location)

    location_lower = location.lower()
    if "boston" in location_lower:
        town = "Town05"
    elif "singapore" in location_lower:
        town = "Town03"
    else:
        logger.warning("Unknown location '%s', defaulting to Town03", location)
        town = "Town03"

    logger.info("Selected town: %s (for location: %s)", town, location)

    result = {
        "town": town,
        "location": location,
        "scene_name": scene.get("name", ""),
        "scene_description": scene.get("description", ""),
    }
    log_json(result, out_dir / "town.json", "town selection")
    return result
