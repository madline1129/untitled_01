import time
from pathlib import Path
import numpy as np
from real2sim.core import NuscMeta, log_json, setup_logging
from real2sim.stage4_spawn import EGO_BLUEPRINT, measure_blueprint_dims, stage4_spawn_actors


def _run_select_ego_bp(carla_world, out_dir: Path, logger):
    """Measure all CARLA vehicle blueprints and rank by L2 distance to
    the nuScenes ego (Renault Zoe) dimensions."""

    NUSC_EGO_DIMS = {"length": 4.085, "width": 1.730, "height": 1.562}
    nusc_arr = np.array([NUSC_EGO_DIMS["length"],
                         NUSC_EGO_DIMS["width"],
                         NUSC_EGO_DIMS["height"]], dtype=np.float32)

    logger.info("=" * 60)
    logger.info("SELECT EGO BLUEPRINT ANALYSIS")
    logger.info("nuScenes ego: Renault Zoe  L=%.3f  W=%.3f  H=%.3f",
                NUSC_EGO_DIMS["length"], NUSC_EGO_DIMS["width"], NUSC_EGO_DIMS["height"])
    logger.info("=" * 60)

    bp_lib = carla_world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                   if bp.has_attribute("number_of_wheels")]
    logger.info("Found %d vehicle blueprints with wheels", len(vehicle_bps))

    results = []
    failed = []
    t0 = time.time()

    for i, bp in enumerate(vehicle_bps):
        bp_id = bp.id
        dims = measure_blueprint_dims(bp, carla_world)
        if dims is None:
            failed.append(bp_id)
            logger.warning("  [%3d/%d] %-45s FAILED to spawn", i + 1, len(vehicle_bps), bp_id)
            continue
        length, width, height = dims
        dist = float(np.sqrt((length - nusc_arr[0])**2 +
                             (width - nusc_arr[1])**2 +
                             (height - nusc_arr[2])**2))
        results.append({
            "blueprint": bp_id, "length": length, "width": width, "height": height,
            "l2_distance": round(dist, 4),
        })
        if (i + 1) % 10 == 0:
            logger.info("  [%3d/%d] measured %d (%.1fs)",
                        i + 1, len(vehicle_bps), len(results), time.time() - t0)

    results.sort(key=lambda r: r["l2_distance"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    out_data = {
        "nusc_ego_dimensions_m": NUSC_EGO_DIMS,
        "total_blueprints": len(vehicle_bps),
        "measured": len(results),
        "failed_spawn": len(failed),
        "failed_blueprints": failed,
        "results": results,
    }
    log_json(out_data, out_dir / "ego_blueprint_analysis.json",
             "ego blueprint analysis")

    logger.info("")
    logger.info("Top 20 closest blueprints to nuScenes ego:")
    logger.info("%4s  %8s  %7s  %6s  %7s  Blueprint", "Rank", "L2", "Length", "Width", "Height")
    logger.info("-" * 80)
    for r in results[:20]:
        logger.info("%4d  %8.4f  %7.2f  %6.2f  %7.2f  %s",
                    r["rank"], r["l2_distance"],
                    r["length"], r["width"], r["height"],
                    r["blueprint"])

    logger.info("")
    logger.info("Recommended replacements for EGO_BLUEPRINT:")
    logger.info("  Current: %s", EGO_BLUEPRINT)
    if results:
        best = results[0]
        logger.info("  Best:    %s  (L2=%.4f, L=%.2f W=%.2f H=%.2f)",
                    best["blueprint"], best["l2_distance"],
                    best["length"], best["width"], best["height"])
        cars = [r for r in results if r["blueprint"].startswith("vehicle.car.")]
        if cars:
            bc = cars[0]
            logger.info("  Best car: %s  (L2=%.4f, L=%.2f W=%.2f H=%.2f)",
                        bc["blueprint"], bc["l2_distance"],
                        bc["length"], bc["width"], bc["height"])
    logger.info("Done.")
