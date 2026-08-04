#!/usr/bin/env python3
"""
pipeline.py — Unified real2sim pipeline.

Reconstructs a nuScenes sample (6 cameras, object annotations, local map)
in CARLA across geometry-first stages:
  1. Weather determination (lookup table from scene description)
  2. Town selection (Boston→Town05, Singapore→Town03)
  3. Rotation-invariant map-map matching → ego spawn point
  4. Spawn annotation actors
  5. Spawn ego, mount 6 cameras, capture, save snapshot
  6. (Legacy) random search stage is deprecated

Stages 1-2 require only nuScenes metadata (no CARLA).
Stages 3-5 require a running CARLA server.

Usage:
  python real2sim/pipeline.py [--nusc-root PATH] [--sample-id TOKEN] [--scene-name NAME]
                              [--port PORT] [--top-k K] [--top-q Q] [--resume-from DIR]
                              [--list-scenes]
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import socket

import cv2
import numpy as np
from dotenv import load_dotenv
import carla
from PIL import Image
from real2sim.core import (
    NuscMeta,
    REPO_ROOT,
    compute_carla_trajectory,
    ensure_carla_running,
    filter_barely_visible_annotations,
    get_robust_world,
    log_json,
    setup_logging,
)
from real2sim.sample_selection import find_random_sample, find_sample_by_scene_name
from real2sim.stage12_weather_town import stage1_determine_weather, stage2_determine_town
from real2sim.stage3_local_map import (
    TOP_K,
    destroy_all_actors,
    stage3_local_map,
)
from real2sim.stage4_spawn import (
    IMAGE_H,
    IMAGE_W,
    measure_blueprint_dims,
    nusc_scene_to_carla,
    stage4_spawn_actors,
)
from real2sim.stage5_capture import (
    CAMERA_CHANNELS,
    CAMERA_VIEWS,
    _build_sample_cameras,
    _render_nusc_bev,
    stage5_capture,
)
from real2sim.select_ego_bp import _run_select_ego_bp

# ── VLM annotation (debug) ──────────────────────────────────────────────────

VLM_ANNOTATE_TEMPLATE = (
    'You are an expert driving-scene annotator.\n'
    '\n'
    'Examples:\n'
    '---\n'
    'Video: a white van ahead gradually slows down with brake lights on, '
    'then a pedestrian crosses from left to right in front of it.\n'
    'Description: In FRONT camera, the white van brakes with tail lights '
    'illuminated; a pedestrian walks across the road from left to right.\n'
    '---\n'
    'Video: a black sedan in the left lane accelerates and changes to the '
    'ego lane, then continues straight.\n'
    'Description: In FRONT camera, the black sedan in the left lane merges '
    'into the ego lane and accelerates.\n'
    '---\n'
    'Video: all vehicles move normally in their lanes, no pedestrians, '
    'no traffic light changes.\n'
    'Description: None\n'
    '---\n'
    'Now analyze this video:\n'
    "The video is from the FRONT camera of a moving vehicle E. "
    "Describe the changes of states of dynamic objects, "
    "like pedestrians, vehicles, traffic lights. "
    "DO NOT describe the movement of the vehicle E. "
    "DO NOT describe the static surrounding. "
    'If there is no such changes in the video, output "None".'
)


def _vlm_annotate(meta, target_sample, nusc_root, out_dir, num_frames,
                   endpoint, model, logger):
    """Send downsampled nuScenes FRONT frames to VLM and save annotation."""
    from sim2sim.vlm_client import VLMClient

    sample_by_token = {s["token"]: s for s in meta.samples}
    cal_to_sensor = {cs["token"]: cs["sensor_token"] for cs in meta.calibrated_sensors}
    sensor_to_channel = {s["token"]: s["channel"] for s in meta.sensors}
    cam_channels = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    vlm = VLMClient(endpoint=endpoint, model=model)
    tile_w, tile_h = IMAGE_W // 4, IMAGE_H // 4  # 400×225
    src_h, src_w = IMAGE_H, IMAGE_W

    sample_tok = target_sample
    future_paths = []
    n_cond = 3
    frame_idx = 0

    while sample_tok and frame_idx < num_frames:
        cams = {}
        for sd in meta.sample_data:
            if sd["sample_token"] != sample_tok or not sd.get("is_key_frame"):
                continue
            ch = sensor_to_channel.get(cal_to_sensor.get(sd["calibrated_sensor_token"], ""), "")
            if ch in cam_channels:
                cams[ch] = sd

        front_sd = cams.get("CAM_FRONT")
        if front_sd:
            path = nusc_root / front_sd["filename"]
            if path.exists():
                img = np.array(Image.open(path).convert("RGB"))
                small = np.array(Image.fromarray(img).resize((tile_w, tile_h)))
                if frame_idx >= n_cond:
                    tmp = Path(out_dir) / f"_vlm_frame_{frame_idx:04d}.jpg"
                    Image.fromarray(small).save(str(tmp), quality=85)
                    future_paths.append(str(tmp))

        s = sample_by_token.get(sample_tok)
        sample_tok = s["next"] if s else None
        frame_idx += 1

    if not future_paths:
        logger.warning("No future frames for VLM annotation")
        return

    total_bytes = sum(Path(p).stat().st_size for p in future_paths)
    total_pixels = len(future_paths) * tile_w * tile_h
    fps = 2.0
    logger.info(
        "VLM clip: frames=%d  fps=%.1f  duration=%.1fs  "
        "source=%dx%d  downsampled=%dx%d  "
        "raw_pixels=%d  encoded_bytes=%d  avg_frame_bytes=%d  compression=%.1fx",
        len(future_paths), fps, len(future_paths) / fps,
        src_w, src_h, tile_w, tile_h,
        total_pixels * 3, total_bytes, total_bytes // len(future_paths),
        (total_pixels * 3) / total_bytes if total_bytes else 0,
    )
    raw = vlm.describe(future_paths, prompt=VLM_ANNOTATE_TEMPLATE)
    is_none = raw.strip().lower().rstrip(".") == "none"
    annotation = {
        "world_prompt": None if is_none else raw,
        "raw": raw,
        "n_frames": len(future_paths),
        "vlm_endpoint": endpoint,
        "vlm_model": model,
    }
    log_json(annotation, Path(out_dir) / "vlm_annotation.json", "VLM world-prompt annotation")

    # Cleanup temp files
    for p in future_paths:
        Path(p).unlink(missing_ok=True)


# ── paths ───────────────────────────────────────────────────────────────────
DEFAULT_NUSC_ROOT = REPO_ROOT / "data" / "nuscenes"
DEFAULT_META_DIR = DEFAULT_NUSC_ROOT / "v1.0-trainval"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "real2sim"
logger = logging.getLogger("real2sim")


# ============================================================================
# VIDEO RENDERING (nuScenes only, no CARLA)
# ============================================================================


def _render_nusc_video(meta, target_sample, nusc_root, out_dir, num_frames, logger):
    """Walk the scene sample chain and render a 1x6 grid video + BEV video from nuScenes data."""
    from safebench.util.run_util import VideoWriter

    sample_by_token = {s["token"]: s for s in meta.samples}
    tile_w, tile_h = IMAGE_W // 4, IMAGE_H // 4
    col_bar_h = 20
    frame_list = []
    bev_frames = []

    sample_tok = target_sample
    for fi in range(num_frames):
        if sample_tok is None:
            logger.warning("Ran out of samples after %d frames (requested %d)", fi, num_frames)
            break

        cams = _build_sample_cameras(meta, sample_tok)
        if len(cams) < 6:
            logger.warning("Sample %s has only %d/6 cameras — stopping", sample_tok, len(cams))
            break

        # ── Camera grid frame ──
        tiles = []
        for ch in CAMERA_CHANNELS:
            sd = cams.get(ch)
            if sd is None:
                tile = np.full((IMAGE_H, IMAGE_W, 3), 64, dtype=np.uint8)
            else:
                path = nusc_root / sd["filename"]
                if path.exists():
                    tile = np.array(Image.open(path).convert("RGB"))
                else:
                    tile = np.full((IMAGE_H, IMAGE_W, 3), 64, dtype=np.uint8)
            tile = np.array(Image.fromarray(tile).resize((tile_w, tile_h)))
            tiles.append(tile)

        row = np.hstack(tiles)
        col_bar = np.ones((col_bar_h, row.shape[1], 3), dtype=np.uint8) * 30
        for i, view in enumerate(CAMERA_VIEWS):
            x = i * tile_w + 8
            cv2.putText(col_bar, view, (x, col_bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        frame = np.vstack([col_bar, row])
        frame_list.append(frame)

        # ── BEV frame ──
        bev = _render_nusc_bev(meta, sample_tok, nusc_root)
        label_h = 24
        bev_label = np.full((label_h, bev.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(bev_label, f"frame {fi}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        bev = np.vstack([bev_label, bev])
        bev_frames.append(bev)

        s = sample_by_token.get(sample_tok)
        sample_tok = s["next"] if s else None

    fps = 2.0
    cam_path = out_dir / "nusc_video.mp4"
    vw = VideoWriter(filename=str(cam_path), fps=fps)
    for f in frame_list:
        vw.add(f)
    vw.close()
    logger.info("Saved camera grid video: %s (%d frames, ~%.1f s)", cam_path, len(frame_list), len(frame_list) / fps)

    bev_path = out_dir / "nusc_bev_video.mp4"
    vw = VideoWriter(filename=str(bev_path), fps=fps)
    for f in bev_frames:
        vw.add(f)
    vw.close()
    logger.info("Saved BEV video: %s (%d frames, ~%.1f s)", bev_path, len(bev_frames), len(bev_frames) / fps)


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real2Sim pipeline: reconstruct nuScenes sample in CARLA")
    parser.add_argument("--nusc-root", type=Path, default=DEFAULT_NUSC_ROOT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", type=str, default=None,
                        help="Specific nuScenes sample token (default: first with 6 cams)")
    parser.add_argument("--scene-name", type=str, default=None,
                        help="nuScenes scene name (e.g. scene-0061); overrides --sample-id")
    parser.add_argument("--list-scenes", action="store_true",
                        help="List available scene names and exit")
    parser.add_argument("--port", type=int, default=2004,
                        help="CARLA server port")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume from a previous output directory (skips stages 1-3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for scene selection when --scene-name is not given")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Stage-3 geometry ranking: keep top-K candidates")

    parser.add_argument("--loc", type=str, default="boston",
                        help="Filter scenes by location (boston, singapore)")
    parser.add_argument("--match-colors", default=True, action=argparse.BooleanOptionalAction,
                        help="Extract vehicle colors from nuScenes camera images")
    parser.add_argument("--select-ego-bp", action="store_true",
                        help="Measure all vehicle blueprints and find the closest match "
                             "to nuScenes ego (Renault Zoe) dimensions, then exit")
    parser.add_argument("--map-sim-ncc", action="store_true",
                        help="Use per-pixel normalized cross-correlation (instead of LPIPS) "
                             "for local-map similarity scoring during spawn-point search")
    parser.add_argument("--refine-radius", type=float, default=5.0,
                        help="Neighbourhood refinement radius (metres) around the best "
                             "map-match waypoint. 0 disables. LPIPS is CNN-based (roughly "
                             "translation-invariant), but the coarse sampling grid misses "
                             "nearby positions — this fills that gap. (default: 5.0)")
    parser.add_argument("--nusc-video", action="store_true",
                        help="Render nuScenes camera video (1x6 grid) and exit; skips all CARLA stages")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of frames: for --nusc-video renders video; for the CARLA pipeline exports "
                             "multi-frame trajectory (includes the target sample)")
    parser.add_argument("--min-pixel-area", type=float, default=50.0,
                        help="Minimum visible pixel area for an annotation bbox projected across 6 camera "
                             "views. Annotations below this threshold are filtered out before spawning "
                             "(default: 50)")

    # VLM debug annotation
    parser.add_argument("--vlm-annotate", action="store_true",
                        help="Send downsampled nuScenes clip to VLM for world-prompt annotation")
    parser.add_argument("--vlm-endpoint", type=str, default="http://localhost:11434",
                        help="Ollama endpoint (default: http://localhost:11434)")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl:4b",
                        help="VLM model name (default: qwen3-vl:4b)")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()

    # ── Setup output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir == DEFAULT_OUTPUT_ROOT:
        out_dir = args.output_dir / f"{timestamp}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / "pipeline.log")
    logger.info("Real2Sim Pipeline starting")
    logger.info("Output directory: %s", out_dir)
    logger.info("nuScenes root: %s", args.nusc_root)

    # Save run config
    log_json(vars(args), out_dir / "config.json", "run config")

    # ── Validate ──
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    
    # ── Load location index ──
    loc_index = None
    allowed_scene_names = None
    if args.loc:
        loc_path = REPO_ROOT / "real2sim" / "nusc-loc.json"
        if not loc_path.exists():
            raise RuntimeError(
                f"Location index not found at {loc_path}. "
                f"Run `python real2sim/extract_nusc_loc.py` first."
            )
        with open(loc_path) as f:
            loc_index = json.load(f)
        loc_lower = args.loc.lower()
        if loc_lower not in loc_index:
            raise RuntimeError(
                f"Unknown location '{args.loc}'. "
                f"Available: {sorted(loc_index.keys())}"
            )
        allowed_scene_names = set(loc_index[loc_lower]["scene_names"])
        logger.info("Filtering to location '%s' (%d scenes)",
                     args.loc, len(allowed_scene_names))

    # ── Load nuScenes metadata ──
    meta = NuscMeta(args.meta_dir)

    # ── --list-scenes (no CARLA needed) ──
    if args.list_scenes:
        scenes = sorted(set(s["name"] for s in meta.scenes))
        if allowed_scene_names:
            scenes = [n for n in scenes if n in allowed_scene_names]
        for sn in scenes:
            sc = next(s for s in meta.scenes if s["name"] == sn)
            desc = sc.get("description", "")[:100].replace("\n", " ")
            print(f"  {sn:20s}  {desc}")
        return

    # ── Resolve target sample ──
    if args.resume_from:
        target_data = _load_json(args.resume_from / "target_sample.json")
        target_sample = target_data["sample_token"]
        weather_params = _load_json(args.resume_from / "stage1_weather" / "weather_params.json")
        stage2_result = _load_json(args.resume_from / "stage2_town" / "town.json")
        sample_cams = _build_sample_cameras(meta, target_sample)
        logger.info("Resuming from %s (sample %s)", args.resume_from, target_sample)
    elif args.scene_name:
        target_sample, sample_cams = find_sample_by_scene_name(meta, args.scene_name)
    elif args.sample_id:
        target_sample = args.sample_id
        sample_cams = _build_sample_cameras(meta, target_sample)
        cam_channels = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        missing = [ch for ch in cam_channels if ch not in sample_cams]
        if missing:
            logger.warning("Sample %s missing cameras: %s", target_sample, missing)
    else:
        target_sample, sample_cams = find_random_sample(
            meta, args.seed, allowed_scene_names=allowed_scene_names)

    logger.info("Target sample: %s", target_sample)
    sample_by_token = {s["token"]: s for s in meta.samples}
    scene_name = None
    for sc in meta.scenes:
        tok = sc["first_sample_token"]
        while tok:
            if tok == target_sample:
                scene_name = sc["name"]
                break
            s = sample_by_token.get(tok)
            tok = s.get("next") if s else None
        if scene_name:
            break
    log_json({"sample_token": target_sample, "scene_token": sample_cams.get("CAM_FRONT", {}).get("sample_token", ""), "scene_name": scene_name or "unknown"},
             out_dir / "target_sample.json", "target sample")

    # ── nuScenes Video (skip all CARLA stages) ──
    if args.nusc_video:
        _render_nusc_video(meta, target_sample, args.nusc_root, out_dir, args.num_frames, logger)
        return

    # ── Everything below requires CARLA ──
    ensure_carla_running(args.port, logger)

    # ── --select-ego-bp (standalone, no nuScenes metadata needed) ──
    if args.select_ego_bp:
        logger.info("Connecting to CARLA %s:%d for blueprint analysis ...", args.host, args.port)
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        try:
            _run_select_ego_bp(world, out_dir, logger)
        finally:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        return

    timing = {}
    _T = lambda: time.time()

    # ── Stage 1: Weather ──
    t0 = _T()
    if args.resume_from:
        weather_params = weather_params
        logger.info("SKIP Stage 1 (resume)")
    else:
        s1_out = out_dir / "stage1_weather"
        s1_out.mkdir(parents=True, exist_ok=True)
        weather_params = stage1_determine_weather(meta, target_sample, s1_out)
    timing["stage1_weather"] = round(_T() - t0, 2)
    logger.info("  stage1_weather: %.2f s", timing["stage1_weather"])

    # ── Stage 2: Town Selection ──
    t0 = _T()
    if args.resume_from:
        stage2_result = stage2_result
        logger.info("SKIP Stage 2 (resume)")
    else:
        s2_out = out_dir / "stage2_town"
        s2_out.mkdir(parents=True, exist_ok=True)
        stage2_result = stage2_determine_town(meta, target_sample, s2_out)
    timing["stage2_town"] = round(_T() - t0, 2)
    logger.info("  stage2_town: %.2f s", timing["stage2_town"])

    logger.info("Connecting to CARLA %s:%d (timeout=1000s)", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(1000.0)
    logger.info("CARLA connected: %s", client.get_server_version())

    try:
        town = stage2_result["town"]
        world = get_robust_world(client, town, clean_residuals=True)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        world.tick()

        # ── Vehicle dimension cache + pedestrian age-group cache ──
        bp_lib = world.get_blueprint_library()
        # _build_dim_cache(world, bp_lib)       # temporarily disabled
        # _build_pedestrian_cache(bp_lib)        # temporarily disabled

        # ── Vehicle color pre-computation (from nuScenes camera images) ──
        color_cache = {}
        # if args.match_colors:                   # temporarily disabled
        #     logger.info("Pre-computing vehicle colors from nuScenes images ...")
        #     color_cache = _precompute_vehicle_colors(meta, target_sample, args.nusc_root)

        # ── Stage 3: geometry ranking → top candidate ──
        t0 = _T()
        s3_out = out_dir / "stage3_map"
        s3_out.mkdir(parents=True, exist_ok=True)
        final_spawn, _ = stage3_local_map(
            meta=meta,
            target_sample=target_sample,
            nusc_root=args.nusc_root,
            carla_world=world,
            out_dir=s3_out,
            top_k=args.top_k,
        )
        timing["stage3_map_match"] = round(_T() - t0, 2)
        logger.info("  stage3_map_match: %.2f s", timing["stage3_map_match"])

        # ── Filter out barely visible annotations ──
        t0 = _T()
        exclude_tokens = filter_barely_visible_annotations(
            meta, target_sample, min_pixel_area=args.min_pixel_area,
        )
        if exclude_tokens:
            logger.info("Bbox visibility: %d annotations below %d-pixel threshold — will skip",
                         len(exclude_tokens), args.min_pixel_area)
        timing["filter_visibility"] = round(_T() - t0, 2)
        logger.info("  filter_visibility: %.2f s", timing["filter_visibility"])

        # ── Stage 4: spawn all actors (annotations + ego) ──
        t0 = _T()
        destroy_all_actors(client, world)
        s4_out = out_dir / "stage4_spawn"
        s4_out.mkdir(parents=True, exist_ok=True)
        _, ego_actor, vel_map = stage4_spawn_actors(
            meta,
            target_sample,
            final_spawn,
            world,
            carla_client=client,
            out_dir=s4_out,
            color_cache=color_cache,
            seed=args.seed,
            exclude_ann_tokens=exclude_tokens,
        )
        timing["stage4_spawn"] = round(_T() - t0, 2)
        logger.info("  stage4_spawn: %.2f s", timing["stage4_spawn"])

        # ── Stage 5: camera capture and snapshot ──
        t0 = _T()
        s5_out = out_dir / "stage5_capture"
        s5_out.mkdir(parents=True, exist_ok=True)
        stage5_capture(
            meta,
            target_sample,
            world,
            weather_params,
            args.nusc_root,
            s5_out,
            ego=ego_actor,
            vel_map=vel_map,
            best_rotation=final_spawn.get("best_rotation", 0),
        )
        timing["stage5_capture"] = round(_T() - t0, 2)
        logger.info("  stage5_capture: %.2f s", timing["stage5_capture"])
        # Copy target_sample.json into the snapshot dir for downstream pipelines
        shutil.copy2(str(out_dir / "target_sample.json"), str(s5_out / "target_sample.json"))

        # ── Multi-frame trajectory export (when --num-frames > 1) ──
        if args.num_frames > 1:
            t0 = _T()
            trajectory = compute_carla_trajectory(
                meta, target_sample, final_spawn, weather_params, town,
                args.num_frames,
            )
            log_json(trajectory, s5_out / "carla_trajectory.json", "multi-frame CARLA trajectory")
            timing["export_trajectory"] = round(_T() - t0, 2)
            logger.info("  export_trajectory: %.2f s", timing["export_trajectory"])

        # ── VLM annotation (debug) ──
        if args.vlm_annotate and args.num_frames > 1:
            t0 = _T()
            _vlm_annotate(meta, target_sample, args.nusc_root, s5_out,
                          args.num_frames, args.vlm_endpoint, args.vlm_model, logger)
            timing["vlm_annotate"] = round(_T() - t0, 2)
            logger.info("  vlm_annotate: %.2f s", timing["vlm_annotate"])

        logger.info("Final outputs saved to %s", out_dir)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        logger.info("Synchronous mode disabled — CARLA returned to async")

    timing["total"] = round(sum(v for v in timing.values()), 2)
    log_json(timing, out_dir / "timing.json", "per-stage timing")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("All outputs saved to: %s", out_dir)
    logger.info("Per-stage timing: %s", timing)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
