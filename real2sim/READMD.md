# real2sim — nuScenes → CARLA Scene Reconstruction

Reconstruct a nuScenes autonomous-driving scene inside the CARLA simulator. Given a single key-frame sample (6 camera images, 3D annotations, ego trajectory), the pipeline determines weather, selects the matching CARLA town, finds the correct ego spawn point via rotation-invariant map matching (LPIPS), transforms all actors into CARLA coordinates, and renders multi-view + BEV images with per-view similarity metrics (SSIM/PSNR/LPIPS) against ground truth.

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `stage12_weather_town.py` | Keyword-match scene description → CARLA weather preset (9 presets) |
| 2 | `stage12_weather_town.py` | Map location → CARLA town (Boston→Town05, Singapore→Town03) |
| 3 | `stage3_local_map.py` | Rotation-invariant local-map matching via LPIPS on drivable-area renders |
| 4 | `stage4_spawn.py` | 3-step coordinate transform + actor spawning (vehicles, pedestrians, static) |
| 5 | `stage5_capture.py` | Camera mounting (6 RGB + 1 BEV), rendering, similarity metrics, snapshot |

## Quick Start

```bash
# Basic run
python real2sim/pipeline.py --scene-name scene-0061

# List available scenes
python real2sim/pipeline.py --list-scenes --loc singapore

# Render nuScenes video only (no CARLA)
python real2sim/pipeline.py --nusc-video

# Resume from previous run
python real2sim/pipeline.py --resume-from output/real2sim/20250530_143000
```

## Prerequisites

- **CARLA** 0.9.13+ — set `CARLA_ROOT` in `.env` at repo root
- **nuScenes** dataset — minimally `v1.0-trainval` metadata + camera images
- Python packages: `pip install -r ../requirements.txt && pip install carla nuscenes-devkit lpips pyquaternion python-dotenv scikit-image`

## Output

```
output/real2sim/TIMESTAMP/
├── config.json, target_sample.json, pipeline.log, timing.json
├── stage1_weather/   ─ scene_info.json, weather_params.json
├── stage2_town/      ─ town.json
├── stage3_map/       ─ spawn_point.json, geo_candidates.json, top_maps/
├── stage4_spawn/     ─ actor_spawn_log.json, actor_velocities.json
└── stage5_capture/   ─ camera_calibration.json, similarity_metrics.json,
                        carla_snapshot.json, comparison_montage.png,
                        FRONT_carla.jpg, ..., BEV_carla.jpg
```

## Key Files

| File | Role |
|------|------|
| `pipeline.py` | Main entry point and orchestrator |
| `core.py` | Shared utilities: `NuscMeta` data loader, coordinate transforms, CARLA connection |
| `stage3_local_map.py` | LPIPS-based rotation-invariant map matching |
| `stage4_spawn.py` | nuScenes→CARLA coord transform + blueprint mapping + actor spawning |
| `stage5_capture.py` | Camera mounting + rendering + evaluation |
| `verify_snapshot.py` | Post-pipeline snapshot save/load fidelity check |
| `nusc-carla-mapping.md` | nuScenes category → CARLA blueprint reference |
| `carla-kb.md` | CARLA knowledge base (towns, blueprints, sensors) |
