"""Helpers for Stage-3 hierarchical spawn selection.

Pure ranking utilities used by real2sim/pipeline.py.
"""


def compute_spawn_success_rate(spawn_result):
    total = int(spawn_result.get("total_annotations", 0))
    skipped = int(spawn_result.get("skipped", 0))
    spawned = int(spawn_result.get("spawned", 0))
    effective = total - skipped
    if effective <= 0:
        return 0.0
    return float(spawned) / float(effective)


def compute_mean_ssim(sim_results):
    values = []
    for view, metrics in sim_results.items():
        if str(view).startswith("_"):
            continue
        if not isinstance(metrics, dict):
            continue
        score = metrics.get("ssim")
        if score is None:
            continue
        values.append(float(score))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def select_top_q_by_spawn(candidates, q):
    ranked = sorted(candidates, key=lambda c: c.get("spawn_rate", 0.0), reverse=True)
    return ranked[: max(0, int(q))]


def select_best_by_ssim(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.get("ssim_score", 0.0))
