"""Command-line interface for dataset conversion and validation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .adapters.nuplan import convert_nuplan_database
from .adapters.waymo import convert_waymo_tfrecord
from .io import read_scenario, write_scenario
from .models import CanonicalScenario
from .runtime import (
    RuntimeConfig,
    compile_runtime_scenario,
    validate_runtime_directory,
    write_runtime_scenario,
)
from .validation import validate_scenario
from .visualize import (
    animate_nuplan_comparison,
    animate_nuplan_conversion,
    visualize_nuplan_conversion,
)


@dataclass
class BatchResult:
    succeeded: int = 0
    warned: int = 0
    failed: int = 0


def _input_files(path: Path, predicate: Callable[[Path], bool]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"input does not exist: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and predicate(candidate))
    if not files:
        raise ValueError(f"no supported input files found in: {path}")
    return files


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_batch(
    scenarios: Iterable[CanonicalScenario],
    output_dir: Path,
    source_stem: str,
    result: BatchResult,
) -> None:
    for scenario in scenarios:
        warnings = validate_scenario(scenario)
        filename = f"{_safe_name(source_stem)}_{_safe_name(scenario.source.source_scenario_id)}.json"
        write_scenario(scenario, output_dir / filename)
        result.succeeded += 1
        if warnings:
            result.warned += 1
            print(f"[warning] {filename}: {'; '.join(warnings)}", file=sys.stderr)


def _summary(result: BatchResult) -> None:
    print(f"summary: succeeded={result.succeeded} warned={result.warned} failed={result.failed}")


def _convert_nuplan(args: argparse.Namespace) -> int:
    files = _input_files(args.input, lambda path: path.suffix == ".db")
    result = BatchResult()
    for path in files:
        try:
            scenarios = convert_nuplan_database(path, args.maps_root, args.scene_id)
            _write_batch(scenarios, args.output, path.stem, result)
        except Exception as error:  # Keep a large batch running and report each bad DB.
            result.failed += 1
            print(f"[error] {path}: {error}", file=sys.stderr)
    _summary(result)
    return 1 if result.failed else 0


def _convert_waymo(args: argparse.Namespace) -> int:
    files = _input_files(args.input, lambda path: "tfrecord" in path.name)
    result = BatchResult()
    for path in files:
        try:
            _write_batch(convert_waymo_tfrecord(path), args.output, path.name, result)
        except Exception as error:
            result.failed += 1
            print(f"[error] {path}: {error}", file=sys.stderr)
    _summary(result)
    return 1 if result.failed else 0


def _validate(args: argparse.Namespace) -> int:
    files = _input_files(args.input, lambda path: path.suffix == ".json")
    result = BatchResult()
    for path in files:
        try:
            warnings = validate_scenario(read_scenario(path))
            result.succeeded += 1
            if warnings:
                result.warned += 1
                print(f"[warning] {path}: {'; '.join(warnings)}", file=sys.stderr)
        except Exception as error:
            result.failed += 1
            print(f"[error] {path}: {error}", file=sys.stderr)
    _summary(result)
    return 1 if result.failed else 0


def _visualize_nuplan(args: argparse.Namespace) -> int:
    image, summary = visualize_nuplan_conversion(
        args.input, args.output, args.maps_root, args.scene_id
    )
    print(f"image: {image}")
    print(f"summary: {summary}")
    return 0


def _animate_nuplan(args: argparse.Namespace) -> int:
    animation = animate_nuplan_conversion(
        args.input, args.output, args.maps_root, args.scene_id, args.fps, args.stride
    )
    print(f"animation: {animation}")
    return 0


def _compare_nuplan(args: argparse.Namespace) -> int:
    before, after, comparison = animate_nuplan_comparison(
        args.input, args.output, args.maps_root, args.scene_id, args.fps, args.stride
    )
    print(f"before: {before}")
    print(f"after: {after}")
    print(f"comparison: {comparison}")
    return 0


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        max_agents=args.max_agents,
        history_steps=args.history_steps,
        max_future_steps=args.max_future_steps,
        max_map_features=args.max_map_features,
        max_map_points=args.max_map_points,
        max_map_edges=args.max_map_edges,
        max_traffic_lights=args.max_traffic_lights,
        max_route_features=args.max_route_features,
    )


def _compile_rl(args: argparse.Namespace) -> int:
    files = _input_files(args.input, lambda path: path.suffix == ".json" and path.name != "manifest.json")
    result = BatchResult()
    config = _runtime_config(args)
    for path in files:
        try:
            runtime = compile_runtime_scenario(read_scenario(path), config)
            runtime_dir = args.output / _safe_name(path.stem)
            write_runtime_scenario(runtime, runtime_dir)
            warnings = validate_runtime_directory(runtime_dir)
            result.succeeded += 1
            if warnings:
                result.warned += 1
                print(f"[warning] {path.name}: {'; '.join(warnings)}", file=sys.stderr)
            print(f"[ok] {path.name} -> {runtime_dir}")
        except Exception as error:
            result.failed += 1
            print(f"[error] {path}: {error}", file=sys.stderr)
    _summary(result)
    return 1 if result.failed else 0


def _validate_rl(args: argparse.Namespace) -> int:
    directories = [args.input] if (args.input / "manifest.json").is_file() else sorted(
        path.parent for path in args.input.rglob("manifest.json")
    )
    if not directories:
        raise ValueError(f"no runtime manifest found in: {args.input}")
    result = BatchResult()
    for directory in directories:
        try:
            warnings = validate_runtime_directory(directory)
            result.succeeded += 1
            if warnings:
                result.warned += 1
                print(f"[warning] {directory}: {'; '.join(warnings)}", file=sys.stderr)
        except Exception as error:
            result.failed += 1
            print(f"[error] {directory}: {error}", file=sys.stderr)
    _summary(result)
    return 1 if result.failed else 0


def _add_runtime_capacity_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = RuntimeConfig()
    parser.add_argument("--max-agents", type=int, default=defaults.max_agents)
    parser.add_argument("--history-steps", type=int, default=defaults.history_steps)
    parser.add_argument("--max-future-steps", type=int, default=defaults.max_future_steps)
    parser.add_argument("--max-map-features", type=int, default=defaults.max_map_features)
    parser.add_argument("--max-map-points", type=int, default=defaults.max_map_points)
    parser.add_argument("--max-map-edges", type=int, default=defaults.max_map_edges)
    parser.add_argument("--max-traffic-lights", type=int, default=defaults.max_traffic_lights)
    parser.add_argument("--max-route-features", type=int, default=defaults.max_route_features)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenario-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    nuplan = commands.add_parser("convert-nuplan", help="convert nuPlan SQLite scenes")
    nuplan.add_argument("--input", type=Path, required=True, help=".db file or directory")
    nuplan.add_argument("--output", type=Path, required=True)
    nuplan.add_argument("--maps-root", type=Path)
    nuplan.add_argument("--scene-id", help="optional scene token or scene name")
    nuplan.set_defaults(handler=_convert_nuplan)

    waymo = commands.add_parser("convert-waymo", help="convert Waymo Scenario TFRecords")
    waymo.add_argument("--input", type=Path, required=True, help="TFRecord file or directory")
    waymo.add_argument("--output", type=Path, required=True)
    waymo.set_defaults(handler=_convert_waymo)

    validate = commands.add_parser("validate", help="validate canonical JSON")
    validate.add_argument("--input", type=Path, required=True, help="JSON file or directory")
    validate.set_defaults(handler=_validate)

    visualize = commands.add_parser("visualize-nuplan", help="compare raw and canonical nuPlan trajectories")
    visualize.add_argument("--input", type=Path, required=True, help="nuPlan .db file")
    visualize.add_argument("--output", type=Path, required=True, help="output .png path")
    visualize.add_argument("--maps-root", type=Path)
    visualize.add_argument("--scene-id")
    visualize.set_defaults(handler=_visualize_nuplan)

    animate = commands.add_parser("animate-nuplan", help="render a top-down nuPlan trajectory replay")
    animate.add_argument("--input", type=Path, required=True, help="nuPlan .db file")
    animate.add_argument("--output", type=Path, required=True, help="output .gif path")
    animate.add_argument("--maps-root", type=Path)
    animate.add_argument("--scene-id")
    animate.add_argument("--fps", type=int, default=10)
    animate.add_argument("--stride", type=int, default=1, help="render every Nth scenario frame")
    animate.set_defaults(handler=_animate_nuplan)

    compare = commands.add_parser("compare-nuplan", help="render synchronized before/after GIFs")
    compare.add_argument("--input", type=Path, required=True, help="nuPlan .db file")
    compare.add_argument("--output", type=Path, required=True, help="combined output .gif path")
    compare.add_argument("--maps-root", type=Path)
    compare.add_argument("--scene-id")
    compare.add_argument("--fps", type=int, default=10)
    compare.add_argument("--stride", type=int, default=1)
    compare.set_defaults(handler=_compare_nuplan)

    compile_rl = commands.add_parser("compile-rl", help="compile canonical JSON into RL tensors")
    compile_rl.add_argument("--input", type=Path, required=True, help="canonical JSON or directory")
    compile_rl.add_argument("--output", type=Path, required=True, help="runtime output directory")
    _add_runtime_capacity_arguments(compile_rl)
    compile_rl.set_defaults(handler=_compile_rl)

    validate_rl = commands.add_parser("validate-rl", help="validate RL runtime tensor directories")
    validate_rl.add_argument("--input", type=Path, required=True, help="runtime scene or parent directory")
    validate_rl.set_defaults(handler=_validate_rl)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
