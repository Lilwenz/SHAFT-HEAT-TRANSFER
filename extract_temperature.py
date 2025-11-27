"""Utility for exporting nodal temperatures from an Abaqus ODB file.

This script must be executed with the Abaqus Python interpreter so that the
`odbAccess` module is available, for example:

    abaqus python extract_temperature.py --odb ..\shaft\Job-4.odb \
        --step HeatStep --node-sets SENSORS-N ALL-N
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Abaqus-specific modules are only available inside its Python runtime
    from odbAccess import openOdb
    from abaqusConstants import NODAL
except ImportError as exc:  # pragma: no cover - will not run outside Abaqus
    raise SystemExit(
        "This script must be executed with 'abaqus python'. Regular CPython "
        "does not ship the odbAccess module."
    ) from exc

Coordinates = Tuple[float, float, float]


def _build_coordinate_cache(instances: Sequence) -> Dict[str, Dict[int, Coordinates]]:
    """Cache node coordinates per instance so we do not loop over nodes repeatedly."""
    cache: Dict[str, Dict[int, Coordinates]] = {}
    for instance in instances:
        coord_map: Dict[int, Coordinates] = {}
        for node in instance.nodes:
            coord_map[node.label] = node.coordinates
        cache[instance.name] = coord_map
    return cache


def _resolve_node_sets(root_assembly, requested_sets: Sequence[str]):
    """Return Abaqus node set objects ensuring all names exist."""
    resolved = []
    for set_name in requested_sets:
        if set_name not in root_assembly.nodeSets:
            available = ", ".join(sorted(root_assembly.nodeSets.keys()))
            raise SystemExit(
                f"Node set '{set_name}' not found in the ODB. Available sets: {available}"
            )
        resolved.append(root_assembly.nodeSets[set_name])
    return resolved


def _open_writer(path: str, header: Sequence[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = open(path, "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(header)
    return handle, writer


def export_temperatures(
    odb_path: str,
    output_csv: str,
    step_name: str,
    node_sets: Optional[Sequence[str]] = None,
    field_variable: str = "NT11",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    frame_stride: int = 1,
    split_by_node_set: bool = False,
) -> None:
    """Export (time, node label, coordinates, temperature) rows to CSV."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    odb = openOdb(path=odb_path, readOnly=True)
    root_assembly = odb.rootAssembly

    if step_name not in odb.steps:
        available = ", ".join(sorted(odb.steps.keys()))
        raise SystemExit(
            f"Step '{step_name}' not found in ODB '{odb_path}'. Available steps: {available}"
        )
    step = odb.steps[step_name]

    regions = None
    if node_sets:
        regions = _resolve_node_sets(root_assembly, node_sets)

    coord_cache = _build_coordinate_cache(root_assembly.instances.values())

    header = ["frame_index", "step_time", "node_set", "instance", "node_label", "x", "y", "z", field_variable]

    writer_handles: Dict[str, Tuple] = {}
    try:
        if split_by_node_set:
            base, ext = os.path.splitext(output_csv)
            if not ext:
                ext = ".csv"

            def get_writer(set_name: str):
                safe_name = set_name.replace(os.sep, "_").replace(" ", "_")
                path = f"{base}__{safe_name}{ext}"
                if set_name not in writer_handles:
                    writer_handles[set_name] = _open_writer(path, header)
                return writer_handles[set_name][1]

        else:
            handle, base_writer = _open_writer(output_csv, header)
            writer_handles["__all__"] = (handle, base_writer)

            def get_writer(_set_name: str):
                return base_writer

        for frame_index, frame in enumerate(step.frames):
            if frame_index % frame_stride:
                continue
            current_time = frame.frameValue
            if start_time is not None and current_time < start_time:
                continue
            if end_time is not None and current_time > end_time:
                break

            if field_variable not in frame.fieldOutputs:
                raise SystemExit(
                    f"Field '{field_variable}' not stored in frame {frame_index} of step '{step_name}'."
                )
            field = frame.fieldOutputs[field_variable]

            if regions:
                region_iter = (
                    (node_set.name, field.getSubset(region=node_set, position=NODAL))
                    for node_set in regions
                )
            else:
                region_iter = (("ALL", field.getSubset(position=NODAL)),)

            for region_name, subset in region_iter:
                for value in subset.values:
                    instance_name = value.instance.name
                    coords = coord_cache[instance_name][value.nodeLabel]
                    writer = get_writer(region_name)
                    writer.writerow(
                        [
                            frame_index,
                            current_time,
                            region_name,
                            instance_name,
                            value.nodeLabel,
                            coords[0],
                            coords[1],
                            coords[2],
                            value.data,
                        ]
                    )
    finally:
        for handle, _ in writer_handles.values():
            handle.close()
        odb.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Abaqus nodal temperatures to CSV.")
    parser.add_argument(
        "--odb",
        default=os.path.join("..", "shaft", "Job-4.odb"),
        help="Path to the Job-4 ODB file (default: ../shaft/Job-4.odb)",
    )
    parser.add_argument(
        "--step",
        default="HeatStep",
        help="Name of the analysis step to read (default: HeatStep)",
    )
    parser.add_argument(
        "--node-sets",
        nargs="*",
        default=None,
        help="Optional list of node set names. If omitted, export all nodes.",
    )
    parser.add_argument(
        "--field",
        default="NT11",
        help="Scalar field variable to export (default: NT11 for temperature).",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Skip frames earlier than this step time (seconds).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Stop exporting once step time exceeds this value (seconds).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Export every Nth frame to reduce file size (default: 1, export all frames).",
    )
    parser.add_argument(
        "--split-node-sets",
        action="store_true",
        help="Write one CSV per node set using the --output path as the base name.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("..", "shaft-heat-transfer", "job4_temperatures.csv"),
        help="Destination CSV path (default: ../shaft-heat-transfer/job4_temperatures.csv)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    export_temperatures(
        odb_path=args.odb,
        output_csv=args.output,
        step_name=args.step,
        node_sets=args.node_sets,
        field_variable=args.field,
        start_time=args.start_time,
        end_time=args.end_time,
        frame_stride=args.frame_stride,
        split_by_node_set=args.split_node_sets,
    )


if __name__ == "__main__":
    main()
