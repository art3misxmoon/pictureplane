#!/usr/bin/env python3
"""
fix_parallel_projection.py

Forces parallel (orthographic) projection in a PyVista `export_html()` file.

Why this is needed:
PyVista's export_html() embeds the scene as a base64-encoded zip containing
an index.json scene graph. The camera node(s) in that graph often have no
"parallelProjection" key at all, so vtk.js falls back to its default
(perspective) -- even if you set `plotter.camera.parallel_projection = True`
before exporting.

This script:
  1. Extracts the embedded base64 zip from the HTML.
  2. Unzips it and loads index.json.
  3. Finds the "primary" camera(s) -- the ones referenced by a render
     window's `extra.camera` field (this is how vtk.js knows which camera
     to use for interaction/reset) -- and patches them with
     parallelProjection = True and a computed parallelScale so the zoom
     level looks the same as the original perspective view.
  4. Re-zips, re-encodes, and writes a new HTML file.

Usage:
    python3 fix_parallel_projection.py input.html [output.html]
    python3 fix_parallel_projection.py input.html -o output.html --all-cameras
    python3 fix_parallel_projection.py input.html --scale 3.0

Options:
    --all-cameras   Patch every vtkCamera/vtkOpenGLCamera node found, not
                     just the ones referenced as a render window's primary
                     camera. Use this if the default doesn't visibly change
                     your scene (unusual scene graph shapes).
    --scale FLOAT   Override the computed parallelScale with a fixed value.
"""

import argparse
import base64
import io
import json
import math
import re
import sys
import zipfile


BASE64_PATTERN = re.compile(r'base64Str\s*=\s*"([^"]*)"')


def load_scene(html_text):
    match = BASE64_PATTERN.search(html_text)
    if not match:
        raise ValueError(
            "Could not find an embedded 'base64Str = \"...\"' scene blob in this HTML. "
            "Is this a PyVista export_html() file?"
        )
    b64 = match.group(1)
    zip_bytes = base64.b64decode(b64)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if "index.json" not in names:
            raise ValueError(f"Expected 'index.json' inside embedded zip, found: {names}")
        scene_json = json.loads(zf.read("index.json"))
    return b64, scene_json


def find_primary_camera_ids(node, ids=None):
    """Cameras referenced via a node's extra.camera field (render windows)."""
    if ids is None:
        ids = set()
    if isinstance(node, dict):
        extra = node.get("extra")
        if isinstance(extra, dict) and "camera" in extra:
            ids.add(extra["camera"])
        for v in node.values():
            find_primary_camera_ids(v, ids)
    elif isinstance(node, list):
        for v in node:
            find_primary_camera_ids(v, ids)
    return ids


def find_camera_nodes(node, target_ids=None, found=None):
    """All camera-type nodes, optionally filtered to target_ids."""
    if found is None:
        found = []
    if isinstance(node, dict):
        node_type = str(node.get("type", ""))
        if "Camera" in node_type:
            if target_ids is None or node.get("id") in target_ids:
                found.append(node)
        for v in node.values():
            find_camera_nodes(v, target_ids, found)
    elif isinstance(node, list):
        for v in node:
            find_camera_nodes(v, target_ids, found)
    return found


def patch_camera(node, scale_override=None):
    props = node.setdefault("properties", {})
    props["parallelProjection"] = True

    if scale_override is not None:
        props["parallelScale"] = scale_override
        return

    pos = props.get("position")
    fp = props.get("focalPoint")
    view_angle = props.get("viewAngle", 30)
    if pos and fp:
        dist = math.dist(pos, fp)
        props["parallelScale"] = dist * math.tan(math.radians(view_angle) / 2)
    else:
        # No position/focalPoint info to compute a sensible scale from;
        # leave any existing parallelScale, or default to vtk.js's own default (1).
        props.setdefault("parallelScale", 1)


def fix_html(html_text, all_cameras=False, scale_override=None):
    b64, scene_json = load_scene(html_text)

    if all_cameras:
        camera_nodes = find_camera_nodes(scene_json)
    else:
        primary_ids = find_primary_camera_ids(scene_json)
        camera_nodes = find_camera_nodes(scene_json, target_ids=primary_ids)
        if not camera_nodes:
            # Fallback: scene graph didn't expose extra.camera anywhere.
            camera_nodes = find_camera_nodes(scene_json)

    if not camera_nodes:
        raise ValueError("No camera nodes found in the scene graph -- nothing to patch.")

    for node in camera_nodes:
        patch_camera(node, scale_override)

    new_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(new_zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.json", json.dumps(scene_json))
    new_b64 = base64.b64encode(new_zip_bytes.getvalue()).decode("ascii")

    new_html = html_text.replace(b64, new_b64)
    return new_html, len(camera_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the PyVista-exported HTML file")
    parser.add_argument("output", nargs="?", help="Path to write the fixed HTML file")
    parser.add_argument("-o", "--output-flag", dest="output_flag", help="Alternate way to specify output path")
    parser.add_argument("--all-cameras", action="store_true", help="Patch every camera node, not just primary ones")
    parser.add_argument("--scale", type=float, default=None, help="Force a specific parallelScale value")
    args = parser.parse_args()

    output_path = args.output_flag or args.output
    if not output_path:
        if args.input.lower().endswith(".html"):
            output_path = args.input[: -len(".html")] + "_parallel.html"
        else:
            output_path = args.input + "_parallel.html"

    with open(args.input, "r", encoding="utf-8") as f:
        html_text = f.read()

    try:
        new_html, n_patched = fix_html(html_text, all_cameras=args.all_cameras, scale_override=args.scale)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_html)

    print(f"Patched {n_patched} camera node(s). Wrote: {output_path}")


if __name__ == "__main__":
    main()
