import os
import re
import glob
import numpy as np
import cv2
import pyvista as pv

# ---- Configuration ----
LAYER_DIR = "iwamura-output"       # folder containing layer_00.png, layer_01.png, ..., background.png
LAYER_SPACING = .5        # distance between adjacent sheets along the depth (z) axis
PLANE_HEIGHT = 2.0         # world-space height of each plane; width is derived from image aspect ratio


def natural_sort_key(path):
    """Sort layer_00, layer_01, ... layer_10 in numeric order, and push background.png to the end."""
    name = os.path.basename(path)
    if name == "background.png":
        return (1, 0)
    match = re.search(r"(\d+)", name)
    num = int(match.group(1)) if match else 0
    return (0, num)


def load_layer_files(layer_dir):
    files = glob.glob(os.path.join(layer_dir, "layer_*.png"))
    bg = os.path.join(layer_dir, "background.png")
    if os.path.exists(bg):
        files.append(bg)
    files.sort(key=natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No layer_*.png or background.png files found in '{layer_dir}'."
        )
    return files


def load_rgba(path):
    """Load a png as full-resolution RGBA (uint8), adding an opaque alpha channel if missing."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")

    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return np.ascontiguousarray(img_rgba)


def main():
    files = load_layer_files(LAYER_DIR)
    print(f"Found {len(files)} layers:")
    for f in files:
        print(f"  {f}")

    plotter = pv.Plotter()
    # Correctly sorts overlapping transparent surfaces from any viewing angle
    plotter.enable_depth_peeling(number_of_peels=len(files) + 2, occlusion_ratio=0.0)

    for i, path in enumerate(files):
        img_rgba = load_rgba(path)
        h, w = img_rgba.shape[:2]
        aspect = w / h
        plane_w = PLANE_HEIGHT * aspect
        z = -i * LAYER_SPACING  # foreground (layer_00) at z=0, subsequent layers recede

        plane = pv.Plane(
            center=(0, 0, z),
            direction=(0, 0, 1),
            i_size=plane_w,
            j_size=PLANE_HEIGHT,
            i_resolution=1,
            j_resolution=1,
        )
        plane.texture_map_to_plane(inplace=True)

        texture = pv.Texture(img_rgba)

        plotter.add_mesh(
            plane,
            texture=texture,
            opacity=1.0,   # per-pixel transparency comes from the image's own alpha channel
        )
        print(f"Placed {os.path.basename(path)} at z={z:.2f}  ({w}x{h})")

    plotter.add_axes()
    # plotter.show_grid()
    plotter.set_background("black")

    # ---- Keyboard zoom controls ----
    zoom_step = 1.1  # >1 = zoom in per press, camera.zoom(<1) = zoom out

    def zoom_in():
        plotter.camera.zoom(zoom_step)
        plotter.render()

    def zoom_out():
        plotter.camera.zoom(1 / zoom_step)
        plotter.render()

    plotter.add_key_event("i", zoom_in)   # press 'i' to zoom in
    plotter.add_key_event("o", zoom_out)  # press 'o' to zoom out


    # starting view
    plotter.view_xy()
    plotter.reset_camera()
    plotter.enable_parallel_projection()
    plotter.export_html(f"{LAYER_DIR}/scene.html")

    print("Controls: click-drag = rotate, scroll = zoom, [i] = zoom in, [o] = zoom out")
    plotter.show(title="Layer stack - click and drag to rotate, scroll or i/o to zoom")
    # plotter.add_background_image('code.png')


if __name__ == "__main__":
    main()