import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

os.makedirs("output", exist_ok=True)

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

image_path = "womanleadinghorse.png"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not load {image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ---- Tunable parameters ----
params = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "min_mask_region_area": 500,
}

current_masks = []  # list of dicts from SAM, each with a "segmentation" key

def generate_masks():
    global current_masks
    print("Generating masks with:", params)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=params["points_per_side"],
        pred_iou_thresh=params["pred_iou_thresh"],
        stability_score_thresh=params["stability_score_thresh"],
        min_mask_region_area=params["min_mask_region_area"],
    )
    current_masks = generator.generate(image_rgb)
    # sort largest -> smallest as a simple background-to-foreground proxy
    current_masks.sort(key=lambda m: m["area"], reverse=True)
    print(f"Found {len(current_masks)} masks.")

def show_preview():
    display = image_bgr.copy()
    overlay = display.copy()
    rng = np.random.default_rng(42)  # fixed seed so colors stay consistent between redraws
    for m in current_masks:
        color = rng.integers(0, 255, size=3).tolist()
        overlay[m["segmentation"]] = color
    display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

    hud_lines = [
        f"points_per_side={params['points_per_side']} [q/a]",
        f"pred_iou_thresh={params['pred_iou_thresh']:.2f} [w/s]",
        f"stability_score_thresh={params['stability_score_thresh']:.2f} [e/d]",
        f"min_mask_region_area={params['min_mask_region_area']} [r/f]",
        f"masks found: {len(current_masks)}",
        "[SPACE]=regenerate  [x]=export & quit",
    ]
    for i, line in enumerate(hud_lines):
        y = 20 + i * 20
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SAM Auto Layer Picker", display)

cv2.namedWindow("SAM Auto Layer Picker")
generate_masks()
show_preview()

print("Adjust parameters, press [SPACE] to regenerate, [x] to export all layers + background.")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        params["points_per_side"] = max(4, params["points_per_side"] - 4)
    elif key == ord('a'):
        params["points_per_side"] += 4
    elif key == ord('w'):
        params["pred_iou_thresh"] = max(0.0, round(params["pred_iou_thresh"] - 0.02, 2))
    elif key == ord('s'):
        params["pred_iou_thresh"] = min(1.0, round(params["pred_iou_thresh"] + 0.02, 2))
    elif key == ord('e'):
        params["stability_score_thresh"] = max(0.0, round(params["stability_score_thresh"] - 0.02, 2))
    elif key == ord('d'):
        params["stability_score_thresh"] = min(1.0, round(params["stability_score_thresh"] + 0.02, 2))
    elif key == ord('r'):
        params["min_mask_region_area"] = max(0, params["min_mask_region_area"] - 100)
    elif key == ord('f'):
        params["min_mask_region_area"] += 100

    if key in (ord('q'), ord('a'), ord('w'), ord('s'), ord('e'), ord('d'), ord('r'), ord('f')):
        show_preview()  # update HUD text immediately, mask overlay updates after regenerate

    elif key == ord(' '):
        generate_masks()
        show_preview()

    elif key == ord('x'):
        break

cv2.destroyAllWindows()

# ---- Export ----
covered = np.zeros(image_rgb.shape[:2], dtype=bool)

for i, m in enumerate(current_masks):
    mask = m["segmentation"]
    layer_rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    layer_rgba[..., :3] = image_rgb
    layer_rgba[..., 3] = (mask * 255).astype(np.uint8)
    out_path = f"output/layer_{i:02d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(layer_rgba, cv2.COLOR_RGBA2BGRA))
    covered |= mask
    print(f"Saved {out_path} (area={m['area']})")

# whatever no mask claimed = background
background_rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
background_rgba[..., :3] = image_rgb
background_rgba[..., 3] = np.where(covered, 0, 255).astype(np.uint8)
cv2.imwrite("output/background.png", cv2.cvtColor(background_rgba, cv2.COLOR_RGBA2BGRA))
print("Saved output/background.png")