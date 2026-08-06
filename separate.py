import cv2
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamPredictor

# ---- Setup ----
output_dir = "hofmann-output"
image_path = "input/hofmann.png"
os.makedirs(output_dir, exist_ok=True)

# Pick the fastest available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# vit_h is the most accurate but slowest checkpoint.
# Swap to vit_b or vit_l if this is too slow on your hardware.
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not load {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # untouched original, used for final pixel output
working_image = image_rgb.copy()                        # what SAM actually looks at; gets inpainted as we go
predictor.set_image(working_image)

layers = []  # accepted layers, foreground -> background

# ---- Global claim tracking: prevents any pixel from ending up in more than one layer ----
claimed_mask = np.zeros(image_rgb.shape[:2], dtype=bool)

# ---- Per-object state (current click group; resets on 'm' merge or 'c' clear) ----
points = []
point_labels = []
all_logits = None
all_scores = None
candidate_idx = 0
threshold = 0.0

# ---- Per-layer accumulated state (pieces already merged via 'm'; resets after SPACE or 'c') ----
accumulated_mask = None


def recompute_mask():
    global all_logits, all_scores
    if not points:
        all_logits = None
        all_scores = None
        return
    coords = np.array(points)
    labels = np.array(point_labels)
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
        return_logits=True
    )
    all_logits = masks
    all_scores = scores


def get_current_mask():
    """Current object's mask, always excluding pixels already claimed by earlier layers."""
    if all_logits is None:
        return None
    m = all_logits[candidate_idx] > threshold
    return m & ~claimed_mask


def get_total_preview_mask():
    """Everything merged so far in this layer, unioned with whatever's currently live."""
    current = get_current_mask()
    if accumulated_mask is None:
        return current
    if current is None:
        return accumulated_mask
    return (accumulated_mask | current) & ~claimed_mask


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        point_labels.append(1)
        recompute_mask()
        show_preview()
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        point_labels.append(0)
        recompute_mask()
        show_preview()


def show_preview():
    display = cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR).copy()

    # chosen pixels in gray
    dim_overlay = display.copy()
    dim_overlay[claimed_mask] = (60, 60, 60)
    display = cv2.addWeighted(dim_overlay, 0.35, display, 0.65, 0)

    # accumulated (merged-in) pieces in orange
    if accumulated_mask is not None:
        overlay = display.copy()
        overlay[accumulated_mask] = (0, 140, 255)
        display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

    # current masks in red
    current = get_current_mask()
    if current is not None:
        overlay = display.copy()
        overlay[current] = (0, 0, 255)
        display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

    for (x, y), lbl in zip(points, point_labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(display, (x, y), 4, color, -1)

    score_txt = f"{all_scores[candidate_idx]:.3f}" if all_scores is not None else "n/a"
    hud_lines = [
        f"candidate {candidate_idx + 1}/3  score={score_txt}  threshold={threshold:+.2f}",
        f"accumulated pieces in this layer: {'yes' if accumulated_mask is not None else 'none yet'}",
        f"layers saved so far: {len(layers)}",
        "[m]=merge object into layer  [SPACE]=accept layer  [c]=clear layer  [q]=quit",
    ]
    for i, line in enumerate(hud_lines):
        y = 25 + i * 22
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("SAM Layer Picker", display)


def reset_current_object():
    global points, point_labels, all_logits, all_scores, candidate_idx, threshold
    points = []
    point_labels = []
    all_logits = None
    all_scores = None
    candidate_idx = 0
    threshold = 0.0


def reset_layer():
    global accumulated_mask
    reset_current_object()
    accumulated_mask = None


cv2.namedWindow("SAM Layer Picker")
cv2.setMouseCallback("SAM Layer Picker", on_click)
show_preview()

print("Left-click  = add foreground point   Right-click = add background point")
print("  [w/s] = cycle candidate mask     [a/d] = shrink/grow mask (threshold)")
print("  [m]   = merge current object into this layer, then click the next object")
print("  [c]   = clear everything for this layer, start over")
print("  [SPACE] = accept layer (accumulated + current), remove it, continue")
print("  [q]   = quit, save layers + remaining background")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('w'):
        candidate_idx = (candidate_idx - 1) % 3
        show_preview()
    elif key == ord('s'):
        candidate_idx = (candidate_idx + 1) % 3
        show_preview()
    elif key == ord('a'):
        threshold -= 0.5
        show_preview()
    elif key == ord('d'):
        threshold += 0.5
        show_preview()

    elif key == ord('m'):
        current = get_current_mask()
        if current is not None and current.any():
            accumulated_mask = current if accumulated_mask is None else (accumulated_mask | current)
            reset_current_object()
            show_preview()
            print("Merged object into layer. Click the next object to add, or press SPACE to finish this layer.")
        else:
            print("Nothing to merge yet - click an object first.")

    elif key == ord('c'):
        reset_layer()
        show_preview()

    elif key == ord(' '):
        mask = get_total_preview_mask()
        if mask is not None and mask.any():
            # Final layer pixels come from the ORIGINAL image, not the (possibly inpainted) working image
            layer_rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
            layer_rgba[..., :3] = image_rgb
            layer_rgba[..., 3] = (mask * 255).astype(np.uint8)
            layers.append(layer_rgba)

            # Lock these pixels out of every future mask
            claimed_mask |= mask

            working_image[mask] = 0
            predictor.set_image(working_image)

            reset_layer()
            show_preview()
            print(f"Layer {len(layers)} saved. {len(layers)} total so far.")
        else:
            print("No mask selected yet - click an object first.")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# ---- Save foreground layers ----
for i, layer in enumerate(layers):
    out_path = f"{output_dir}/layer_{i:02d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA))
    print(f"Saved {out_path}")

# ---- Save whatever was never claimed as the background layer ----
# Uses the ORIGINAL image pixels, not the inpainted working copy.
background_rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
background_rgba[..., :3] = image_rgb
background_rgba[..., 3] = np.where(~claimed_mask, 255, 0).astype(np.uint8)
cv2.imwrite(f"{output_dir}/background.png", cv2.cvtColor(background_rgba, cv2.COLOR_RGBA2BGRA))
print(f"Saved {output_dir}/background.png")