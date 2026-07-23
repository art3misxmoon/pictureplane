# pip3 install segment-anything torch torchvision (done!)
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# downloaded this model checkpoint (default)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# choose image and set up
image = cv2.imread("womanleadinghorse.png")

# set up rgb colorspace
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# point predictor to rgb image
predictor.set_image(image_rgb)

# make copy to work with
working_image = image_rgb.copy()
# list of layers
layers = []
current_mask = None




# ---- Mouse callback ----
def on_click(event, x, y, flags, param):
    global current_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([[x, y]])
        label = np.array([1])  # foreground click
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True
        )
        current_mask = masks[np.argmax(scores)]
        show_preview()

def show_preview():
    """Overlay the current mask on the working image in red."""
    display = cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR).copy()
    if current_mask is not None:
        overlay = display.copy()
        overlay[current_mask] = (0, 0, 255)  # red highlight, BGR
        display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)
    cv2.imshow("SAM Layer Picker", display)

# ---- Window setup ----
cv2.namedWindow("SAM Layer Picker")
cv2.setMouseCallback("SAM Layer Picker", on_click)
show_preview()

print("Click an object to preview its mask.")
print("  [SPACE] = accept mask as next layer, remove it, and continue")
print("  [r]     = reset current mask preview (click again)")
print("  [q]     = quit and save results")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord(' '):  # accept current mask as a layer
        if current_mask is not None:
            layer_rgba = np.zeros((*working_image.shape[:2], 4), dtype=np.uint8)
            layer_rgba[..., :3] = working_image
            layer_rgba[..., 3] = (current_mask * 255).astype(np.uint8)
            layers.append(layer_rgba)

            # Remove the extracted object so the next click hits what's beneath.
            # Simple version: zero it out. See note below about inpainting.
            working_image[current_mask] = 0
            predictor.set_image(working_image)

            current_mask = None
            show_preview()
            print(f"Layer {len(layers)} saved. {len(layers)} total so far.")
        else:
            print("No mask selected yet — click an object first.")

    elif key == ord('r'):
        current_mask = None
        show_preview()

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# ---- Save layers to disk ----
for i, layer in enumerate(layers):
    out_path = f"output/layer_{i:02d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA))
    print(f"Saved {out_path}")