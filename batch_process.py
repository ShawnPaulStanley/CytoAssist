"""
CytoAssist Batch Image Processor

Monitors a folder for new images, runs them through the ResNet18 model,
and saves predictions + Grad-CAM heatmaps to an output folder.

Usage:
    python batch_process.py                     # Process once and exit
    python batch_process.py --watch             # Continuously monitor folder
    python batch_process.py --input ./my_images --output ./my_results
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# -----------------------------
# Constants
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["Benign", "Suspicious"]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Default paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "backend", "cytoassist_resnet18.pth")
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "batch_input")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "batch_output")


# -----------------------------
# Logging utilities
# -----------------------------
def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


# -----------------------------
# Model loading
# -----------------------------
def load_model(model_path: str) -> nn.Module:
    """Load a ResNet18 model on CPU.

    The checkpoint may be:
    - a state_dict
    - a dict containing 'state_dict'
    - a serialized nn.Module
    """
    log_section("MODEL LOADING")
    log(f"Loading model from: {model_path}")
    device = torch.device("cpu")

    # Build architecture (ResNet18) with a single sigmoid output
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid(),
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # If a whole module was saved, use it directly
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        # Extract possible state_dict
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Strip 'module.' prefix from DataParallel checkpoints
        if isinstance(state_dict, dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k
                if key.startswith("module."):
                    key = key[len("module."):]
                new_state_dict[key] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    log("Model loaded successfully ✓")
    return model


# -----------------------------
# Preprocessing
# -----------------------------
def get_preprocess() -> transforms.Compose:
    """Preprocessing required by the trained ResNet18."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------
# Prediction
# -----------------------------
def predict_with_probs(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float, float, int]:
    """Run CPU inference and return (benign_prob, suspicious_prob, predicted_index)."""
    device = torch.device("cpu")
    with torch.no_grad():
        suspicious_prob = float(model(input_tensor.to(device)).item())

    benign_prob = 1.0 - suspicious_prob
    pred_idx = 1 if suspicious_prob >= 0.5 else 0
    return benign_prob, suspicious_prob, pred_idx


# -----------------------------
# Grad-CAM implementation
# -----------------------------
class GradCAM:
    """Minimal Grad-CAM for ResNet-like models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: capture activations
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

        # Backward hook: capture gradients
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
        else:
            self._bwd_handle = self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        """Remove hooks to avoid side effects."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Generate a Grad-CAM heatmap."""
        device = torch.device("cpu")
        input_tensor = input_tensor.to(device)

        # Forward
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        # Backward target for single-output sigmoid model
        score = logits.sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Global-average-pool gradients to get channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=False)

        # ReLU and normalize to [0, 1]
        cam = F.relu(cam)
        cam = cam[0]  # remove batch
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy()


def apply_colormap_and_overlay(
    base_image_rgb: Image.Image,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a heatmap (values in [0,1]) on top of an RGB PIL image."""
    import matplotlib.cm as cm

    base_np = np.array(base_image_rgb).astype(np.float32) / 255.0

    # Resize heatmap to image size
    heatmap_img = Image.fromarray((heatmap_01 * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(base_image_rgb.size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    # Apply colormap (jet) -> RGBA -> RGB
    cmap = cm.get_cmap("jet")
    colored = cmap(heatmap_resized)[:, :, :3]  # drop alpha

    # Blend
    overlay = (1.0 - alpha) * base_np + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# -----------------------------
# Batch processing logic
# -----------------------------
def get_pending_images(input_dir: str, processed_set: set) -> List[str]:
    """Get list of image files that haven't been processed yet."""
    pending = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS and filename not in processed_set:
            pending.append(filename)
    return sorted(pending)


def process_single_image(
    image_path: str,
    model: nn.Module,
    preprocess: transforms.Compose,
    output_dir: str,
) -> dict:
    """Process a single image and save results."""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    log(f"Processing: {filename}")

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    # Run prediction
    benign_prob, suspicious_prob, pred_idx = predict_with_probs(model, input_tensor)
    prediction = CLASS_NAMES[pred_idx]

    log(f"  → {prediction} (Suspicious: {suspicious_prob*100:.1f}%, Benign: {benign_prob*100:.1f}%)")

    # Generate Grad-CAM
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model=model, target_layer=target_layer)
    heatmap = cam.generate(input_tensor, pred_idx)
    cam.close()

    # Create overlay
    overlay = apply_colormap_and_overlay(img, heatmap)

    # Save heatmap overlay
    heatmap_filename = f"{name_without_ext}_heatmap.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    overlay.save(heatmap_path)
    log(f"  → Saved heatmap: {heatmap_filename}")

    return {
        "filename": filename,
        "prediction": prediction,
        "benign_prob": benign_prob,
        "suspicious_prob": suspicious_prob,
        "heatmap_file": heatmap_filename,
        "timestamp": datetime.now().isoformat(),
    }


def append_to_csv(output_dir: str, result: dict):
    """Append a result row to results.csv."""
    csv_path = os.path.join(output_dir, "results.csv")
    file_exists = os.path.exists(csv_path)

    fieldnames = ["filename", "prediction", "benign_prob", "suspicious_prob", "heatmap_file", "timestamp"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_batch(
    input_dir: str,
    output_dir: str,
    model: nn.Module,
    preprocess: transforms.Compose,
    processed_set: set,
) -> int:
    """Process all pending images in the input directory. Returns count processed."""
    pending = get_pending_images(input_dir, processed_set)
    
    if not pending:
        return 0

    log_section(f"PROCESSING {len(pending)} IMAGE(S)")

    for filename in pending:
        image_path = os.path.join(input_dir, filename)
        try:
            result = process_single_image(image_path, model, preprocess, output_dir)
            append_to_csv(output_dir, result)
            processed_set.add(filename)
        except Exception as e:
            log(f"ERROR processing {filename}: {e}")

    return len(pending)


def main():
    parser = argparse.ArgumentParser(
        description="CytoAssist Batch Image Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_process.py                          # Process once and exit
    python batch_process.py --watch                  # Continuously monitor folder
    python batch_process.py --watch --interval 5     # Check every 5 seconds
    python batch_process.py --input ./images --output ./results
        """,
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with images to process (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously watch input folder for new images",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Seconds between folder checks in watch mode (default: 3)",
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.input):
        log(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load model
    model = load_model(args.model)
    preprocess = get_preprocess()

    # Track processed files to avoid reprocessing
    processed_set: set = set()

    log_section("BATCH PROCESSOR READY")
    log(f"Input folder:  {args.input}")
    log(f"Output folder: {args.output}")
    log(f"Watch mode:    {'ON' if args.watch else 'OFF'}")

    if args.watch:
        log(f"Monitoring folder every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                count = run_batch(args.input, args.output, model, preprocess, processed_set)
                if count > 0:
                    log(f"Batch complete. {len(processed_set)} total images processed.")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log("\nStopping batch processor.")
    else:
        count = run_batch(args.input, args.output, model, preprocess, processed_set)
        if count == 0:
            log("No images found to process.")
        else:
            log_section("BATCH COMPLETE")
            log(f"Processed {count} image(s). Results saved to: {args.output}")


if __name__ == "__main__":
    main()
