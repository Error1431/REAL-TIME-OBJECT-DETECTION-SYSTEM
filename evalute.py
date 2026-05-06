"""
evaluate.py
Evaluates a trained YOLOv8 model on the test set and reports key metrics.

WHAT IS mAP (Mean Average Precision)?
--------------------------------------
The most important metric for object detection. Here's how it's calculated:

1. Precision: Of all boxes my model drew, what fraction were correct?
   Precision = True Positives / (True Positives + False Positives)

2. Recall: Of all real objects in the image, what fraction did I find?
   Recall = True Positives / (True Positives + False Negatives)

3. AP (Average Precision) for one class:
   Plot Precision vs Recall as you vary the confidence threshold.
   The area under that curve = AP. Range: 0–1 (1 = perfect).

4. mAP50: Average of AP across all classes, at IoU threshold 0.50.
   "Correct" = predicted box overlaps ground truth by ≥50%.

5. mAP50-95: Averaged at IoU thresholds from 0.50 to 0.95 (step 0.05).
   This is stricter — the box must overlap very precisely to count as correct.
   Industry standard for serious benchmarks (COCO uses this).

WHAT YOU'LL GET FROM THIS SCRIPT:
  - Per-class AP scores
  - Overall mAP50 and mAP50-95
  - A confusion matrix image (saved to disk)
  - A Precision-Recall curve image (saved to disk)
  
These outputs are perfect for your project report and interview demonstrations.

Run with:  python evaluate.py
"""

import os
from ultralytics import YOLO

# ====== CONFIGURATION ======
# Update this path after training to point to your best weights
BEST_WEIGHTS = "runs/train/custom_model_v1/weights/best.pt"
DATA_YAML    = "./dataset/data.yaml"
IMGSZ        = 640
DEVICE       = "mps"  # Use MPS for fast evaluation
SAVE_DIR     = "runs/evaluate"
# ===========================


def main():
    if not os.path.exists(BEST_WEIGHTS):
        print(f"ERROR: Weights not found at '{BEST_WEIGHTS}'.")
        print("  → Run 'python train.py' first to generate the trained weights.")
        return

    print(f"Loading model: {BEST_WEIGHTS}")
    model = YOLO(BEST_WEIGHTS)

    print(f"Running evaluation on TEST split...")
    print(f"  Data:   {DATA_YAML}")
    print(f"  Device: {DEVICE.upper()}\n")

    # model.val() runs inference on the validation/test set and computes all metrics
    # split="test" tells it to use the 'test' images, not 'val'
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        device=DEVICE,
        split="test",   # evaluate on the held-out test set (most honest result)
        save_json=True, # save predictions as COCO-format JSON
        plots=True,     # save confusion matrix, PR curve, F1 curve to disk
        project=SAVE_DIR,
        name="results",
        verbose=True,
    )

    # Print the headline numbers
    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"  mAP50:      {metrics.box.map50:.3f}   (target: >0.70 for a good model)")
    print(f"  mAP50-95:   {metrics.box.map:.3f}   (target: >0.50 for a good model)")
    print(f"  Precision:  {metrics.box.mp:.3f}")
    print(f"  Recall:     {metrics.box.mr:.3f}")
    print("=" * 52)
    print(f"\n  Charts saved to: {SAVE_DIR}/results/")
    print(f"  Open these files to include in your project report:")
    print(f"    - confusion_matrix.png")
    print(f"    - PR_curve.png")
    print(f"    - F1_curve.png")

    # Interpretation guide
    print("\n--- INTERPRETING YOUR RESULTS ---")
    map50 = metrics.box.map50
    if map50 >= 0.85:
        print("🟢 Excellent! Your model is production-quality.")
    elif map50 >= 0.70:
        print("🟡 Good! Solid model. Can improve with more data or more epochs.")
    elif map50 >= 0.50:
        print("🟠 Fair. Collect more annotated images (especially hard cases).")
    else:
        print("🔴 Low accuracy. Check your annotations for errors, add more data.")


if __name__ == "__main__":
    main()
