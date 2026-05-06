"""
train.py
Fine-tunes YOLOv8n on your custom Roboflow dataset.

WHY FINE-TUNING (not training from scratch)?
---------------------------------------------
YOLOv8n was pre-trained on COCO with 80 classes and 330k images.
It already "knows" what edges, shapes, textures, and objects look like.
Fine-tuning (Transfer Learning) takes those learned weights as the starting
point and adapts the final layers to YOUR classes.

Result:
  - You need WAY less data (100–500 images vs. 100,000+)
  - Training is WAY faster (minutes/hours vs. days/weeks)
  - Final accuracy is usually better than training from scratch on small data

HOW YOLOV8 TRAINING WORKS:
  1. For each batch of images, YOLO predicts bounding boxes + classes
  2. Loss = how far off the predictions are from the ground-truth labels
     (combines box loss + classification loss + DFL loss)
  3. Backpropagation adjusts weights to reduce loss
  4. Repeat for many epochs → weights improve

HYPERPARAMETERS (the values you control):
  - epochs: How many full passes through the training data
  - imgsz:  Input image size (640 = standard, smaller = faster but less accurate)
  - batch:  Images per training step (limited by your RAM; 8–16 for M1 8GB)
  - lr0:    Initial learning rate — how big each weight update step is
  - patience: For early stopping — stop if val loss doesn't improve for N epochs

Run with:  python train.py
"""

from ultralytics import YOLO
import yaml


# ====== CONFIGURATION ======
DATA_YAML   = "./dataset/data.yaml"  # Path to the dataset config file from Roboflow
MODEL       = "yolov8n.pt"           # Start from pretrained nano weights (best for M1 8GB)
EPOCHS      = 50                     # 50 is a good starting point for 200–500 images
IMGSZ       = 640                    # Standard YOLO input size
BATCH       = 8                      # Safe for 8GB RAM. Reduce to 4 if you get OOM errors.
DEVICE      = "mps"                  # Apple Silicon GPU. Use "cpu" if MPS causes issues.
PROJECT_DIR = "runs/train"           # Where results, weights, and charts are saved
RUN_NAME    = "custom_model_v1"      # Subfolder name for this training run
# ===========================


def print_dataset_info():
    """Print a summary of the dataset we're about to train on."""
    print("\n=== Dataset Info ===")
    try:
        with open(DATA_YAML, "r") as f:
            data = yaml.safe_load(f)
        print(f"  Classes: {data.get('nc', '?')} → {data.get('names', [])}")
        print(f"  Train:   {data.get('train', '?')}")
        print(f"  Val:     {data.get('val', '?')}")
    except FileNotFoundError:
        print("  ⚠️ data.yaml not found. Run download_dataset.py first!")
        exit(1)
    print("===================\n")


def main():
    print_dataset_info()
    
    print(f"Loading base model: {MODEL}")
    model = YOLO(MODEL)

    print(f"Starting fine-tuning on device: {DEVICE.upper()}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Img size: {IMGSZ}×{IMGSZ}")
    print(f"  Batch:    {BATCH}")
    print(f"  Output:   {PROJECT_DIR}/{RUN_NAME}/\n")

    # train() handles everything: data loading, augmentation, optimization, metrics.
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=15,    # Stop early if val mAP doesn't improve for 15 epochs
        save=True,      # Save best & last weights
        plots=True,     # Generate loss curves, confusion matrix, PR curve plots
        verbose=True,
    )

    # After training, the best weights are at:
    best_weights = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
    print(f"\n✅ Training complete!")
    print(f"   Best weights: {best_weights}")
    print(f"   Run 'python evaluate.py' to see mAP and confusion matrix.")
    print(f"   Then run 'python run_detection.py --model {best_weights}' to use them!")

    return results


if __name__ == "__main__":
    main()
