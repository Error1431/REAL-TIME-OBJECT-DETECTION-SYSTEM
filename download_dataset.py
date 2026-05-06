"""
download_dataset.py
Downloads your custom dataset from Roboflow directly into the project.

STEP-BY-STEP ROBOFLOW WORKFLOW
================================

STEP 1: Create a Roboflow Account
  → Go to: https://roboflow.com  →  Sign Up (free)

STEP 2: Create a New Project
  → Click "Create New Project"
  → Project Name: e.g., "face-mask-detection" or your use case
  → Project Type: Object Detection
  → Annotation Group: whatever suits (e.g., "batch-1")

STEP 3: Upload Images
  You need at least 100 images for a decent model. 200–500 is better.
  Ways to collect images:
    a) Take photos yourself with your webcam (variety of angles, lighting)
    b) Download from Google Images (use browser extension: "Download All Images")
    c) Pull from free datasets: Open Images, Kaggle, COCO subsets
    d) Roboflow Universe: https://universe.roboflow.com — find a similar dataset

  Upload them to Roboflow via the web UI.

STEP 4: Annotate (Label) Images
  Roboflow's annotation tool (Annotate tab):
    - Click an image
    - Draw bounding boxes around each object
    - Assign class names (e.g., "mask", "no-mask")
  
  TIPS for good annotations:
    - Draw tight boxes — don't include too much background
    - Annotate EVERY instance in the image (missing objects = bad training signal)
    - Aim for consistent box tightness
    - Include varied: angles, distances, lighting, backgrounds

  Labeling shortcut: if your dataset is large, try Roboflow's
  "Label Assist" (AI auto-labeling) — it pre-labels with a base model,
  you just correct mistakes. MUCH faster.

STEP 5: Generate Dataset
  → Click "Generate New Version"
  → Add Augmentations (VERY IMPORTANT for small datasets):
      - Flip: Horizontal ✅
      - Rotation: ±15° ✅
      - Brightness: ±25% ✅
      - Blur: up to 2px ✅
  → Augmentations triple your effective dataset size for free!
  → Split: 80% train / 10% valid / 10% test
  → Click "Generate"

STEP 6: Export / Get API Key
  → In your Project → Versions → click your version
  → Click "Export Dataset"
  → Format: YOLOv8 (Ultralytics)
  → "Show download code" → Copy the snippet
  → It looks like:
      rf = Roboflow(api_key="YOUR_KEY_HERE")
      project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
      version = project.version(1)
      dataset = version.download("yolov8")

STEP 7: Fill in the values below and run:
  python download_dataset.py
"""

from roboflow import Roboflow

# ====== FILL THESE IN FROM YOUR ROBOFLOW PROJECT ======
ROBOFLOW_API_KEY  = "YOUR_API_KEY_HERE"       # Found in: Roboflow → Settings → API Key
ROBOFLOW_WORKSPACE = "YOUR_WORKSPACE_NAME"    # e.g., "akash-rana-abc123"
ROBOFLOW_PROJECT   = "YOUR_PROJECT_NAME"      # e.g., "face-mask-detection"
ROBOFLOW_VERSION   = 1                        # Usually 1 for your first export
# ========================================================

def main():
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    print(f"Loading project: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    print("Downloading dataset in YOLOv8 format to ./dataset/ ...")
    dataset = version.download("yolov8", location="./dataset")

    print(f"\n✅ Dataset downloaded successfully!")
    print(f"   Location: ./dataset/")
    print(f"   data.yaml: ./dataset/data.yaml")
    print(f"\nNow run: python train.py")

if __name__ == "__main__":
    main()
