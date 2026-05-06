import cv2
from ultralytics import YOLO

def main():
    # 1. Load the model
    # We use YOLOv8n (nano), the smallest and fastest model. 
    # The '.pt' extension means it's a PyTorch model weights file.
    model = YOLO("yolov8n.pt")
    
    # Configuration to ensure we use Apple Silicon (MPS)
    device = "mps" 

    # 2. Run Inference
    # We pass the image path and device.
    # YOLO returns a list of Results objects (one per image).
    print("Running inference on test_image.jpg...")
    results = model("test_image.jpg", device=device)
    
    # 3. Analyze the output
    # Since we passed one image, we get the first item
    result = results[0] 

    print("\n--- INFERENCE RESULTS ---")
    
    # 'result.boxes' contains the bounding box objects
    boxes = result.boxes
    
    print(f"Number of objects detected: {len(boxes)}")
    
    # Loop through every detected object and read its data
    for i, box in enumerate(boxes):
        print(f"\nObject {i+1}:")
        
        # 1. Coordinates (x1, y1, x2, y2)
        # xyxy means top-left config (x1, y1) and bottom-right config (x2, y2)
        coords = box.xyxy[0].tolist() 
        print(f"  Coordinates (x1,y1,  x2,y2): {coords}")
        
        # 2. Confidence Score
        # A percentage representing how sure the model is (0.0 to 1.0)
        confidence = box.conf[0].item()
        print(f"  Confidence Score: {confidence:.2f} ({confidence*100:.0f}%)")
        
        # 3. Class Index & Name
        # The class property returns an integer index.
        # We can look up the actual string name in the model's 'names' dictionary.
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        print(f"  Class detected: ID {class_id} -> '{class_name}'")

    # 4. Save a marked-up image to disk
    # YOLO provides a built-in method to plot the boxes on the image
    annotated_frame = result.plot()
    
    cv2.imwrite("test_image_output.jpg", annotated_frame)
    print("\n✅ Saved annotated image to 'test_image_output.jpg'")

if __name__ == "__main__":
    main()
