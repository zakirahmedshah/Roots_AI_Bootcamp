from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("yolov8n.pt")

# Image path (make sure test.jpg exists)
image_path = "test.jpg"

# Run prediction
results = model(image_path)

# Get annotated image
annotated_img = results[0].plot()

# Save in SAME folder as script
save_path = os.path.join(os.getcwd(), "output.jpg")
cv2.imwrite(save_path, annotated_img)

print("Saved at:", save_path)
print("Check your folder now for output.jpg")