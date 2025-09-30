from ultralytics import YOLO

model = YOLO('yolov13s.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=128, 
  imgsz=640,
  scale=0.9,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.05,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.15,  # S:0.15; L:0.5; X:0.6
  workers = 16,
  device="0",
)

# Evaluate model performance on the validation set
# metrics = model.val('coco.yaml')

# # Perform object detection on an image
# results = model("path/to/your/image.jpg")
# results[0].show()
