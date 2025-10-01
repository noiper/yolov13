from ultralytics import YOLO

model = YOLO('yolov13n.pt')
model.predict()
