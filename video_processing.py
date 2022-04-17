from object_detection import YoloModel
from PIL import Image
from tracker import CentroidTracker
import cv2

model = YoloModel("yolov5l.yaml", "yolov5l.pt")
tracker = CentroidTracker()

filename = "video.mp4"

vid = cv2.VideoCapture(filename)
  
while True:
      
    ret, frame = vid.read()

    pil_img = Image.fromarray(frame.astype('uint8'), 'RGB')

    result = model(pil_img)

    objects = tracker.update(result)

    for obj in objects.values():
        direction = obj.get_direction(5)
        text = f"ID-{obj.id} {direction}"
        center = obj.center
        cv2.putText(frame, text, (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (center[0], center[1]), 4, (0, 255, 0), -1)

  
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break