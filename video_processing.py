from object_detection import YoloModel
from PIL import Image
from tracker import CentroidTracker
import cv2

model = YoloModel("yolov5l.yaml", "yolov5l.pt")
tracker = CentroidTracker()

filename = "/media/ardalan/8074437274436A4C/Work/video_rules/drive/Screen direction/Example 3/video.mp4"

vid = cv2.VideoCapture(filename)


frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = vid.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('example_3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

  
while True:
      
    ret, frame = vid.read()

    if not ret:
        break

    pil_img = Image.fromarray(frame.astype('uint8'), 'RGB')

    result = model(pil_img)

    objects = tracker.update(result)

    for obj in objects.values():
        direction = obj.get_direction(8)
        text = f"ID-{obj.id} {direction}"
        center = obj.center
        cv2.putText(frame, text, (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (center[0], center[1]), 4, (0, 255, 0), -1)

  
    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
vid.release()
cv2.destroyAllWindows()