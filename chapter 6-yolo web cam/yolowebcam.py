from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
"dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair dryer", "toothbrush"]
numbers = [i for i in range(0,len(class_names))]

dict_classes = dict(zip(numbers, class_names))
while True:
    success, img = cap.read()

    model = YOLO(model='../Yolo-weights/yolov8l.pt')
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes

        for box in boxes:

            #bounding box

            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(img, (x1,y1),(x2,y2), (0,200,200), 3)
            # cv2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = abs(x1 - x2), abs(y1 - y2)
            bbox = int(x1), int(y1), int(w), int(h)

            #confidence
            conf = math.ceil(box.conf[0]*100)/100
            cvzone.cornerRect(img, bbox=bbox)


            #classes

            cls = box.cls[0]


            cvzone.putTextRect(img, f'{conf}, {x1, y1}, {dict_classes[math.ceil(cls)]}', thickness=1, pos=(max(0, x1-20), max(0, y1-20)), scale=0.8)

            print(conf)

            # cvzone

    cv.imshow('Image', img)

    cv.waitKey(1)

