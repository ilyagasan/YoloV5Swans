from ultralytics import YOLO
import cvzone
import cv2
import os
import math


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


_images_path = 'Images'
list_names_images = os.listdir('Images/')

model = YOLO('../Yolo-weights/yolov8l.pt')

cap = cv2.VideoCapture('video/office.MOV')
while cap.isOpened():
    _, img = cap.read()
   

    results = model(img, show = False)
    for j in results:
        boxes = j.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = abs(x1 - x2), abs(y1 - y2)
            bbox = int(x1), int(y1), int(w), int(h)

            # confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            cvzone.cornerRect(img, bbox=bbox)

            cvzone.cornerRect(img,bbox)

            # classes

            cls = box.cls[0]

            cvzone.putTextRect(img, f'{conf}, {x1, y1}, {dict_classes[math.ceil(cls)]}', thickness=1,
                               pos=(max(0, x1 - 20), max(0, y1 - 20)), scale=0.8)
            print(conf)

    cv2.imshow('Yolo', img)
    cv2.waitKey(0)



