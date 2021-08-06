import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        print('test')
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)

            #Custom draw
            bboxC = detection.location_data.relative_bounding_box
            img_h, img_w, img_c = img.shape
            bbox = int(bboxC.xmin * img_w), int(bboxC.ymin * img_h), int(bboxC.width * img_w), int(bboxC.height * img_h) #Transform normalized pos values into pixel values
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, str(round(float(detection.score[0]), 2)), (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display score value


    #FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display in window

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'): #shut down with q
        break