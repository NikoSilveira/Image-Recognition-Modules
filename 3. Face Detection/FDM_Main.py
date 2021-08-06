import cv2
import time
import FaceDetectionModule as fdm

cap = cv2.VideoCapture("video.mp4")
pTime = 0

detector = fdm.FaceDetector()

while True:
    success, img = cap.read()
    img, bboxes_list = detector.findFaces(img)

    #FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display in window

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'): #shut down with q
        break