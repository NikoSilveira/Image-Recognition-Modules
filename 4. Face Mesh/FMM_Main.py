import cv2
import time
import FaceMeshModule as fmm

cap = cv2.VideoCapture("video.mp4")
pTime = 0

detector = fmm.FaceMeshDetector()

while True:
    success, img = cap.read()
    img, faces_list = detector.findFaceMesh(img)

    if len(faces_list) != 0:
        print(faces_list[0]) #Print num of faces

    #FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display in window

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'): #shut down with q
        break