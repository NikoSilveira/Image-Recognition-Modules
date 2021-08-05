import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0) #Camera

pTime = 0
cTime = 0

detector = htm.handDetector()
    
while True:
    success, img = cap.read() #Hand image
    img = detector.findHands(img)
    lm_list = detector.findPosition(img)

    if len(lm_list) != 0: #Check if list not empty
        print(lm_list[8])

    #FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display in window

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'): #shut down with q
        break