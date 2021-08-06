import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #Init pose processing vars
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=False):
        lm_list = []  #Empty landmark list

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark): #Iterate through all landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)   #Transform landmark pos values into pixel values
                lm_list.append([id, cx, cy])            #CHOOSE HERE WHAT TO APPEND

                if draw: #Custom draw for circles. Default: False
                    cv2.circle(img, (cx, cy), 7, (255,0,0), cv2.FILLED)  #Custom draw for circles
                
        return lm_list
    


#### Dummy Main Function####

#Ignore if being imported into another file
#Copy main() contents to another file, add import of module and add to file

def main():
    cap = cv2.VideoCapture('video.mp4')
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lm_list = detector.findPosition(img)

        if len(lm_list) != 0:
            print(lm_list[14])
    
        #FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display in window

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'): #shut down with q
            break

if __name__ == "__main__":
    main()