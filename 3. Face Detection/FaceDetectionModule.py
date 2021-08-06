import cv2
import mediapipe as mp
import time

class FaceDetector():

    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        #Init face detection vars
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxes_list = [] #Accepts more than 1 bbox for several faces

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                bboxC = detection.location_data.relative_bounding_box
                img_h, img_w, img_c = img.shape
                bbox = int(bboxC.xmin * img_w), int(bboxC.ymin * img_h), int(bboxC.width * img_w), int(bboxC.height * img_h) #Transform normalized pos values into pixel values (X, Y, W, H)

                bboxes_list.append([id, bbox, detection.score])

                cv2.rectangle(img, bbox, (255,0,255), 2) #Custom draw
                cv2.putText(img, str(round(float(detection.score[0]), 2)), (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,210,0), 2) #display score value

        return img, bboxes_list


#### Dummy Main Function####

#Ignore if being imported into another file
#Copy main() contents to another file, add import of module and add to file

def main():
    cap = cv2.VideoCapture("video.mp4")
    pTime = 0

    detector = FaceDetector()

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

if __name__ == "__main__":
    main()