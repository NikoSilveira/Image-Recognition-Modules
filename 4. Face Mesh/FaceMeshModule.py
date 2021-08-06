import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, static_mode=False, maxFaces=2, minDetectionCon=0.5, MinTrackCon=0.5):
        self.static_mode = static_mode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.MinTrackCon = MinTrackCon

        #Init face mesh vars
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.maxFaces, self.minDetectionCon, self.MinTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1) #Specs for mesh

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces_list = [] #List for each different face

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face_list = [] #List for face features

                for id, lm in enumerate(faceLms.landmark):
                    img_h, img_w, img_c = img.shape
                    x, y = int(lm.x * img_w), int(lm.y * img_h) #Transform normalized pos values into pixel values
                    face_list.append([id, x, y])

                faces_list.append(face_list)

        return img, faces_list


#### Dummy Main Function####

#Ignore if being imported into another file
#Copy main() contents to another file, add import of module and add to file

def main():
    cap = cv2.VideoCapture("video.mp4")
    pTime = 0

    detector = FaceMeshDetector()

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

if __name__ == "__main__":
    main()