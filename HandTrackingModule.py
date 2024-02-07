import cv2
import mediapipe as mp
import os
import time

class handDetector():
    def __init__(self, img, scale = 1, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        #metadata
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.maxHands) #self.mode, self.maxHands, self.detectionCon, self.trackCon
        self.mpDraw = mp.solutions.drawing_utils

        height, width, _ = img.shape
        self.new_height = int(height*scale)      
        self.new_width = int(self.new_height * width / height)
        print(self.new_height, self.new_width)
        
    #draws hand locations on the image
    def findHands(self, img, draw = True):
        img = cv2.resize(img, (self.new_width, self.new_height))

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    #returns absolute coordinates of the hands
    def findPosition(self, img, handNo = 0, draw = True):
        #landmarks
        lmList = []

        #find landmarks for both hands
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x*self.new_width), int(lm.y*self.new_height)

                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

def main(): 
    pTime = 0
    cTime = 0

    #meta data
    path = "IMG_2106.mp4"
    cap = cv2.VideoCapture(os.path.join(os.getcwd(), path))
    success, img = cap.read()
    detector = handDetector(img, 0.4)

    while True:
        success, img = cap.read()

        #exit when video ends
        if not success:
            break

        img = detector.findHands(img) 
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])

        #display fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        #show image
        cv2.imshow("image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()