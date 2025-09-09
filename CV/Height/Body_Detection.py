""" # from _typeshed import SupportsWrite
import cv2 as cv
import mediapipe as mp
from playsound import playsound
import numpy as np
import pyttsx3
import pygame
import time
import math
from numpy.lib import utils
mpPose = mp.solutions.pose
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
mpDraw = mp.solutions.drawing_utils
drawing = mpDraw.DrawingSpec(thickness = 1 , circle_radius = 1)
pose = mpPose.Pose()
capture = cv.VideoCapture(0)
lst=[]
n=0
scale = 3
ptime = 0
count = 0
brake = 0
x=150
y=195
def speak(audio):

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate',150)

    engine.setProperty('voice', voices[0].id)
    engine.say(audio)

    # Blocks while processing all the currently
    # queued commands
    engine.runAndWait()
speak("I am about to measure your height now sir")
speak("Although I reach a precision upto ninety eight percent")
while True:
    isTrue,img = capture.read()
    img_rgb = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(result.pose_landmarks.landmark):
            lst[n] = lst.append([id,lm.x,lm.y])
            n+1
            # print(lm.z)
            # if len(lst)!=0:
            #     print(lst[3])
            h , w , c = img.shape
            if id == 32 or id==31 :
                cx1 , cy1 = int(lm.x*w) , int(lm.y*h)
                cv.circle(img,(cx1,cy1),15,(0,0,0),cv.FILLED)
                d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
                # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)
                di = round(d*0.5)
                pygame.mixer.init()
                pygame.mixer.music.load("check.mp3")
                pygame.mixer.music.play()
                speak(f"You are {di} centimeters tall")
                speak("I am done")
                speak("You can relax now")
                speak("Press q and give me some rest now.")
                if ord('q'):
                    cv.cv.destroyAllWindows()
                break
                dom = ((lm.z-0)**2 + (lm.y-0)**2)**0.5
                # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)

                cv.putText(img ,"Height : ",(40,70),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0),thickness=2)
                cv.putText(img ,str(di),(180,70),cv.FONT_HERSHEY_DUPLEX,1,(255,255,0),thickness=2)
                cv.putText(img ,"cms" ,(240,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
                cv.putText(img ,"Stand atleast 3 meter away" ,(40,450),cv.FONT_HERSHEY_PLAIN,2,	(0,0,255),thickness=2)
            
                # cv.putText(img ,"Go back" ,(240,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
            if id == 6:
                cx2 , cy2 = int(lm.x*w) , int(lm.y*h)
                # cx2 = cx230
                cy2 = cy2 + 20
                cv.circle(img,(cx2,cy2),15,(0,0,0),cv.FILLED)
    img = cv.resize(img , (700,500))
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv.putText(img , "FPS : ",(40,30),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
    cv.putText(img , str(int(fps)),(160,30),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
    cv.imshow("Task",img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows() """
""" 
# body_detection.py
import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

def measure_height_from_webcam(cam_index=0):
    capture = cv.VideoCapture(cam_index)
    height_cm = None
    ptime = 0

    while True:
        isTrue, img = capture.read()
        img_rgb = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            h, w, _ = img.shape
            cx1 = cy1 = cx2 = cy2 = None

            for id, lm in enumerate(result.pose_landmarks.landmark):
                if id in [31, 32]:  # left/right ankle
                    cx1, cy1 = int(lm.x*w), int(lm.y*h)
                if id == 6:  # nose/upper ref
                    cx2, cy2 = int(lm.x*w), int(lm.y*h)

            if cx1 and cx2:
                d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
                height_cm = round(d * 0.5)  # crude scale
                cv.putText(img, f"Height: {height_cm} cm", (40,70),
                           cv.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

        img = cv.resize(img , (700,500))
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img , f"FPS: {int(fps)}", (40,30),
                   cv.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

        cv.imshow("Height Detection", img)

        if cv.waitKey(20) & 0xFF == ord('q') and height_cm:
            break

    capture.release()
    cv.destroyAllWindows()
    return height_cm

if __name__ == "__main__":
    print("Measured Height:", measure_height_from_webcam())
 """

""" 
import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()


def measure_height_from_webcam(cam_index=0):
   
    capture = cv.VideoCapture(cam_index)
    height_cm = None
    ptime = 0

    while True:
        ok, img = capture.read()
        if not ok:
            break

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            h, w, _ = img.shape
            cx1 = cy1 = cx2 = cy2 = None

            for id, lm in enumerate(result.pose_landmarks.landmark):
                if id in [31, 32]:  # left/right ankle
                    cx1, cy1 = int(lm.x * w), int(lm.y * h)
                if id == 6:  # nose
                    cx2, cy2 = int(lm.x * w), int(lm.y * h)

            if cx1 and cx2:
                d = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
                height_cm = round(d * 0.5)  # crude scale factor
                cv.putText(
                    img,
                    f"Height: {height_cm} cm",
                    (40, 70),
                    cv.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

            mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Show FPS
        img = cv.resize(img, (700, 500))
        ctime = time.time()
        fps = 1 / (ctime - ptime) if ptime else 0
        ptime = ctime
        cv.putText(img, f"FPS: {int(fps)}", (40, 30),
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv.imshow("Height Detection", img)

        # Exit with 'q' once a height is detected
        if cv.waitKey(20) & 0xFF == ord("q") and height_cm:
            break

    capture.release()
    cv.destroyAllWindows()
    return height_cm


if __name__ == "__main__":
    print("Measured Height:", measure_height_from_webcam())
 """


# Height-Detection/Body_Detection.py
import cv2 as cv
import mediapipe as mp
import time
import numpy as np

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

def measure_height_from_webcam(cam_index=0, tolerance=2, stable_time=5):
    capture = cv.VideoCapture(cam_index)
    height_cm = None
    ptime = 0

    stable_start = None
    last_heights = []

    while True:
        isTrue, img = capture.read()
        if not isTrue:
            break

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        current_height = None
        if result.pose_landmarks:
            h, w, _ = img.shape
            cx1 = cy1 = cx2 = cy2 = None

            for id, lm in enumerate(result.pose_landmarks.landmark):
                if id in [31, 32]:  # ankle
                    cx1, cy1 = int(lm.x*w), int(lm.y*h)
                if id == 6:  # nose
                    cx2, cy2 = int(lm.x*w), int(lm.y*h)

            if cx1 and cx2:
                d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
                current_height = round(d * 0.5)  # crude scaling
                cv.putText(img, f"Height: {current_height} cm", (40,70),
                           cv.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

        # --- Stabilization logic ---
        if current_height:
            last_heights.append(current_height)
            if len(last_heights) > 30:  # keep last ~1 sec
                last_heights.pop(0)

            avg_height = np.mean(last_heights)
            fluctuation = np.max(last_heights) - np.min(last_heights)

            if fluctuation <= tolerance:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_time:
                    height_cm = round(avg_height, 1)
                    print(f"✅ Height stabilized: {height_cm} cm")
                    break
            else:
                stable_start = None  # reset if unstable

        # FPS display
        img = cv.resize(img, (700,500))
        ctime = time.time()
        fps = 1/(ctime-ptime) if ptime else 0
        ptime = ctime
        cv.putText(img, f"FPS: {int(fps)}", (40,30),
                   cv.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

        cv.imshow("Height Detection", img)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
    return height_cm

if __name__ == "__main__":
    print("Measured Height:", measure_height_from_webcam())
