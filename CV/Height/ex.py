""" import cv2 as cv
import mediapipe as mp
import math
import pyttsx3

# Text-to-speech setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open camera
cap = cv.VideoCapture(0)

# Calibration factor (you must adjust for real-world height)
# Example: 1 pixel ≈ 0.5 cm (depends on camera distance + FOV)
PIXEL_TO_CM = 0.5

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    h, w, c = img.shape
    top_point = None
    bottom_point = None

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Nose = top reference
            if id == 0:
                top_point = (cx, cy)
                cv.circle(img, top_point, 8, (0, 255, 0), cv.FILLED)

            # Ankle (left=31, right=32) = bottom reference
            if id in [31, 32]:
                bottom_point = (cx, cy)
                cv.circle(img, bottom_point, 8, (0, 0, 255), cv.FILLED)

        # If both points detected → calculate height
        if top_point and bottom_point:
            d = math.dist(top_point, bottom_point)  # pixel distance
            di = round(d * PIXEL_TO_CM)  # convert to cm

            cv.putText(img, f"Height: {di} cm", (40, 70),
                       cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

            speak(f"You are {di} centimeters tall")

    cv.imshow("Height Detection", img)

    # Quit with 'q'
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
 """
""" 
import cv2 as cv
import mediapipe as mp
import math
import pyttsx3
import time
import csv
import os

# Text-to-speech setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open camera
cap = cv.VideoCapture(0)

# Calibration factor (adjust after calibration)
PIXEL_TO_CM = 0.5

# Tracking stability
last_height = None
stable_since = None
STABILITY_TIME = 5  # seconds required for no fluctuation

# CSV setup
CSV_FILE = "heights_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Height (cm)"])  # header row

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    h, w, c = img.shape
    top_point = None
    bottom_point = None

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Nose = top reference
            if id == 0:
                top_point = (cx, cy)
                cv.circle(img, top_point, 8, (0, 255, 0), cv.FILLED)

            # Ankle (left=31, right=32) = bottom reference
            if id in [31, 32]:
                bottom_point = (cx, cy)
                cv.circle(img, bottom_point, 8, (0, 0, 255), cv.FILLED)

        # If both points detected → calculate height
        if top_point and bottom_point:
            d = math.dist(top_point, bottom_point)  # pixel distance
            di = round(d * PIXEL_TO_CM)  # convert to cm

            cv.putText(img, f"Height: {di} cm", (40, 70),
                       cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

            # Check for stability (within ±2 cm)
            if last_height is None or abs(di - last_height) > 2:
                last_height = di
                stable_since = time.time()
            else:
                if stable_since and (time.time() - stable_since >= STABILITY_TIME):
                    print(f"Final Height = {last_height} cm")
                    speak(f"Your final height is {last_height} centimeters")

                    # Log to CSV
                    with open(CSV_FILE, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), last_height])

                    # Reset so it doesn't spam
                    stable_since = None

    cv.imshow("Height Detection", img)

    # Quit with 'q'
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
 """

""" import cv2 as cv
import mediapipe as mp
import math
import pyttsx3
import csv
import time

# Text-to-speech setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open camera
cap = cv.VideoCapture(0)

# Calibration factor (adjust for real-world use)
PIXEL_TO_CM = 0.5  

# Stability tracking
stable_height = None
stable_start = None
STABILITY_TIME = 5  # seconds

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    h, w, c = img.shape
    top_point = None
    bottom_point = None

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Nose = top reference
            if id == 0:
                top_point = (cx, cy)
                cv.circle(img, top_point, 8, (0, 255, 0), cv.FILLED)

            # Ankle (left=31, right=32) = bottom reference
            if id in [31, 32]:
                bottom_point = (cx, cy)
                cv.circle(img, bottom_point, 8, (0, 0, 255), cv.FILLED)

        # If both points detected → calculate height
        if top_point and bottom_point:
            d = math.dist(top_point, bottom_point)  # pixel distance
            di = round(d * PIXEL_TO_CM)  # convert to cm

            cv.putText(img, f"Height: {di} cm", (40, 70),
                       cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

            # Check stability
            if stable_height is None or abs(di - stable_height) > 2:  
                # reset stability tracking
                stable_height = di
                stable_start = time.time()
            else:
                # If stable for required time
                if time.time() - stable_start >= STABILITY_TIME:
                    final_height = stable_height
                    print(f"Final Height: {final_height} cm")

                    # Log into CSV
                    with open("height_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), final_height])

                    # Speak once
                    speak(f"You are {final_height} centimeters tall")

                    # Exit program after logging
                    break

    cv.imshow("Height Detection", img)

    # Quit with 'q'
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows() """

import cv2 as cv
import mediapipe as mp
import math
import pyttsx3
import csv
import time
import os

# Text-to-speech setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open camera
cap = cv.VideoCapture(0)

# Calibration factor (adjust for real-world use)
PIXEL_TO_CM = 0.5  

# Stability tracking
stable_height = None
stable_start = None
STABILITY_TIME = 5  # seconds

# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    h, w, c = img.shape
    top_point = None
    bottom_point = None

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Nose = top reference
            if id == 0:
                top_point = (cx, cy)
                cv.circle(img, top_point, 8, (0, 255, 0), cv.FILLED)

            # Ankle (left=31, right=32) = bottom reference
            if id in [31, 32]:
                bottom_point = (cx, cy)
                cv.circle(img, bottom_point, 8, (0, 0, 255), cv.FILLED)

        # If both points detected → calculate height
        if top_point and bottom_point:
            d = math.dist(top_point, bottom_point)  # pixel distance
            di = round(d * PIXEL_TO_CM)  # convert to cm

            cv.putText(img, f"Height: {di} cm", (40, 70),
                       cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

            # Check stability
            if stable_height is None or abs(di - stable_height) > 2:  
                # reset stability tracking
                stable_height = di
                stable_start = time.time()
            else:
                # If stable for required time
                if time.time() - stable_start >= STABILITY_TIME:
                    final_height = stable_height
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

                    print(f"Final Height: {final_height} cm at {timestamp}")

                    # Log into CSV
                    with open("logs/height_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, final_height])

                    # Save screenshot
                    screenshot_path = f"screenshots/height_{timestamp}.png"
                    cv.imwrite(screenshot_path, img)
                    print(f"Screenshot saved: {screenshot_path}")

                    # Speak once
                    speak(f"You are {final_height} centimeters tall")

                    # Exit program after logging
                    break

    cv.imshow("Height Detection", img)

    # Quit with 'q'
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
