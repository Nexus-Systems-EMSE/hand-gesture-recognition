import cv2
import mediapipe as mp
import numpy as np


class Finger:
    def __init__(self):
        self.mcp = 0
        self.dip = 0
        self.pip = 0

        self.tip_x = 0
        self.tip_y = 0
        self.tip_z = 0


THUMB = Finger()
INDEX = Finger()
MIDDLE_FINGER = Finger()
RING_FINGER = Finger()
PINKY = Finger()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image_hight, image_width, _ = image.shape
        # print("Image shape : ", image_width, " x ", image_hight)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                INDEX.tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                INDEX.tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight
                INDEX.tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                print('x = ', INDEX.tip_x, ' y = ', INDEX.tip_y)

        cv2.imshow('MediaPipe Hands', image)



cap.release()
