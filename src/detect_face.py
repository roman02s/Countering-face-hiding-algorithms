import cv2
import numpy as np
# This function detects faces and returns the cropped face


def detect_face(img):
    # Convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    # Let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # If no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    # Under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]
    # Return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]
