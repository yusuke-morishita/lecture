# Using OpenCV
import cv2

# Initialize a face detector by OpenCV
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read an image
img = cv2.imread('lfw\Aaron_Guiel\Aaron_Guiel_0001.jpg')

# Convert color space
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector.detectMultiScale(img_g)

# Show detection results
x, y, w, h = faces[0]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('image', img)
cv2.waitKey(-1)

