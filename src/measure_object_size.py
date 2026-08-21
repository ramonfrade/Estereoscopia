import cv2
from object_detector import * # HomogeneousBgDetector

img = cv2.imread('phone.jpg')

detector = HomogeneousBgDetector()

cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

