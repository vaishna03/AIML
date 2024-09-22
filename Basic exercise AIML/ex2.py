import cv2 
import imutils

img=cv2.imread('original.jpg')
resizedImg=imutils.resize(img,width=1000)

cv2.imshow('originalImage2.jpg',img)
cv2.imshow('resized.jpg',resizedImg)

cv2.imwrite('resizedImage12.jpg',resizedImg)
cv2.waitKey(5000)

cv2.destroyAllWindows()

