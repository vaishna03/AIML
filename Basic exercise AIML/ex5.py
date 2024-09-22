import cv2
img = cv2.imread('logo.jpg')

smoothImg1 = cv2.GaussianBlur(img,(41,41),0)


cv2.imshow("original",img)
cv2.imshow("SmoothImg1",img)





