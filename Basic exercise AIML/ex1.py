import cv2 
img = cv2.imread("cinderella.jpg") 

cv2.imshow('show',img)

cv2.imwrite('mycinderella.jpg',img)

cv2.waitKey(5000)

cv2.destroyAllWindows()
