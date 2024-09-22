import cv2
alg="haarcascade_frontalface_default.xml"  #face detection alg is here
haar_cascade=cv2.CascadeClassifier(alg)  
cam=cv2.VideoCapture(0)      #vid capture in our default camera so 0
while True:
	_,img = cam.read()   # img is reading 

	grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # the img color is changing to gray

	face = haar_cascade.detectMultiScale(grayImg,1.3,4)  #that gray image is giving as input here

	for(x,y,w,h) in face:   
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),5)
	cv2.imshow("FaceDetection",img)
	key= cv2.waitKey(10)
	if key == 27: # Esc key to stop
		break
cam.release()
cv2.destroyAllWindows()

#x and y are the coordinates of the top-left corner of the rectangle.
#w is the width of the rectangle.
#h is the height of the rectangle.

#(x+w, y+h) specifies the bottom-right corner of the rectangle.
#x+w is the x-coordinate of the bottom-right corner (right edge of the rectangle).
#y+h is the y-coordinate of the bottom-right corner (bottom edge of the rectangle).
#(255, 255, 0) is the color of the rectangle in BGR format. In this case, it is a cyan color (blue + green).
#5 is the thickness of the rectangle's border.