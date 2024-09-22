import cv2
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    
    for (x, y, w, h) in face:
        center = (x + w // 2, y + h // 2)
        radius = int(0.5 * (w + h) * 0.5)
        cv2.circle(img, center, radius, (0, 0, 0), 5)
    
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27:  # Esc key to stop
        break

cam.release()
cv2.destroyAllWindows()
