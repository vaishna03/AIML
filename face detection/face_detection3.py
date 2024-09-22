import cv2

# Load the Haar cascade file
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
# Start capturing video from the default camera (usually the webcam)
cam = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    _, img = cam.read()
    
    # Convert the frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    
    # Draw triangles around the detected faces
    for (x, y, w, h) in faces:
        # Calculate the coordinates of the triangle vertices
        pt1 = (x + w // 2, y)  # Top middle
        pt2 = (x, y + h)       # Bottom left
        pt3 = (x + w, y + h)   # Bottom right

        # Draw the triangle
        cv2.line(img, pt1, pt2, (255, 255, 0), 5)
        cv2.line(img, pt2, pt3, (255, 255, 0), 5)
        cv2.line(img, pt3, pt1, (255, 255, 0), 5)
    
    # Display the frame with triangles
    cv2.imshow("FaceDetection", img)
    
    # Break the loop if the ESC key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
