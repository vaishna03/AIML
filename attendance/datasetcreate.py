import cv2, os

# Path to Haar Cascade XML file
haar_file = 'haarcascade_frontalface_default.xml'

# Directory where datasets will be stored
datasets = 'datatsets'

# Sub-directory name (this will be the person's name)
sub_data = 'vaishu'  # Change 'person_name' to the name of the person

# Create a directory for the person's dataset
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# Set image size for face recognition
(width, height) = (130, 100)

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize webcam
webcam = cv2.VideoCapture(0)

count = 1
while count <= 50:  # Capture 50 images
    print(f'Capturing image {count}')
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        count += 1
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Press 'ESC' to exit early
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
