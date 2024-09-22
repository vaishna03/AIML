import cv2
import numpy as np

def faceBox(faceNet, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxs

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Initialize the models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define the lists for age and gender predictions
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(20-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Female', 'Male']

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]

        face = frame[y1:y2, x1:x2]
        if face.size == 0:  # Skip processing if face is empty
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Age-Gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
