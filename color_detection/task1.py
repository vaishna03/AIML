import cv2
import numpy as np
import pandas as pd

def load_color_dataset(file_path):
    var = pd.read_csv(file_path)
    return var

def detect_color(frame, color_dataset):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = []

    for index, row in color_dataset.iterrows():
        try:
            lower_bound = np.array([row['h_min'], row['s_min'], row['v_min']])
            upper_bound = np.array([row['h_max'], row['s_max'], row['v_max']])
        except KeyError as e:
            print(f"Missing column in dataset: {e}")
            continue

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                radius = int(0.5 * (w + h))
                
                detected_colors.append((row['color_name'], (cx, cy, radius)))

    return detected_colors

def main():
    color_dataset = load_color_dataset("C:/Users/vaishna/Downloads/color1.csv")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        detected_objects = detect_color(frame, color_dataset)
        
        for color_name, (cx, cy, radius) in detected_objects:
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(frame, color_name, (cx - 10, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
