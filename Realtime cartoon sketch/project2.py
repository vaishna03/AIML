import cv2
import numpy as np

def find_dominant_color(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for the hue channel
    hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])

    # Find the index of the maximum value in the histogram
    dominant_hue = np.argmax(hist)

    # Create an HSV color with the dominant hue
    dominant_color_hsv = np.uint8([[[dominant_hue, 255, 255]]])

    # Convert the HSV color to RGB
    dominant_color_rgb = cv2.cvtColor(dominant_color_hsv, cv2.COLOR_HSV2BGR)[0][0]

    # Convert the RGB color to hex format
    dominant_color_hex = '#{:02x}{:02x}{:02x}'.format(int(dominant_color_rgb[0]),
                                                      int(dominant_color_rgb[1]),
                                                      int(dominant_color_rgb[2]))

    # Print the dominant color in RGB and hex format
    print(f"Dominant Color (RGB): {dominant_color_rgb}")
    print(f"Dominant Color (Hex): {dominant_color_hex}")

    # Optionally, display the dominant color
    dominant_color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dominant_color_image[:] = dominant_color_rgb
    cv2.imshow('Dominant Color', dominant_color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
find_dominant_color('G:/python_aiml/photo.jpg')
