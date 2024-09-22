import cv2
from PIL import Image
import pytesseract
from googletrans import Translator


image_path = input("Enter the path to the image: ")

image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load the image.")
    exit()

cv2.imshow("Image", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

pil_image = Image.fromarray(thresholded_image)


text = pytesseract.image_to_string(pil_image, lang='eng')  

print("Recognized Text: ")
print(text)
print()
print('Tamil-ta','Hindi-hi','Malayalam-ml','Arabic-ar')
option=input("Choose Any Language you preferred : ")

translator = Translator()

text_to_translate = text

if option=="ta":
    lang="ta"
elif option=="hi":
    lang="hi"
elif option=="ml":
    lang="ml"
elif option=="ar":
    lang="ar"
else:
    lang="en"
translated_text = translator.translate(text_to_translate, dest=lang)

print("Translated Text: ")
print(translated_text.text)
