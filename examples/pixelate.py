import cv2
import neural_obfuscator as no

model = no.Obfuscator(method="pixelate")
img = cv2.imread("../assets/mila.jpg")
img_gdpr = model.obfuscate(img)
no.show_image(img_gdpr)
